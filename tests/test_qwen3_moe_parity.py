"""Qwen3-MoE custom forward vs HF Qwen3MoeForCausalLM, on a tiny random config (CPU, bf16).

The tiny model is saved to disk with a word-level tokenizer so the backend loads it through the
same `load_model` path a real checkpoint takes. The full Qwen3-30B-A3B check is `heavy` + CUDA.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from inference_server.models import qwen3_moe
from inference_server.models.gemma4 import KVCache
from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM, route

REAL_DIR = Path(__file__).parent / "fixtures" / "qwen3-30b-a3b"   # config.json + generation_config.json
from tests.conftest import TINY_EOS as EOS, TINY_VOCAB as VOCAB, tiny_qwen3_moe_cfg as _tiny_cfg  # noqa: E402

# Shared 6-token prefix; odd lengths so a repeat is a partial prefix hit. A FULL hit (prompt a
# multiple of the block size) re-feeds the last token in CustomTorchBackend.prefill() — known bug.
PROMPTS = [[5, 17, 300, 42, 8, 99, 123, 7, 55], [5, 17, 300, 42, 8, 99, 600]]


@pytest.fixture(scope="module")
def tiny_dir(tiny_qwen3_moe_dir):
    return tiny_qwen3_moe_dir


@pytest.fixture(scope="module")
def hf(tiny_dir):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def ours(tiny_dir):
    return Qwen3MoeForCausalLM.from_hf(tiny_dir).eval()


def _hf_greedy(hf, prompt, n):
    out = hf.generate(torch.tensor([prompt]), max_new_tokens=n, do_sample=False)[0, len(prompt):].tolist()
    return out[:out.index(EOS)] if EOS in out else out


# ---------------------------------------------------------------- routing

def test_route_picks_top_k_and_renormalises():
    logits = torch.tensor([[0.0, 2.0, 1.0, -1.0], [3.0, 0.0, 0.0, 3.0]])
    w, e = route(logits, top_k=2, norm_topk_prob=True)
    assert e[0].tolist() == [1, 2] and sorted(e[1].tolist()) == [0, 3]
    torch.testing.assert_close(w.sum(-1), torch.ones(2))
    p = torch.softmax(logits[0], -1)
    torch.testing.assert_close(w[0], p[[1, 2]] / p[[1, 2]].sum())
    w_raw, _ = route(logits, top_k=2, norm_topk_prob=False)
    torch.testing.assert_close(w_raw[0], p[[1, 2]])        # no renormalisation → raw softmax mass


def test_route_matches_hf_router():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
    cfg = _tiny_cfg()
    router = Qwen3MoeTopKRouter(cfg)
    torch.nn.init.normal_(router.weight)
    x = torch.randn(10, cfg.hidden_size)
    _, hf_w, hf_e = router(x)
    w, e = route(torch.nn.functional.linear(x, router.weight), cfg.num_experts_per_tok, True)
    assert torch.equal(e, hf_e) and torch.equal(w, hf_w)


def test_moe_block_matches_hf(hf, ours):
    x = torch.randn(2, 9, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(ours.model.layers[0].mlp(x), hf.model.layers[0].mlp(x))


# ---------------------------------------------------------------- full forward

def test_logits_match_hf(hf, ours):
    ids = torch.tensor(PROMPTS[0]).unsqueeze(0)
    with torch.no_grad():
        torch.testing.assert_close(ours(ids), hf(ids).logits, atol=1e-3, rtol=0)


def test_logits_index_selects_one_row(ours):
    ids = torch.tensor(PROMPTS[0]).unsqueeze(0)
    with torch.no_grad():
        full = ours(ids)
        last = ours(ids, logits_index=torch.tensor([ids.shape[1] - 1]))
    assert torch.equal(last[:, 0], full[:, -1])


def test_incremental_decode_matches_hf_greedy(hf, ours):
    cache = KVCache(num_layers=ours.model.num_layers)
    out = []
    with torch.no_grad():
        tok = int(ours(torch.tensor([PROMPTS[0]]), kv_cache=cache)[0, -1].argmax())
        for _ in range(8):
            if tok == EOS:
                break
            out.append(tok)
            tok = int(ours(torch.tensor([[tok]]), kv_cache=cache)[0, -1].argmax())
    assert out == _hf_greedy(hf, PROMPTS[0], 8)


# ---------------------------------------------------------------- through the backend + scheduler

@pytest.fixture(scope="module")
def backend(tiny_dir):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUSTOM_BACKEND_BLOCKS", "64")
        mp.setenv("CUSTOM_BACKEND_BLOCK_SIZE", "2")   # short prompts still fill blocks → prefix sharing
        b = CustomTorchBackend(device="cpu", model_name=tiny_dir)
        b.load_model(tiny_dir)
    return b


def test_backend_builds_full_attention_pools(backend):
    assert len(backend.pools) == 2
    assert all(p.window is None and p.num_kv_heads == 2 and p.head_dim == 16 for p in backend.pools)
    assert backend._eos_ids == {EOS}
    assert (backend.THINK_START, backend.THINK_END) == (VOCAB - 2, VOCAB - 1)
    assert not (backend._graph_on or backend._compile_on or backend._prefill_graph_on)


async def test_scheduler_greedy_matches_hf(backend, hf):
    """Two concurrent prompts with a shared prefix: batched prefill, prefix cache, batched decode."""
    from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest

    free_before = [p.free_count for p in backend.pools]
    sched = ContinuousBatchScheduler(backend, max_batch_size=8)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(p), max_tokens=8, session_id=f"s{i}",
                                 future=loop.create_future()) for i, p in enumerate(PROMPTS)]
        outs = await asyncio.gather(*(sched.submit(r) for r in reqs))
        # Same prompts again: now served from the prefix cache.
        reqs = [ScheduledRequest(token_ids=list(p), max_tokens=8, session_id=f"r{i}",
                                 future=loop.create_future()) for i, p in enumerate(PROMPTS)]
        again = await asyncio.gather(*(sched.submit(r) for r in reqs))
    finally:
        await sched.stop()
    expected = [_hf_greedy(hf, p, 8) for p in PROMPTS]
    assert outs == expected and again == expected
    assert backend.prefix_cache.hits > 0
    held = backend.prefix_cache.stats()["blocks_held"]
    assert [free_before[i] - p.free_count for i, p in enumerate(backend.pools)] == [held] * 2


# ---------------------------------------------------------------- config handling

def test_from_config_reads_the_real_30b_config(monkeypatch):
    from transformers import AutoConfig
    seen = {}

    class _Built(Exception):
        pass

    def record(**kw):
        seen.update(kw)
        raise _Built

    monkeypatch.setattr(qwen3_moe, "Qwen3MoeModel", record)
    with pytest.raises(_Built):
        Qwen3MoeForCausalLM.from_config(AutoConfig.from_pretrained(REAL_DIR))
    assert seen["num_layers"] == 48 and seen["num_experts"] == 128 and seen["top_k"] == 8
    assert (seen["num_q_heads"], seen["num_kv_heads"], seen["head_dim"]) == (32, 4, 128)
    assert seen["norm_topk_prob"] and seen["rope_theta"] == 1e6 and seen["eps"] == 1e-6
    kv_bytes = seen["num_layers"] * 2 * seen["num_kv_heads"] * seen["head_dim"] * 2
    assert kv_bytes == 96 * 1024


def test_real_stop_set_includes_im_end():
    from types import SimpleNamespace

    from inference_server.backends.base import stop_token_ids
    assert 151645 in stop_token_ids(str(REAL_DIR), SimpleNamespace(eos_token_id=None))


@pytest.mark.parametrize("over", [{"mlp_only_layers": [0]}, {"attention_bias": True},
                                  {"use_sliding_window": True, "sliding_window": 64},
                                  {"tie_word_embeddings": True}])
def test_unsupported_config_is_rejected(over):
    with pytest.raises(ValueError, match="does not support"):
        Qwen3MoeForCausalLM.from_config(_tiny_cfg(**over))


def test_backend_rejects_unknown_model_type(tmp_path):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    (tmp_path / "config.json").write_text('{"model_type": "gpt2"}')
    with pytest.raises(ValueError, match="gemma4 and qwen3_moe"):
        CustomTorchBackend(device="cpu").load_model(str(tmp_path))


def test_chat_template_disables_thinking():
    """The shim's enable_thinking=False renders Qwen3's empty think block, so answers start at once."""
    from transformers import AutoTokenizer

    from lab import chat_template
    try:
        tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-30B-A3B tokenizer not in the local HF cache")
    text = tk.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True,
                                  enable_thinking=chat_template.SHIM_ENABLE_THINKING, tokenize=False)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert chat_template.fingerprint("Qwen/Qwen3-30B-A3B")["enable_thinking"] is False


# ---------------------------------------------------------------- the real model

@pytest.mark.heavy
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3-30B-A3B parity needs a CUDA GPU (61 GB bf16)")
def test_full_model_matches_hf():
    """Ours adopts HF's tensors (no copy), so both fit on one 80 GB GPU."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen3-30B-A3B"
    hf_full = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    model = Qwen3MoeForCausalLM.from_state_dict(AutoConfig.from_pretrained(name), hf_full.state_dict()).eval()
    ids = AutoTokenizer.from_pretrained(name)("The capital of France is", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        a, b = model(ids).float(), hf_full(ids).logits.float()
    assert torch.equal(a[0, -1].argmax(), b[0, -1].argmax())
    assert (a - b).abs().mean() < 1e-2
