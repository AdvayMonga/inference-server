"""Teacher-forced scoring for the output-equivalence check: backend.score_logprobs on a tiny Qwen3-MoE
(CPU, bf16) against HF's own logits, and the shim's vLLM-shaped `prompt_logprobs` response."""

from __future__ import annotations

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_server.openai_shim import router

VOCAB, EOS = 1000, 997
SEQ = [5, 17, 300, 42, 8, 99, 123, 7, 55, 600, 3]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM as HF

    d = tmp_path_factory.mktemp("tiny_qwen3_moe_score")
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(vocab_size=VOCAB, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
                         moe_intermediate_size=32, norm_topk_prob=True, tie_word_embeddings=False,
                         rope_parameters={"rope_type": "default", "rope_theta": 1e6},
                         max_position_embeddings=512, eos_token_id=EOS, pad_token_id=0, bos_token_id=1)
    HF._from_config(cfg, dtype=torch.bfloat16).save_pretrained(d)
    vocab = {f"t{i}": i for i in range(VOCAB - 2)} | {"<think>": VOCAB - 2, "</think>": VOCAB - 1}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def backend(tiny_dir):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUSTOM_BACKEND_BLOCKS", "64")
        mp.setenv("CUSTOM_BACKEND_BLOCK_SIZE", "2")
        b = CustomTorchBackend(device="cpu", model_name=tiny_dir)
        b.load_model(tiny_dir)
    return b


def test_scores_match_hf_logits(backend, tiny_dir):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        ref = torch.log_softmax(hf(torch.tensor([SEQ])).logits[0, :-1].float(), dim=-1)
    out = backend.score_logprobs(SEQ, top_k=5)
    assert len(out) == len(SEQ) and out[0] is None
    for i in range(1, len(SEQ)):
        assert SEQ[i] in out[i]                                    # the actual token is always reported
        assert abs(out[i][SEQ[i]] - ref[i - 1, SEQ[i]].item()) < 2e-2
        assert max(out[i], key=out[i].get) == int(ref[i - 1].argmax())


class _StubBackend:
    def score_logprobs(self, ids, k):
        return [None] + [{t: -0.5, t + 1: -1.5} for t in ids[1:]]


def _client(backend):
    app = FastAPI(); app.include_router(router)
    app.state.scheduler = object(); app.state.tokenizer = object(); app.state.backend = backend
    return TestClient(app)


def test_shim_returns_vllm_shaped_prompt_logprobs():
    r = _client(_StubBackend()).post("/v1/completions", json={"prompt": [3, 9, 4], "max_tokens": 1, "prompt_logprobs": 2})
    assert r.status_code == 200
    plp = r.json()["choices"][0]["prompt_logprobs"]
    assert plp[0] is None and plp[1] == {"9": {"logprob": -0.5, "rank": 1}, "10": {"logprob": -1.5, "rank": 2}}
    assert r.json()["usage"]["completion_tokens"] == 0


def test_shim_refuses_scoring_without_backend_support():
    r = _client(object()).post("/v1/completions", json={"prompt": [3, 9], "prompt_logprobs": 2})
    assert r.status_code == 501
