"""Teacher-forced scoring for the output-equivalence check: backend.score_logprobs on a tiny Qwen3-MoE
(CPU, bf16) against HF's own logits, and the shim's vLLM-shaped `prompt_logprobs` response."""

from __future__ import annotations

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_server.openai_shim import router

SEQ = [5, 17, 300, 42, 8, 99, 123, 7, 55, 600, 3]


@pytest.fixture(scope="module")
def backend(tiny_qwen3_moe_dir):
    tiny_dir = tiny_qwen3_moe_dir
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUSTOM_BACKEND_BLOCKS", "64")
        mp.setenv("CUSTOM_BACKEND_BLOCK_SIZE", "2")
        b = CustomTorchBackend(device="cpu", model_name=tiny_dir)
        b.load_model(tiny_dir)
    return b


def test_scores_match_hf_logits(backend, tiny_qwen3_moe_dir):
    tiny_dir = tiny_qwen3_moe_dir
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


class _Tok:
    vocab_size, n_tokens, context_window = 990, 1000, 8      # ids 990-999: added special tokens


def _client(backend):
    app = FastAPI(); app.include_router(router)
    app.state.scheduler = object(); app.state.tokenizer = _Tok(); app.state.backend = backend
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


def test_scoring_echoes_the_trace_header_and_rejects_streaming():
    c = _client(_StubBackend())
    r = c.post("/v1/completions", json={"prompt": [3, 9], "prompt_logprobs": 2}, headers={"X-Trace-Id": "abc"})
    assert r.status_code == 200 and r.headers["X-Trace-Id"] == "abc"
    r = c.post("/v1/completions", json={"prompt": [3, 9], "prompt_logprobs": 2, "stream": True})
    assert r.status_code == 400


def test_scoring_a_full_window_prompt_ignores_max_tokens():
    """Scoring generates nothing, so the prompt + max_tokens check does not apply."""
    r = _client(_StubBackend()).post("/v1/completions", json={"prompt": list(range(8)), "prompt_logprobs": 2})
    assert r.status_code == 200


@pytest.mark.parametrize("prompt", [[], [999999], [-1], list(range(9))])
def test_pretokenized_prompts_are_validated(prompt):
    r = _client(_StubBackend()).post("/v1/completions", json={"prompt": prompt, "prompt_logprobs": 2})
    assert r.status_code == 400


def test_scoring_refuses_sequences_past_the_model_positions(backend):
    with pytest.raises(ValueError):
        backend.score_logprobs(list(range(1, 600)))


def test_added_special_token_ids_are_accepted():
    r = _client(_StubBackend()).post("/v1/completions", json={"prompt": [3, 995], "prompt_logprobs": 2})
    assert r.status_code == 200
