"""Parity harness for the custom Gemma 4 forward (M1).

Snapshots HF Gemma 4 E2B-it's outputs on a fixed prompt to a fixture file,
then any custom forward can be compared byte-for-byte against the fixture.

Run modes:
    pytest tests/test_gemma4_parity.py::test_capture_hf_fixture    # produce ground truth
    pytest tests/test_gemma4_parity.py::test_custom_matches_hf     # check custom forward (skipped until M1 lands)

The fixture lives in tests/fixtures/gemma4_e2b_parity.pt and is checked into the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "gemma4_e2b_parity.pt"
MODEL_NAME = "google/gemma-4-E2B-it"
PROMPT = "The capital of France is"
ATOL = 1e-3  # bf16 tolerance — tighten to 1e-5 if we move to fp32


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.mark.skipif(os.environ.get("CAPTURE_HF_FIXTURE") != "1", reason="set CAPTURE_HF_FIXTURE=1 to regenerate")
def test_capture_hf_fixture():
    """Produce the ground-truth fixture. Run with CAPTURE_HF_FIXTURE=1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()

    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)

    hidden_states = [h.detach().cpu() for h in out.hidden_states]  # [embed_out, layer_0_out, ..., final_norm_out]
    logits = out.logits.detach().cpu()
    next_token_logits = logits[:, -1, :]
    top5 = torch.topk(next_token_logits, k=5, dim=-1)

    FIXTURE_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "prompt": PROMPT,
            "model_name": MODEL_NAME,
            "input_ids": input_ids.cpu(),
            "hidden_states": hidden_states,
            "logits": logits,
            "top5_token_ids": top5.indices.cpu(),
            "top5_token_logits": top5.values.cpu(),
            "num_layers": len(hidden_states) - 1,
        },
        FIXTURE_PATH,
    )


def test_fixture_present():
    """Sanity: someone has captured the HF fixture."""
    assert FIXTURE_PATH.exists(), (
        f"Missing {FIXTURE_PATH}. Run with CAPTURE_HF_FIXTURE=1 once to generate."
    )


@pytest.mark.skip(reason="Enable once src/inference_server/models/gemma4.py exists (M1)")
def test_custom_matches_hf():
    """Byte-for-byte parity check of the custom forward against the HF fixture."""
    from inference_server.models.gemma4 import GemmaForCausalLM  # noqa: F401  # not yet built

    fixture = torch.load(FIXTURE_PATH, map_location="cpu")
    # custom = GemmaForCausalLM.from_safetensors(MODEL_NAME).eval()
    # out = custom(fixture["input_ids"], output_hidden_states=True)
    # for i, (h_hf, h_custom) in enumerate(zip(fixture["hidden_states"], out.hidden_states)):
    #     assert torch.allclose(h_hf, h_custom.cpu(), atol=ATOL), f"layer {i} hidden state mismatch"
    # assert torch.allclose(fixture["logits"], out.logits.cpu(), atol=ATOL)
    pytest.fail("Custom forward not implemented yet")
