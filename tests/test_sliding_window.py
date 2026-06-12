"""Sliding-window attention correctness (M-sliding).

Gemma 4 is 28/35 sliding-window (512) layers. The forward must mask each query to the last
512 keys on those layers. The 5-token parity fixture can't catch this (5 << 512), so here we
compare a >512-token prompt against HF (which implements the window) on CPU. Without the
window mask, custom would attend to the full history and diverge.
"""

from __future__ import annotations

import pytest
import torch

MODEL_NAME = "google/gemma-4-E2B-it"
ATOL = 5e-2  # 600 tokens × 35 bf16 layers accumulates more than the 5-token fixture's 1e-3


@pytest.mark.parametrize("seq_len", [600])  # > sliding_window (512) so the window is active
def test_long_prompt_matches_hf(seq_len):
    from transformers import AutoModelForCausalLM

    from inference_server.models.gemma4 import GemmaForCausalLM

    device = torch.device("cpu")
    torch.manual_seed(0)
    ids = torch.randint(0, 100_000, (1, seq_len), device=device)

    custom = GemmaForCausalLM.from_hf(MODEL_NAME).to(device).eval()
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device).eval()

    with torch.no_grad():
        cl = custom(ids)
        hl = hf(ids, use_cache=False).logits

    last_diff = (cl[:, -1, :].float() - hl[:, -1, :].float()).abs().max().item()
    full_diff = (cl.float() - hl.float()).abs().max().item()
    assert torch.equal(cl[:, -1, :].argmax(-1), hl[:, -1, :].argmax(-1)), "argmax diverges from HF"
    assert last_diff < ATOL, f"last-position logits diverge from HF (max {last_diff})"
    assert full_diff < ATOL, f"logits diverge from HF (max {full_diff})"
