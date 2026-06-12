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


def test_windowed_storage_frees_blocks_and_reads_correctly():
    """Unit test (no model): a sliding-windowed pool frees out-of-window blocks on append,
    keeps the in-window tail readable, and returns everything on free_all."""
    from inference_server.models.paged_kv_cache import BlockPool, PagedKVCache

    bs, W = 4, 8  # window = 8 tokens = 2 blocks
    pool = BlockPool(num_blocks=64, block_size=bs, num_kv_heads=1, head_dim=2,
                     dtype=torch.float32, device=torch.device("cpu"), window=W)
    cache = PagedKVCache(pools=[pool])
    free0 = pool.free_count

    for t in range(20):  # decode-style, one token at a time, value = position
        k = torch.full((1, 1, 1, 2), float(t))
        v = torch.full((1, 1, 1, 2), float(t) + 100)
        cache.append(0, k, v)

    # keep_from = (20-8)//4 = 3 → logical blocks 0,1,2 (positions 0..11) evicted; 3,4 retained.
    assert cache.evicted[0] == 3
    assert pool.free_count == free0 - 2, "only the 2 in-window blocks should still be held"

    k_get, _ = cache.get(0)
    assert k_get.shape == (1, 1, 20, 2)
    for t in range(12, 20):  # in-window positions must read back exactly
        assert torch.allclose(k_get[0, 0, t, :], torch.tensor([float(t), float(t)]))

    cache.free_all()
    assert pool.free_count == free0, "free_all must return every block (no double-free of evicted)"


def test_windowed_decode_matches_hf():
    """Decode from a >window cache (sliding layers evicted during prefill) must match HF."""
    from transformers import AutoModelForCausalLM

    from inference_server.models.gemma4 import GemmaForCausalLM
    from inference_server.models.paged_kv_cache import PagedKVCache, make_pools_for_gemma

    device = torch.device("cpu")
    torch.manual_seed(0)
    S = 540  # > 512 → prefill evicts the first sliding-layer block
    prompt = torch.randint(0, 100_000, (1, S), device=device)

    custom = GemmaForCausalLM.from_hf(MODEL_NAME).to(device).eval()
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device).eval()

    with torch.no_grad():
        pools = make_pools_for_gemma(custom, num_blocks_per_pool=128, block_size=16)
        cache = PagedKVCache(pools=pools)
        first = int(custom(prompt, kv_cache=cache)[:, -1, :].argmax(-1).item())  # prefill (evicts)
        assert cache.any_evicted, "expected eviction at S=540 > window 512"
        dl = custom(torch.tensor([[first]], device=device),
                    position_ids=torch.tensor([[S]], device=device), kv_cache=cache)[:, -1, :]
        cache.free_all()

        seq = torch.cat([prompt, torch.tensor([[first]], device=device)], dim=1)
        hl = hf(seq, use_cache=False).logits[:, -1, :]

    assert torch.equal(dl.argmax(-1), hl.argmax(-1)), "windowed-decode argmax diverges from HF"
    assert (dl.float() - hl.float()).abs().max().item() < ATOL, "windowed-decode logits diverge from HF"
