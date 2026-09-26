"""prefill_batch past the sliding window: each query keeps its own window (kb-20260926-ffe24b0f).

Tiny random model, window 16, no real weights. The reference is a plain full forward over
prompt + continuation; prefill_batch's cache is checked by teacher-forcing the continuation.
"""

from __future__ import annotations

import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.gemma4 import GemmaForCausalLM, GemmaModel
from inference_server.models.paged_kv_cache import RadixPrefixCache, make_pools_for_gemma

VOCAB, W = 64, 16
TOL = 1e-4  # monolithic prefill reaches ~5e-6 here; the bug gave ~2.5


def _backend():
    torch.manual_seed(0)
    model = GemmaModel(
        vocab_size=VOCAB, hidden_size=32, num_layers=3, intermediate_size=64,
        num_q_heads=4, num_kv_heads=1, head_dim=8, global_head_dim=8,
        hidden_per_layer=8, layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=W, num_kv_shared_layers=0, use_double_wide_mlp=False,
        eps=1e-6, dtype=torch.float32,
    )
    b = CustomTorchBackend(device="cpu")
    b.model = GemmaForCausalLM(model, final_logit_softcapping=30.0).eval()
    b.pools = make_pools_for_gemma(b.model, num_blocks_per_pool=256, block_size=4)
    b.prefix_cache = RadixPrefixCache(pools=b.pools)
    return b


def _toks(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g).tolist()


@torch.no_grad()
def _continuation_err(b, prompt, cache, cont):
    """Max |logit| gap between teacher-forcing `cont` from `cache` and one full forward."""
    full = b.model(torch.tensor([prompt + cont]))[0, len(prompt):]
    got = torch.cat([b.model(torch.tensor([[t]]), kv_cache=cache)[0, -1:] for t in cont])
    return (got - full).abs().max().item()


def test_prefill_batch_long_prompt_matches_full_forward():
    """Regression: a 50-token prompt (> window) through prefill_batch equals the full forward."""
    b = _backend()
    prompt, cont = _toks(50, 1), _toks(8, 2)
    (cache, _, kv_len), = b.prefill_batch([prompt])
    assert kv_len == len(prompt)
    assert _continuation_err(b, prompt, cache, cont) < TOL


def test_prefill_batch_ragged_wave_with_prefix_hits_matches_per_row():
    """K>1 wave, cached prefix + suffixes of different lengths past the window (the padding case)."""
    b = _backend()
    base = _toks(16, 3)
    b.prefill(base)                                   # warm the prefix cache (fits the window)
    prompts = [base + _toks(30, 4), base + _toks(9, 5), _toks(41, 6)]
    cont = _toks(6, 7)
    out = b.prefill_batch(prompts)
    assert b.last_cache_hit_tokens == 0               # last row missed; the first two hit
    for p, (cache, _, kv_len) in zip(prompts, out):
        assert kv_len == len(p)
        assert _continuation_err(b, p, cache, cont) < TOL


def test_decode_mask_single_query_unchanged():
    """S_q=1 with an explicit mask keeps exactly the rightmost W columns, as before the fix."""
    b = _backend()
    attn = b.model.model.layers[0].self_attn            # a sliding layer
    g = torch.Generator().manual_seed(1)
    n, past = 2, 40
    x = torch.randn(n, 1, 32, generator=g)
    kv = (torch.randn(n, 1, past, 8, generator=g), torch.randn(n, 1, past, 8, generator=g))
    cos, sin = b.model.model.rope_sliding(torch.tensor([[40], [25]]))
    mask = torch.zeros(n, 1, 1, past + 1, dtype=torch.bool)
    for r, L in enumerate([40, 25]):
        mask[r, 0, 0, past - L:] = True
    old_keep = torch.arange(past + 1) >= (past + 1 - W)   # the pre-fix rule
    with torch.no_grad():
        a, _ = attn(x, cos, sin, past_kv=kv, attn_mask=mask)
        ref, _ = attn(x, cos, sin, past_kv=kv, attn_mask=mask & old_keep)
    assert torch.equal(a, ref)
