"""Prefill attention over a paged prefix plus this call's suffix, through torch SDPA, one call per layer for the whole
wave.

The Triton paged prefill kernel read K/V one 16-token block at a time and ran at ~5 TFLOP/s on an H200 against
SDPA's ~600 (1.8 s per layer at 32k tokens). A first SDPA version looped over rows in Python: with 16-19 prompts per
wave that loop (gathers, cats, masks, 48 layers deep) was half the scheduler thread's time, 1.3 s for a 10k-token
wave whose kernels take ~0.2 s. Here every row's cached prefix is gathered from the blocks in one index, the suffix
K/V appended, and one SDPA runs over the right-padded wave with a mask: prefix keys up to each row's prefix length,
suffix keys causal and within its real length."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def wave_mask(prefix: torch.Tensor, suffix: torch.Tensor, p_max: int, s_max: int) -> torch.Tensor:
    """[B, 1, s_max, p_max + s_max] bool: query i of row b sees prefix keys < prefix[b] and suffix keys t <= i,
    t < suffix[b]. Padding queries keep suffix key 0, so no row is left without a key."""
    dev = prefix.device
    ar_p = torch.arange(p_max, device=dev)
    ar_s = torch.arange(s_max, device=dev)
    pre = (ar_p[None, None, :] < prefix[:, None, None]).expand(-1, s_max, -1)              # [B, s, p_max]
    suf = (ar_s[None, :, None] >= ar_s[None, None, :]) & (ar_s[None, None, :] < suffix[:, None, None])  # [B, s, s]
    return torch.cat([pre, suf], dim=2)[:, None]


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx, layer_idx: int,
                 scale: float) -> torch.Tensor:
    """q [B, Hq, S, D], k/v [B, Hkv, S, D]: this call's suffix per row, right-padded to S. `ctx` is the backend's
    prefill context (block tables per layer, `prefix_lens`/`suffix_lens` int32 tensors, `_prefix`/`_suffix` lists).
    Returns [B, S, Hq, D] in q's dtype; padding query rows are unspecified (the caller slices them off)."""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    p_max = max(ctx._prefix) if ctx._prefix else 0
    if p_max == 0 and all(n == S for n in ctx._suffix):          # a cache-miss wave of equal lengths: plain causal
        o = F.scaled_dot_product_attention(q, k, v, is_causal=S > 1, scale=scale, enable_gqa=True)
        return o.transpose(1, 2)
    pool = ctx.pools[layer_idx]
    bs = pool.k.shape[2]
    if p_max:
        nb = (p_max + bs - 1) // bs
        bt = ctx.block_table_tensor(layer_idx)[:, :nb].long()                     # a row's prefix blocks come first
        kp = pool.k[bt].permute(0, 2, 1, 3, 4).reshape(B, Hkv, nb * bs, D)[:, :, :p_max]
        vp = pool.v[bt].permute(0, 2, 1, 3, 4).reshape(B, Hkv, nb * bs, D)[:, :, :p_max]
        k, v = torch.cat([kp, k], dim=2), torch.cat([vp, v], dim=2)
    mask = wave_mask(ctx.prefix_lens.long(), ctx.suffix_lens.long(), p_max, S)
    rep = Hq // Hkv
    if rep > 1:                                   # the masked backends do not all take enable_gqa: expand K/V heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return o.transpose(1, 2)
