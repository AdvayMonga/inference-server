"""Prefill attention over a paged prefix plus this call's suffix, through torch SDPA, one call per layer for the whole
wave.

The Triton paged prefill kernel read K/V one 16-token block at a time and ran at ~5 TFLOP/s on an H200 against
SDPA's ~600 (1.8 s per layer at 32k tokens). A first SDPA version looped over rows in Python: with 16-19 prompts per
wave that loop (gathers, cats, masks, 48 layers deep) was half the scheduler thread's time, 1.3 s for a 10k-token
wave whose kernels take ~0.2 s. Here every row's cached prefix is gathered from the blocks in one index, the suffix
K/V appended, and one SDPA runs over the right-padded wave with a mask: prefix keys up to each row's prefix length,
suffix keys causal and within its real length."""

from __future__ import annotations

import os

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


# Waves of at most this many rows AND more padded tokens than BATCHED_MAX_TOKENS take the per-row path: the masked
# SDPA loses to flash on a few long rows, while on many rows the per-row Python loop loses far more (a 51-row mixed
# prefill step with long prefixes took 3.3 s per row-loop against ~0.1 s of kernels).
PER_ROW_MAX_ROWS = int(os.environ.get("CUSTOM_BACKEND_PREFILL_PER_ROW_MAX_ROWS", "4"))
BATCHED_MAX_TOKENS = int(os.environ.get("CUSTOM_BACKEND_PREFILL_BATCHED_TOKENS", "8192"))
GROUP_MAX_TOKENS = int(os.environ.get("CUSTOM_BACKEND_PREFILL_GROUP_TOKENS", "65536"))   # rows x (prefix+suffix) per SDPA call
_GQA_MASKED = [False]                             # set once a masked SDPA with enable_gqa raised on this build


def _gather_prefix(pool, ids: torch.Tensor, length: int, Hkv: int, D: int):
    bs = pool.k.shape[2]
    nb = (length + bs - 1) // bs
    ids = ids[:nb].long()
    k = pool.k.index_select(0, ids).permute(1, 0, 2, 3).reshape(1, Hkv, nb * bs, D)[:, :, :length]
    v = pool.v.index_select(0, ids).permute(1, 0, 2, 3).reshape(1, Hkv, nb * bs, D)[:, :, :length]
    return k, v


def _per_row(q, k, v, ctx, layer_idx, scale):
    """One flash call per row (a causal-offset mask only when the row has a cached prefix)."""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    out = torch.empty(B, S, Hq, D, dtype=q.dtype, device=q.device)
    pool = ctx.pools[layer_idx]
    bt = ctx.block_table_tensor(layer_idx) if any(ctx._prefix) else None
    for n in range(B):
        P, L = ctx._prefix[n], ctx._suffix[n]
        qn, kn, vn = q[n:n + 1, :, :L], k[n:n + 1, :, :L], v[n:n + 1, :, :L]
        if P > 0:
            kp, vp = _gather_prefix(pool, bt[n], P, Hkv, D)
            kn, vn = torch.cat([kp, kn], dim=2), torch.cat([vp, vn], dim=2)
            mask = torch.ones(L, P + L, dtype=torch.bool, device=q.device).tril(diagonal=P)
            o = F.scaled_dot_product_attention(qn, kn, vn, attn_mask=mask, scale=scale, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(qn, kn, vn, is_causal=L > 1, scale=scale, enable_gqa=True)
        out[n, :L] = o[0].transpose(0, 1)
    return out


def sdpa_prefill_packed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx, layer_idx: int,
                        scale: float) -> torch.Tensor:
    """q/k/v [1, H, T, D] over a packed wave (rows of `ctx._suffix` tokens, each after `ctx._prefix` cached tokens):
    one SDPA per row. The reference for the packed forward where FlashInfer is not used. Returns [1, T, Hq, D]."""
    _, Hq, T, D = q.shape
    Hkv = k.shape[1]
    out = torch.empty(T, Hq, D, dtype=q.dtype, device=q.device)
    pool = ctx.pools[layer_idx]
    bt = ctx.block_table_tensor(layer_idx) if any(ctx._prefix) else None
    s = 0
    for n, (P, L) in enumerate(zip(ctx._prefix, ctx._suffix)):
        qn, kn, vn = q[:, :, s:s + L], k[:, :, s:s + L], v[:, :, s:s + L]
        if P > 0:
            kp, vp = _gather_prefix(pool, bt[n], P, Hkv, D)
            kn, vn = torch.cat([kp, kn], dim=2), torch.cat([vp, vn], dim=2)
            mask = torch.ones(L, P + L, dtype=torch.bool, device=q.device).tril(diagonal=P)
            o = F.scaled_dot_product_attention(qn, kn, vn, attn_mask=mask, scale=scale, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(qn, kn, vn, is_causal=L > 1, scale=scale, enable_gqa=True)
        out[s:s + L] = o[0].transpose(0, 1)
        s += L
    return out.unsqueeze(0)


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx, layer_idx: int,
                 scale: float) -> torch.Tensor:
    """q [B, Hq, S, D], k/v [B, Hkv, S, D]: this call's suffix per row, right-padded to S. `ctx` is the backend's
    prefill context (block tables per layer, `prefix_lens`/`suffix_lens` int32 tensors, `_prefix`/`_suffix` lists).
    Returns [B, S, Hq, D] in q's dtype; padding query rows are unspecified (the caller slices them off)."""
    B, Hq, S, D = q.shape
    p_max = max(ctx._prefix) if ctx._prefix else 0
    if p_max == 0 and all(n == S for n in ctx._suffix):          # a cache-miss wave of equal lengths: plain causal
        o = F.scaled_dot_product_attention(q, k, v, is_causal=S > 1, scale=scale, enable_gqa=True)
        return o.transpose(1, 2)
    if B == 1 or (B <= PER_ROW_MAX_ROWS and B * (p_max + S) > BATCHED_MAX_TOKENS):
        return _per_row(q, k, v, ctx, layer_idx, scale)
    # The gathered K/V and the mask grow with rows x (prefix + suffix): a 256-row mixed step with long prefixes
    # ran out of memory. Rows are processed in groups that stay under the token cap.
    rows_per_group = max(1, GROUP_MAX_TOKENS // max(1, p_max + S))
    if B > rows_per_group:
        out = torch.empty(B, S, Hq, D, dtype=q.dtype, device=q.device)
        bt_all = ctx.block_table_tensor(layer_idx) if p_max else None
        for lo in range(0, B, rows_per_group):
            hi = min(B, lo + rows_per_group)
            out[lo:hi] = _wave(q[lo:hi], k[lo:hi], v[lo:hi], ctx, layer_idx, scale, lo, hi, bt_all)
        return out
    return _wave(q, k, v, ctx, layer_idx, scale, 0, B, ctx.block_table_tensor(layer_idx) if p_max else None)


def _wave(q, k, v, ctx, layer_idx, scale, lo, hi, bt_all):
    """Rows lo:hi of the wave as one masked SDPA call."""
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    prefix = ctx._prefix[lo:hi]
    p_max = max(prefix) if prefix else 0
    pool = ctx.pools[layer_idx]
    bs = pool.k.shape[2]
    if p_max:
        nb = (p_max + bs - 1) // bs
        bt = bt_all[lo:hi, :nb].long()                                            # a row's prefix blocks come first
        kp = pool.k[bt].permute(0, 2, 1, 3, 4).reshape(B, Hkv, nb * bs, D)[:, :, :p_max]
        vp = pool.v[bt].permute(0, 2, 1, 3, 4).reshape(B, Hkv, nb * bs, D)[:, :, :p_max]
        k, v = torch.cat([kp, k], dim=2), torch.cat([vp, v], dim=2)
    mask = wave_mask(ctx.prefix_lens[lo:hi].long(), ctx.suffix_lens[lo:hi].long(), p_max, S)
    if Hq != Hkv and not _GQA_MASKED[0]:         # expanding K/V 8x costs memory and time; try the native GQA first
        try:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)
            return o.transpose(1, 2)
        except RuntimeError:
            _GQA_MASKED[0] = True                 # this torch/backend will not: expand from now on
    if Hq != Hkv:
        rep = Hq // Hkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return o.transpose(1, 2)
