"""Prefill attention over a paged prefix plus this call's suffix, through torch SDPA (flash / efficient kernels).

The Triton paged prefill kernel reads K/V one 16-token block at a time and ran at ~5 TFLOP/s on an H200 against
SDPA's ~600 (1.8 s per layer at 32k tokens): on a long-prompt load that was the whole TTFT. Here each row gathers
its cached prefix K/V out of the blocks (a copy, cheap next to attention), appends the suffix K/V it just computed,
and runs one SDPA call: `is_causal` when there is no prefix, else a causal mask offset by the prefix length."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_prefix(pool_k: torch.Tensor, pool_v: torch.Tensor, block_ids: list[int], length: int) -> tuple:
    """The first `length` cached positions of one row, from its block ids: [1, Hkv, length, D] each."""
    bs = pool_k.shape[2]
    nb = (length + bs - 1) // bs
    ids = torch.tensor(block_ids[:nb], dtype=torch.long, device=pool_k.device)
    k = pool_k.index_select(0, ids).permute(1, 0, 2, 3).reshape(1, pool_k.shape[1], nb * bs, -1)[:, :, :length]
    v = pool_v.index_select(0, ids).permute(1, 0, 2, 3).reshape(1, pool_v.shape[1], nb * bs, -1)[:, :, :length]
    return k, v


def sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ctx, layer_idx: int,
                 scale: float) -> torch.Tensor:
    """q [B, Hq, S, D], k/v [B, Hkv, S, D]: this call's suffix per row, right-padded to S. `ctx` is the backend's
    prefill context (per-row caches with block tables, `_prefix` and `_suffix` lengths). Returns [B, S, Hq, D] in
    q's dtype; padding rows are left unwritten."""
    B, Hq, S, D = q.shape
    out = torch.empty(B, S, Hq, D, dtype=q.dtype, device=q.device)
    pool = ctx.pools[layer_idx]
    for n in range(B):
        P, L = ctx._prefix[n], ctx._suffix[n]
        qn, kn, vn = q[n:n + 1, :, :L], k[n:n + 1, :, :L], v[n:n + 1, :, :L]
        if P > 0:
            kp, vp = gather_prefix(pool.k, pool.v, ctx.caches[n].block_tables[layer_idx], P)
            kn, vn = torch.cat([kp, kn], dim=2), torch.cat([vp, vn], dim=2)
            # query i (at position P + i) sees every prefix key and suffix keys 0..i
            mask = torch.ones(L, P + L, dtype=torch.bool, device=q.device).tril(diagonal=P)
            o = F.scaled_dot_product_attention(qn, kn, vn, attn_mask=mask, scale=scale, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(qn, kn, vn, is_causal=L > 1, scale=scale, enable_gqa=True)
        out[n, :L] = o[0].transpose(0, 1)
    return out
