"""Fused Triton kernels for the Qwen3-MoE layer on CUDA (bf16 activations).

A decode step launched ~4,500 kernels across 48 layers (RMSNorm as 8 ops, RoPE as ~6, SiLU*up as 3, routing as
~6, ...), and at 1-3 us each those tiny launches, not the GEMMs, set the ~20 ms step. Each function here is one
launch with fp32 arithmetic inside and one bf16 rounding at the end; the torch code it replaces stays as the
CPU path and the reference the tests compare against. Off with CUSTOM_BACKEND_FUSED=0."""

from __future__ import annotations

import os

import importlib.util

import torch

ENABLED = os.environ.get("CUSTOM_BACKEND_FUSED", "1") == "1" and importlib.util.find_spec("triton") is not None


class _Lazy:
    """The kernels module, imported on first use."""
    def __getattr__(self, name):
        from inference_server.models import fused_kernels
        return getattr(fused_kernels, name)


K = _Lazy()


def _on(*ts: torch.Tensor) -> bool:
    return ENABLED and all(t.is_cuda and t.dtype == torch.bfloat16 for t in ts)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    K._rmsnorm_kernel[(x2.shape[0],)](x2, weight, y, x2.stride(0), y.stride(0), eps, D=D,
                                     BLOCK=K.triton.next_power_of_2(D))
    return y.view(x.shape)


def add_rmsnorm(h: torch.Tensor, a: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(h + a, rmsnorm(h + a)), both bf16, one launch."""
    D = h.shape[-1]
    h2, a2 = h.reshape(-1, D).contiguous(), a.reshape(-1, D).contiguous()
    ho, y = torch.empty_like(h2), torch.empty_like(h2)
    K._add_rmsnorm_kernel[(h2.shape[0],)](h2, a2, weight, ho, y, h2.stride(0), eps, D=D, BLOCK=K.triton.next_power_of_2(D))
    return ho.view(h.shape), y.view(h.shape)


def rmsnorm_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float) -> torch.Tensor:
    """x [B, S, H, D] -> head-wise RMSNorm then rotary (cos/sin [B, S, D], HF layout: halves repeat)."""
    B, S, H, D = x.shape
    if x.stride(3) != 1 or x.stride(2) != D or (B > 1 and S > 1 and x.stride(0) != S * x.stride(1)):
        x = x.contiguous()                      # any layout but "rows of H*D-contiguous heads" (never the qkv slice)
    row_stride = x.stride(0) if S == 1 else x.stride(1)       # one token per row: a size-1 dim's stride is arbitrary
    y = torch.empty(B, S, H, D, dtype=x.dtype, device=x.device)
    cos2, sin2 = cos.reshape(B * S, -1), sin.reshape(B * S, -1)
    if not cos2.is_contiguous():
        cos2, sin2 = cos2.contiguous(), sin2.contiguous()
    HB = min(8, K.triton.next_power_of_2(H))
    K._rmsnorm_rope_kernel[(B * S, K.triton.cdiv(H, HB))](x, weight, cos2, sin2, y, row_stride, x.stride(2),
                                                         cos2.stride(0), eps, H=H, HB=HB, D=D, HALF=D // 2)
    return y


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    """x [M, 2I] -> silu(x[:, :I]) * x[:, I:]."""
    M, two_i = x.shape
    I = two_i // 2
    x = x.contiguous()
    y = torch.empty(M, I, dtype=torch.bfloat16, device=x.device)      # x may be the fp32 split-K accumulator
    BLOCK = 1024
    K._silu_mul_kernel[(M, K.triton.cdiv(I, BLOCK))](x, y, x.stride(0), y.stride(0), I=I, BLOCK=BLOCK)
    return y


def kv_store(pool_k: torch.Tensor, pool_v: torch.Tensor, bt: torch.Tensor, lb: torch.Tensor, slot: torch.Tensor,
             k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    """pool[bt[r, lb[r]], :, slot[r], :] = new[r, :, 0, :] for K and V in one launch (k/v_new [R, H, 1, D], `bt`
    the layer's [R, cols] block table, the pools [blocks, H, bs, D]; the gather and two index_put scatters it
    replaces were ~9 launches per layer)."""
    R, H, _, D = k_new.shape
    assert k_new.stride(3) == 1 and v_new.stride(3) == 1 and bt.stride(1) == 1
    K._kv_store_kernel[(R,)](k_new, v_new, pool_k, pool_v, bt, lb, slot, k_new.stride(0), k_new.stride(1),
                             v_new.stride(0), v_new.stride(1), bt.stride(0), pool_k.stride(0), pool_k.stride(1),
                             pool_k.stride(2), H=H, D=D, BLOCK=K.triton.next_power_of_2(H * D))


def kv_store_rows(pool_k: torch.Tensor, pool_v: torch.Tensor, bids: torch.Tensor, slots: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor) -> None:
    """pool[bids[t], :, slots[t], :] = new[t] for K and V in one launch (k/v [T, H, D], unit stride along D)."""
    T, H, D = k.shape
    assert k.stride(2) == 1 and v.stride(2) == 1
    K._kv_store_rows_kernel[(T,)](k, v, pool_k, pool_v, bids, slots, k.stride(0), k.stride(1), v.stride(0),
                                  v.stride(1), pool_k.stride(0), pool_k.stride(1), pool_k.stride(2), H=H, D=D,
                                  BLOCK=K.triton.next_power_of_2(H * D))


def packed_slots(prefix: torch.Tensor, suffix: torch.Tensor, starts: torch.Tensor, bt: torch.Tensor, bs: int,
                 total: int, longest: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A packed wave's per-token K/V (block id, slot), int64 [total], from the rows' prefix and suffix lengths, their
    first token's index and block tables [rows, cols], in one launch."""
    bids = torch.empty(total, dtype=torch.long, device=bt.device)
    slots = torch.empty(total, dtype=torch.long, device=bt.device)
    BLOCK = 1024
    K._packed_slots_kernel[(prefix.numel(), K.triton.cdiv(max(longest, 1), BLOCK))](
        prefix, suffix, starts, bt, bids, slots, bt.stride(0), BS=bs, BLOCK=BLOCK)
    return bids, slots


def compose_rows(old_bt: torch.Tensor, old_seq: torch.Tensor, old_idx: torch.Tensor, new_j: torch.Tensor,
                 new_bt: torch.Tensor, new_seq: torch.Tensor, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A decode state's (block table [R, cols], lengths [R]) after joins and leaves, in one launch: row r is old row
    old_idx[r], or (old_idx[r] = -1) joining row new_j[r]; columns past a row's table are 0. Off CUDA, in torch."""
    R = old_idx.numel()
    if not (ENABLED and old_bt.is_cuda):
        o, j = old_idx.long(), new_j.long()
        pad = lambda t: torch.nn.functional.pad(t, (0, cols - t.shape[1]))
        take_old = (o >= 0)[:, None]
        bt = torch.where(take_old, pad(old_bt)[o.clamp_min(0)], pad(new_bt)[j.clamp_min(0)])
        return bt, torch.where(o >= 0, old_seq[o.clamp_min(0)], new_seq[j.clamp_min(0)])
    bt = torch.empty(R, cols, dtype=torch.long, device=old_bt.device)
    seq = torch.empty(R, dtype=torch.long, device=old_bt.device)
    BLOCK = 256
    K._compose_rows_kernel[(R, K.triton.cdiv(cols, BLOCK))](old_bt, old_seq, old_idx, new_j, new_bt, new_seq, bt, seq,
                                                             old_bt.shape[1], old_bt.stride(0), new_bt.shape[1],
                                                             new_bt.stride(0), cols, BLOCK=BLOCK)
    return bt, seq


def assemble_tokens(prev: torch.Tensor | None, devt: torch.Tensor | None, spec: torch.Tensor, n: int) -> torch.Tensor:
    """A decode step's [n] input tokens in one launch. spec = [src, src_dev, host] ([3n] long): row r takes prev[src[r]]
    if src[r] >= 0, else devt[src_dev[r]] if src_dev[r] >= 0, else host[r]."""
    out = torch.empty(n, dtype=torch.long, device=spec.device)
    K._assemble_tokens_kernel[(K.triton.cdiv(max(n, 1), 256),)](
        prev if prev is not None else spec, devt if devt is not None else spec, spec, out, n, BLOCK=256)
    return out


def rmsnorm_rows(h: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The lm head's rows of the final norm in one launch: h [B, S, D] -> [B, n, D] normed rows, where a packed wave
    (B = 1, n = idx.numel()) takes positions idx and a batch (n = 1) takes row b's position idx[b]."""
    B, S, D = h.shape
    packed = B == 1 and idx.numel() > 1
    n = idx.numel()
    x = h.reshape(B * S, D)
    if x.stride(1) != 1:
        x = x.contiguous()
    y = torch.empty(n, D, dtype=h.dtype, device=h.device)
    K._rmsnorm_rows_kernel[(n,)](x, idx, weight, y, x.stride(0), S, eps, PACKED=packed, D=D,
                                 BLOCK=K.triton.next_power_of_2(D))
    return y.view(1, n, D) if packed else y.view(n, 1, D)


def topk_softmax(logits: torch.Tensor, k: int, renorm: bool, fp32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Like models.qwen3_moe.route: (weights in logits' dtype, or fp32 as computed, expert ids int64), both [M, k]."""
    M, E = logits.shape
    logits = logits.contiguous()
    w = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, k, dtype=torch.int64, device=logits.device)
    K._topk_softmax_kernel[(M,)](logits, w, ids, logits.stride(0), E=E, BLOCK_E=K.triton.next_power_of_2(E), K=k,
                               RENORM=renorm)
    return (w if fp32 else w.to(logits.dtype)), ids


def add_cast(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16(h + y) for a bf16 residual `h` and an fp32 `y` of the same shape, one launch."""
    D = h.shape[-1]
    h2, y2 = h.reshape(-1, D).contiguous(), y.reshape(-1, D).contiguous()
    out = torch.empty_like(h2)
    K._add_cast_kernel[(h2.shape[0],)](h2, y2, out, D, D=D, BLOCK=K.triton.next_power_of_2(D))
    return out.view(h.shape)


def embed_rope(ids: torch.Tensor, weight: torch.Tensor, inv_freq: torch.Tensor,
               pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ids, pos [B, S] -> (embedding rows [B, S, H], cos, sin [B, S, D]) in one launch; RotaryEmbedding's tables in
    the weight's dtype."""
    B, S = ids.shape
    H, half = weight.shape[1], inv_freq.numel()
    ids, pos = ids.reshape(-1).contiguous(), pos.reshape(-1).contiguous()
    h = torch.empty(B, S, H, dtype=weight.dtype, device=ids.device)
    cos = torch.empty(B, S, 2 * half, dtype=weight.dtype, device=ids.device)
    sin = torch.empty_like(cos)
    K._embed_rope_kernel[(B * S,)](ids, pos, weight, inv_freq, h, cos, sin, H=H,
                                   BLOCK_H=min(4096, K.triton.next_power_of_2(H)), HALF=half)
    return h, cos, sin


def step_inputs(seq_lens: torch.Tensor, bs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(kv_block = len // bs, kv_slot = len % bs, attention lengths len + 1 as int32) per row, one launch."""
    n = seq_lens.numel()
    kv_block, kv_slot = torch.empty_like(seq_lens), torch.empty_like(seq_lens)
    attn_sl = torch.empty(n, dtype=torch.int32, device=seq_lens.device)
    K._step_inputs_kernel[(K.triton.cdiv(max(n, 1), 256),)](seq_lens, kv_block, kv_slot, attn_sl, n, BS=bs, BLOCK=256)
    return kv_block, kv_slot, attn_sl


def stage_decode(g: dict, cur: torch.Tensor, pos: torch.Tensor, seq: torch.Tensor, bt: torch.Tensor, n: int,
                 scratch: int, new: tuple[torch.Tensor, torch.Tensor] | None = None,
                 seq_next: torch.Tensor | None = None) -> None:
    """A captured graph's static inputs from the live rows (cur, pos [n, 1]; seq [n]; bt [n, cols], any int dtype) in
    one launch: g["tokens"], g["pos"], g["seqlens"], g["bt"] (int32). `new`: per row a block allocated this step
    (-1 for none) and its column, written into bt too. `seq_next`: the lengths after the step."""
    gb = g["bt"]
    bucket, g_cols = gb.shape
    BLOCK_C = 1024
    K._stage_decode_kernel[(bucket, K.triton.cdiv(g_cols, BLOCK_C))](
        cur, pos, seq, bt, new[0] if new else seq, new[1] if new else seq, g["tokens"], g["pos"], g["seqlens"], gb,
        seq_next if seq_next is not None else seq, n, bt.shape[1], cur.stride(0), pos.stride(0), bt.stride(0), g_cols,
        scratch, HAS_NEW=new is not None, ADVANCE=seq_next is not None, BLOCK_C=BLOCK_C)


def argmax(x: torch.Tensor) -> torch.Tensor:
    """x [N, V] (unit stride along V) -> [N] int64, the first index of each row's maximum, as torch.argmax."""
    N, V = x.shape
    CHUNK = 8192
    n_chunks = K.triton.cdiv(V, CHUNK)
    pv = torch.empty(N, n_chunks, dtype=torch.float32, device=x.device)
    pi = torch.empty(N, n_chunks, dtype=torch.int32, device=x.device)
    out = torch.empty(N, dtype=torch.int64, device=x.device)
    K._argmax_partial_kernel[(N, n_chunks)](x, pv, pi, V, x.stride(0), n_chunks, CHUNK=CHUNK, num_warps=8)
    K._argmax_final_kernel[(N,)](pv, pi, out, n_chunks, BLOCK=K.triton.next_power_of_2(n_chunks))
    return out
