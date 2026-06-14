"""Triton paged-attention DECODE kernel (M2.3).

Replaces the gather+pad+SDPA decode path: reads each sequence's K/V directly from the
scattered physical blocks via its block table, in-kernel, with no dense [N,H,Lmax,D]
materialization and no padding. One query token per sequence (decode), FlashAttention-style
online softmax over the sequence's blocks.

Handles Gemma 4 specifics: GQA (Hq query heads share num_kv_heads KV heads), scale=1.0
(magnitudes absorbed by the QK/V norms), and head_dim 256 (sliding) / 512 (full) via a
`D` constexpr — sidestepping the 256 ceiling that ruled out FlashAttention-2.

CUDA-only (Triton). Import lazily on the GPU path; do not import on CPU/MPS.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, Kp, Vp, BT, SL, Out,
    sq_n, sq_h,            # Q strides: [N, Hq, D] (last dim contiguous)
    sk_b, sk_h, sk_s,      # pool strides: [num_blocks, num_kv_heads, block_size, D]
    so_n, so_h,            # Out strides
    bt_n,                  # block_tables row stride: [N, max_blocks]
    scale, window,         # sliding window (W most-recent keys); huge sentinel = full attention
    GROUP: tl.constexpr, BLOCK_SIZE: tl.constexpr, D: tl.constexpr,
):
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    kvh = qh // GROUP

    d = tl.arange(0, D)
    slots = tl.arange(0, BLOCK_SIZE)

    q = tl.load(Q + seq * sq_n + qh * sq_h + d).to(tl.float32)  # [D]
    L = tl.load(SL + seq)
    n_blocks = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    lo = L - window                                             # keys at pos < lo are outside the window
    start_b = tl.maximum(lo, 0) // BLOCK_SIZE                   # skip fully-out blocks (perf)

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    # Runtime (per-sequence) loop bounds — NOT constexpr. A constexpr here would recompile
    # the kernel for every distinct block-count, thrashing under varied/growing seq lengths.
    for b in range(start_b, n_blocks):
        blk = tl.load(BT + seq * bt_n + b)
        offs = b * BLOCK_SIZE + slots
        valid = (offs < L) & (offs >= lo)                       # causal upper + window lower

        kptr = Kp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
        k = tl.load(kptr, mask=valid[:, None], other=0.0).to(tl.float32)  # [BLOCK_SIZE, D]
        s = tl.sum(q[None, :] * k, axis=1) * scale              # [BLOCK_SIZE]
        s = tl.where(valid, s, -float("inf"))

        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)                                   # [BLOCK_SIZE]
        l = l * alpha + tl.sum(p, axis=0)

        vptr = Vp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
        v = tl.load(vptr, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)      # [D]
        m = m_new

    out = acc / l
    tl.store(Out + seq * so_n + qh * so_h + d, out)


def paged_decode_attention(
    q: torch.Tensor,          # [N, Hq, D] — one decode token per sequence
    k_pool: torch.Tensor,     # [num_blocks, num_kv_heads, block_size, D]
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,  # [N, MAX_BLOCKS] int32 — physical block id per logical block
    seq_lens: torch.Tensor,      # [N] int32 — real KV length per sequence
    scale: float = 1.0,
    window: int = 1 << 30,       # sliding window (W most-recent keys); default = full attention
) -> torch.Tensor:
    """Decode attention reading K/V via block tables. Returns [N, Hq, D] (q's dtype)."""
    N, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    out = torch.empty(N, Hq, D, dtype=torch.float32, device=q.device)
    q = q.contiguous()
    _paged_decode_kernel[(N, Hq)](
        q, k_pool, v_pool, block_tables, seq_lens, out,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        scale, window,
        GROUP=Hq // num_kv_heads,
        BLOCK_SIZE=k_pool.shape[2],
        D=D,
    )
    return out.to(q.dtype)


@triton.jit
def _paged_prefill_kernel(
    Q, Kp, Vp, BT, PL, SL, Out,
    sq_n, sq_s, sq_h,      # Q strides: [N, S_q, Hq, D] (last dim contiguous)
    sk_b, sk_h, sk_s,      # pool strides
    so_n, so_s, so_h,      # Out strides: [N, S_q, Hq, D]
    bt_n,
    scale, window,
    GROUP: tl.constexpr, BLOCK_SIZE: tl.constexpr, D: tl.constexpr,
):
    # One program per (sequence, query-head, suffix-query-token). Each query attends, causally, to
    # its row's paged KV (prefix + suffix-so-far) — the multi-query cousin of the decode kernel.
    # The suffix K/V must already be appended to the blocks. Right-padded suffix: query j sits at
    # absolute position prefix_len + j.
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    qj = tl.program_id(2)
    kvh = qh // GROUP

    suffix_len = tl.load(SL + seq)
    if qj >= suffix_len:
        return                                   # padding query (right-pad)
    prefix_len = tl.load(PL + seq)
    q_pos = prefix_len + qj                       # this query's absolute position

    d = tl.arange(0, D)
    slots = tl.arange(0, BLOCK_SIZE)
    q = tl.load(Q + seq * sq_n + qj * sq_s + qh * sq_h + d).to(tl.float32)

    lo = q_pos + 1 - window                        # window lower bound (keys at pos < lo excluded)
    n_blocks = (q_pos + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    start_b = tl.maximum(lo, 0) // BLOCK_SIZE

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for b in range(start_b, n_blocks):
        blk = tl.load(BT + seq * bt_n + b)
        offs = b * BLOCK_SIZE + slots
        valid = (offs <= q_pos) & (offs >= lo)     # causal (≤ q_pos) + window
        kptr = Kp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
        k = tl.load(kptr, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l = l * alpha + tl.sum(p, axis=0)
        vptr = Vp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
        v = tl.load(vptr, mask=valid[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    out = acc / l
    tl.store(Out + seq * so_n + qj * so_s + qh * so_h + d, out)


def paged_prefill_attention(
    q: torch.Tensor,             # [N, S_q, Hq, D] — suffix query tokens (right-padded), per sequence
    k_pool: torch.Tensor,        # [num_blocks, num_kv_heads, block_size, D]
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,  # [N, MAX_BLOCKS] int32
    prefix_lens: torch.Tensor,   # [N] int32 — cached prefix length per sequence
    suffix_lens: torch.Tensor,   # [N] int32 — real suffix length per sequence
    scale: float = 1.0,
    window: int = 1 << 30,
) -> torch.Tensor:
    """Multi-query paged prefill attention (no gather). Suffix K/V must already be in the blocks.
    Returns [N, S_q, Hq, D] (q's dtype); padding query rows are left undefined (caller slices)."""
    N, Sq, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    out = torch.empty(N, Sq, Hq, D, dtype=torch.float32, device=q.device)
    q = q.contiguous()
    _paged_prefill_kernel[(N, Hq, Sq)](
        q, k_pool, v_pool, block_tables, prefix_lens, suffix_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0),
        scale, window,
        GROUP=Hq // num_kv_heads,
        BLOCK_SIZE=k_pool.shape[2],
        D=D,
    )
    return out.to(q.dtype)
