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


# --- Split-K (flash-decoding) variant -------------------------------------------------
# The kernel above launches one program per (sequence, query-head). At the low concurrency
# our SLO actually lives at (n=1-8 rows), that is a few dozen CUDA blocks on a 108-SM A100,
# and each program walks its row's KV blocks SERIALLY. Splitting the KV dimension across
# programs recovers the parallelism: each split reduces a slice of the blocks into a partial
# (m, l, acc), then a combine pass merges them with the same online-softmax rescaling.
#
# The number of splits is chosen from (N, Hq) ONLY — never from sequence length — because the
# grid is baked into the CUDA graph at capture time and must not vary at replay. The per-split
# block RANGE is computed inside the kernel from the runtime seq_len, so a split that lands
# past the end of a short row just writes an empty partial (m=-inf, l=0) and contributes zero.


@triton.jit
def _paged_decode_splitk_kernel(
    Q, Kp, Vp, BT, SL, PM, PL, PA,
    sq_n, sq_h,
    sk_b, sk_h, sk_s,
    pm_n, pm_h,            # partial m/l strides: [N, Hq, SPLITS]
    pa_n, pa_h, pa_s,      # partial acc strides: [N, Hq, SPLITS, D]
    bt_n,
    scale, window,
    GROUP: tl.constexpr, BLOCK_SIZE: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    sp = tl.program_id(2)
    kvh = qh // GROUP

    d = tl.arange(0, D)
    slots = tl.arange(0, BLOCK_SIZE)

    L = tl.load(SL + seq)
    n_blocks = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    lo = L - window
    start_b = tl.maximum(lo, 0) // BLOCK_SIZE

    # Even split of this row's live block range across SPLITS programs (runtime bounds).
    live = n_blocks - start_b
    per = (live + SPLITS - 1) // SPLITS
    b0 = start_b + sp * per
    b1 = tl.minimum(b0 + per, n_blocks)

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    if b0 < b1:
        q = tl.load(Q + seq * sq_n + qh * sq_h + d).to(tl.float32)
        for b in range(b0, b1):
            blk = tl.load(BT + seq * bt_n + b)
            offs = b * BLOCK_SIZE + slots
            valid = (offs < L) & (offs >= lo)
            kptr = Kp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
            k = tl.load(kptr, mask=valid[:, None], other=0.0).to(tl.float32)
            sc = tl.sum(q[None, :] * k, axis=1) * scale
            sc = tl.where(valid, sc, -float("inf"))
            m_new = tl.maximum(m, tl.max(sc, axis=0))
            alpha = tl.exp(m - m_new)
            pr = tl.exp(sc - m_new)
            l = l * alpha + tl.sum(pr, axis=0)
            vptr = Vp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
            v = tl.load(vptr, mask=valid[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(pr[:, None] * v, axis=0)
            m = m_new

    tl.store(PM + seq * pm_n + qh * pm_h + sp, m)
    tl.store(PL + seq * pm_n + qh * pm_h + sp, l)
    tl.store(PA + seq * pa_n + qh * pa_h + sp * pa_s + d, acc)


@triton.jit
def _splitk_combine_kernel(
    PM, PL, PA, Out,
    pm_n, pm_h,
    pa_n, pa_h, pa_s,
    so_n, so_h,
    D: tl.constexpr, SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    d = tl.arange(0, D)

    # Standard online-softmax merge: rescale every partial to the global max, then sum.
    # An empty split carries m=-inf, so exp(m - M) is 0 and it drops out cleanly.
    mmax = -float("inf")
    for sp in range(0, SPLITS):
        mmax = tl.maximum(mmax, tl.load(PM + seq * pm_n + qh * pm_h + sp))

    l_tot = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for sp in range(0, SPLITS):
        m_s = tl.load(PM + seq * pm_n + qh * pm_h + sp)
        w = tl.exp(m_s - mmax)
        l_tot += tl.load(PL + seq * pm_n + qh * pm_h + sp) * w
        acc += tl.load(PA + seq * pa_n + qh * pa_h + sp * pa_s + d) * w

    tl.store(Out + seq * so_n + qh * so_h + d, acc / l_tot)


# Target program count for the split heuristic. A100 has 108 SMs; aiming a little above that
# keeps the tail busy without over-splitting into launch overhead.
_TARGET_PROGRAMS = 128
_MAX_SPLITS = 8


def decode_splits(n_seqs: int, n_qheads: int) -> int:
    """How many KV splits to use, from batch shape ONLY (must be CUDA-graph-stable)."""
    base = n_seqs * n_qheads
    if base >= _TARGET_PROGRAMS:
        return 1
    s = (_TARGET_PROGRAMS + base - 1) // base
    return min(_MAX_SPLITS, 1 << (s - 1).bit_length())      # round up to a power of two


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

    splits = decode_splits(N, Hq)
    if splits > 1:
        pm = torch.empty(N, Hq, splits, dtype=torch.float32, device=q.device)
        pl = torch.empty(N, Hq, splits, dtype=torch.float32, device=q.device)
        pa = torch.empty(N, Hq, splits, D, dtype=torch.float32, device=q.device)
        _paged_decode_splitk_kernel[(N, Hq, splits)](
            q, k_pool, v_pool, block_tables, seq_lens, pm, pl, pa,
            q.stride(0), q.stride(1),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            pm.stride(0), pm.stride(1),
            pa.stride(0), pa.stride(1), pa.stride(2),
            block_tables.stride(0),
            scale, window,
            GROUP=Hq // num_kv_heads, BLOCK_SIZE=k_pool.shape[2], D=D, SPLITS=splits,
        )
        _splitk_combine_kernel[(N, Hq)](
            pm, pl, pa, out,
            pm.stride(0), pm.stride(1),
            pa.stride(0), pa.stride(1), pa.stride(2),
            out.stride(0), out.stride(1),
            D=D, SPLITS=splits,
        )
        return out.to(q.dtype)

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
