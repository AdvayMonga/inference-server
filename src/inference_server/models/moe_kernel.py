"""Triton grouped GEMM for the MoE experts at small M (decode).

torch._grouped_mm (CUTLASS) took ~36 us per call on an H200 for a batch-1 decode step: 8 active experts, one row
each, but a 128-group launch. That was 3.5 ms of a ~13 ms step against a ~0.9 ms weight-read floor. This kernel is
vLLM's fused_moe layout: (token, slot) pairs sorted by expert and padded per expert to BLOCK_M rows, one program per
(row block, N tile); each program reads only its expert's weight tile, so the bytes read are the active experts'
weights, once. Rows come straight from the hidden states through the token map (no gathered copy). Weights bf16, or
fp8 e4m3 with one fp32 scale per output channel (W8A16: the tile is widened to bf16 in registers, the scale applied to
the fp32 accumulator; added 2026-10-18 so CUSTOM_BACKEND_QUANT=fp8 keeps the decode path, where at batch 128 nearly
every expert is streamed each step). Large M keeps the CUTLASS path."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@functools.lru_cache(maxsize=None)
def _arange(n: int, device: torch.device, scale: int = 1) -> torch.Tensor:
    """A constant index vector, built once per (n, device): these were ~6 launches per layer per step."""
    ar = torch.arange(n, device=device)
    return ar * scale if scale != 1 else ar


@triton.jit(do_not_specialize=["M"])
def _moe_gemm_kernel(A, TOK, W, SCALE, A_SCALE, C, SORTED, EXPERT_IDS, M, N, K,
                     stride_am, stride_wn, stride_wk, stride_cm,
                     E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     ROWS_FROM_TOKENS: tl.constexpr, SPLIT_K: tl.constexpr, W_FP8: tl.constexpr, A_FP8: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(EXPERT_IDS + pid_m)
    if e >= E:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.load(SORTED + offs_m)                       # pair index per row; >= M is padding
    valid = rows < M
    if ROWS_FROM_TOKENS:
        src = tl.load(TOK + rows, mask=valid, other=0)    # the token each pair came from
    else:
        src = rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    k_lo = tl.program_id(2) * (K // SPLIT_K)              # this program's K slice (K divides evenly by SPLIT_K)
    a_ptrs = A + src[:, None] * stride_am + k_lo + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if A_FP8:
        # W8A8: both operands fp8 e4m3 on the tensor cores (the weight tile loaded [K, N], as the MMA wants it), the
        # per-channel weight scale and the per-row activation scale applied once in the epilogue.
        w_ptrs = W + e.to(tl.int64) * N * stride_wn + (k_lo + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn
        for k0 in range(0, K // SPLIT_K, BLOCK_K):
            k_ok = (k0 + offs_k) < K // SPLIT_K
            a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            acc = tl.dot(a, w, acc, max_num_imprecise_acc=0)     # fp32 accumulation, as torch's grouped GEMM
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * stride_wk
        acc = acc * tl.load(A_SCALE + src, mask=valid, other=0.0)[:, None]
    else:
        w_ptrs = W + e.to(tl.int64) * N * stride_wn + offs_n[:, None] * stride_wn + (k_lo + offs_k[None, :]) * stride_wk
        for k0 in range(0, K // SPLIT_K, BLOCK_K):
            k_ok = (k0 + offs_k) < K // SPLIT_K
            a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_ok[:, None] & k_ok[None, :], other=0.0)
            if W_FP8:
                w = w.to(tl.bfloat16)
            acc += tl.dot(a, tl.trans(w))
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * stride_wk
    if W_FP8:
        acc = acc * tl.load(SCALE + e.to(tl.int64) * N + offs_n, mask=n_ok, other=0.0)[None, :]
    c_ptrs = C + rows[:, None] * stride_cm + offs_n[None, :]
    if SPLIT_K > 1:                                       # fp32 output, zeroed by the caller
        tl.atomic_add(c_ptrs, acc, mask=valid[:, None] & n_ok[None, :])
    else:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=valid[:, None] & n_ok[None, :])


def align(sorted_experts: torch.Tensor, num_experts: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each expert's run of sorted pairs to a multiple of `block_m`: (sorted pair ids with `M` as padding, the
    expert of each row block, `num_experts` for blocks past the end). Device-side only, static shapes for graphs."""
    M = sorted_experts.numel()
    sorted_experts = sorted_experts.clamp_(0, num_experts - 1)      # never index past the last expert
    dev = sorted_experts.device
    ar = _arange(num_experts, dev)
    ends = torch.searchsorted(sorted_experts, ar, right=True)
    starts = torch.searchsorted(sorted_experts, ar, right=False)
    counts = ends - starts
    padded = (counts + block_m - 1) // block_m * block_m
    pstart = torch.cumsum(padded, 0) - padded                   # padded start of each expert
    max_pad = M + num_experts * (block_m - 1)
    sorted_ids = torch.full((max_pad,), M, dtype=torch.int32, device=dev)
    i = _arange(M, dev)
    dest = pstart[sorted_experts] + (i - starts[sorted_experts])
    sorted_ids[dest] = i.to(torch.int32)
    n_blocks = max_pad // block_m
    block_starts = _arange(n_blocks, dev, block_m)
    total = pstart[-1] + padded[-1] if num_experts else 0
    expert_ids = torch.searchsorted(pstart, block_starts, right=True) - 1
    expert_ids = torch.where(block_starts < total, expert_ids, torch.full_like(expert_ids, num_experts))
    return sorted_ids, expert_ids.to(torch.int32)


@triton.jit
def _align_kernel(SORTED_E, ORDER, W, SORTED_IDS, EXPERT_IDS, TOK, PAIR_W, SCRATCH, M, n_blocks, max_blocks,
                  E: tl.constexpr, E_PAD: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK: tl.constexpr):
    """One program: from the expert-sorted pairs, everything the grouped GEMMs need (`align` did it in ~15 launches,
    plus the token map and the per-pair routing weight that cost two more)."""
    offs = tl.arange(0, BLOCK)
    m = offs < M
    e = tl.load(SORTED_E + offs, mask=m, other=E).to(tl.int32)
    bins = tl.arange(0, E_PAD)
    counts = tl.where(bins < E, tl.histogram(e, E_PAD), 0)          # pairs per expert (bin E holds the padding)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    pstart = tl.cumsum(padded, 0) - padded                          # each expert's first padded row
    ustart = tl.cumsum(counts, 0) - counts                          # and its first sorted pair
    tl.store(SCRATCH + bins, pstart)
    tl.store(SCRATCH + E_PAD + bins, ustart)
    for s in range(0, n_blocks * BLOCK_M, BLOCK):                   # padding rows read M, blocks past the end E
        tl.store(SORTED_IDS + s + offs, tl.zeros((BLOCK,), tl.int32) + M, mask=(s + offs) < n_blocks * BLOCK_M)
    for s in range(0, n_blocks, BLOCK):
        tl.store(EXPERT_IDS + s + offs, tl.zeros((BLOCK,), tl.int32) + E, mask=(s + offs) < n_blocks)
    tl.debug_barrier()
    dest = tl.load(SCRATCH + e, mask=m, other=0) + offs - tl.load(SCRATCH + E_PAD + e, mask=m, other=0)
    tl.store(SORTED_IDS + dest, offs.to(tl.int32), mask=m)
    order = tl.load(ORDER + offs, mask=m, other=0)
    tl.store(TOK + offs, (order // K).to(tl.int32), mask=m)
    tl.store(PAIR_W + offs, tl.load(W + order, mask=m, other=0.0), mask=m)
    nb = padded // BLOCK_M
    first = pstart // BLOCK_M
    for j in range(0, max_blocks):                                  # each expert's row blocks carry its id
        tl.store(EXPERT_IDS + first + j, bins, mask=(j < nb) & (bins < E))


def align_fused(sorted_experts: torch.Tensor, order: torch.Tensor, weights: torch.Tensor, num_experts: int,
                top_k: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`align` plus the token of each sorted pair and its routing weight (fp32), in one launch: (sorted_ids,
    expert_ids, tok, pair_w). `weights` is [T, top_k] fp32 as the router produced it."""
    M, E = sorted_experts.numel(), num_experts
    dev = sorted_experts.device
    n_blocks = (M + E * (block_m - 1)) // block_m
    sorted_ids = torch.empty(n_blocks * block_m, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(n_blocks, dtype=torch.int32, device=dev)
    tok = torch.empty(M, dtype=torch.int32, device=dev)
    pair_w = torch.empty(M, dtype=torch.float32, device=dev)
    e_pad = triton.next_power_of_2(E + 1)
    scratch = torch.empty(2 * e_pad, dtype=torch.int32, device=dev)
    block = max(triton.next_power_of_2(M), e_pad)
    _align_kernel[(1,)](sorted_experts, order, weights.reshape(-1), sorted_ids, expert_ids, tok, pair_w, scratch,
                        M, n_blocks, triton.cdiv(M, block_m), E=E, E_PAD=e_pad, K=top_k, BLOCK_M=block_m,
                        BLOCK=block, num_warps=min(16, max(1, block // 128)))
    return sorted_ids, expert_ids, tok, pair_w


@triton.jit(do_not_specialize=["M"])
def _moe_gemm_scatter_kernel(A, W, SCALE, A_SCALE, OUT, SORTED, EXPERT_IDS, TOK, PAIR_W, M, N, K,
                             stride_am, stride_wn, stride_wk, stride_om,
                             E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             SPLIT_K: tl.constexpr, W_FP8: tl.constexpr, A_FP8: tl.constexpr):
    """The down projection with its epilogue: each pair's row, times its routing weight, added into its token's
    fp32 output row (the multiply, the zero-fill-and-index_add and the cast were four launches). Split over K like
    moe_gemm: the atomics make the slices free."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(EXPERT_IDS + pid_m)
    if e >= E:
        return
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.load(SORTED + offs_m)
    valid = rows < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    k_lo = tl.program_id(2) * (K // SPLIT_K)
    a_ptrs = A + rows[:, None] * stride_am + k_lo + offs_k[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if A_FP8:                                             # W8A8, as in _moe_gemm_kernel; the row scale is per pair
        w_ptrs = W + e.to(tl.int64) * N * stride_wn + (k_lo + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn
        for k0 in range(0, K // SPLIT_K, BLOCK_K):
            k_ok = (k0 + offs_k) < K // SPLIT_K
            a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_ok[:, None] & n_ok[None, :], other=0.0)
            acc = tl.dot(a, w, acc, max_num_imprecise_acc=0)     # fp32 accumulation, as torch's grouped GEMM
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * stride_wk
        acc = acc * tl.load(A_SCALE + rows, mask=valid, other=0.0)[:, None]
    else:
        w_ptrs = W + e.to(tl.int64) * N * stride_wn + offs_n[:, None] * stride_wn + (k_lo + offs_k[None, :]) * stride_wk
        for k0 in range(0, K // SPLIT_K, BLOCK_K):
            k_ok = (k0 + offs_k) < K // SPLIT_K
            a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_ok[:, None] & k_ok[None, :], other=0.0)
            if W_FP8:
                w = w.to(tl.bfloat16)
            acc += tl.dot(a, tl.trans(w))
            a_ptrs += BLOCK_K
            w_ptrs += BLOCK_K * stride_wk
    if W_FP8:
        acc = acc * tl.load(SCALE + e.to(tl.int64) * N + offs_n, mask=n_ok, other=0.0)[None, :]
    acc = acc * tl.load(PAIR_W + rows, mask=valid, other=0.0)[:, None]
    t = tl.load(TOK + rows, mask=valid, other=0).to(tl.int64)
    tl.atomic_add(OUT + t[:, None] * stride_om + offs_n[None, :], acc, mask=valid[:, None] & n_ok[None, :])


def _a8(a: torch.Tensor, a_scale, fp8_w: bool) -> bool:
    a8 = a.dtype == torch.float8_e4m3fn
    assert a8 == (a_scale is not None), "fp8 activations come with their per-row scale"
    assert not a8 or fp8_w, "fp8 activations need fp8 weights (one tensor-core dtype)"
    return a8


def moe_gemm_scatter(a: torch.Tensor, w: torch.Tensor, sorted_ids: torch.Tensor, expert_ids: torch.Tensor, M: int,
                     tok: torch.Tensor, pair_w: torch.Tensor, out: torch.Tensor, block_m: int = 16,
                     w_scale: torch.Tensor | None = None, a_scale: torch.Tensor | None = None, tile=None) -> None:
    """out[tok[i]] += pair_w[i] * (a[i] @ w[expert(i)].T) over the sorted pairs; `out` [T, N] fp32, zeroed.
    `w` bf16, or fp8 e4m3 with `w_scale` [E, N] fp32; `a` bf16, or fp8 e4m3 with `a_scale` [M] fp32 (W8A8)."""
    E, N, K = w.shape
    fp8 = w.dtype == torch.float8_e4m3fn
    assert fp8 == (w_scale is not None), "fp8 weights come with their per-channel scale"
    a8 = _a8(a, a_scale, fp8)
    bn, bk, stages, warps = _tile(block_m, tile)
    grid = (expert_ids.numel(), triton.cdiv(N, bn))
    _moe_gemm_scatter_kernel[(*grid, 1)](a, w, w_scale if fp8 else a, a_scale if a8 else a, out, sorted_ids, expert_ids,
                                         tok, pair_w, M, N, K, a.stride(0), w.stride(1), w.stride(2), out.stride(0),
                                         E=E, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk,
                                         SPLIT_K=1, W_FP8=fp8, A_FP8=a8, num_warps=warps, num_stages=stages)


# Tiles from a 48-config sweep on the H200 at the decode shapes (M = 8 and 64 pairs): BLOCK_N 64, BLOCK_K 128, 4 stages
# took gate_up 29 -> 18 us and down 29 -> 17 us at batch 1 (the ncu diagnosis had been occupancy: splitting K bought
# nothing at any tile; stages and the tile shape did). The same config won at M = 64 (85 -> 76, 50 -> 44 us).
BLOCK_N, BLOCK_K, NUM_STAGES = 64, 128, 4


def _warps(block_m: int) -> int:
    """4 warps for the decode tiles (16 and 32 rows), 8 for 64-row tiles (a 64 x 64 fp32 accumulator per program)."""
    return 8 if block_m >= 64 else 4


def _tile(block_m: int, tile) -> tuple[int, int, int, int]:
    """(BLOCK_N, BLOCK_K, num_stages, num_warps): the decode-tuned constants, or `tile` as given (a sweep at prefill
    sizes, where every path measured 10-13% of fp8 peak on 2026-10-21, picks its own)."""
    return tuple(tile) if tile is not None else (BLOCK_N, BLOCK_K, NUM_STAGES, _warps(block_m))


def moe_gemm(a: torch.Tensor, tok: torch.Tensor | None, w: torch.Tensor, sorted_ids: torch.Tensor,
             expert_ids: torch.Tensor, M: int, block_m: int = 16, split_k: int | None = None,
             w_scale: torch.Tensor | None = None, a_scale: torch.Tensor | None = None, tile=None) -> torch.Tensor:
    """C [M, N] = per pair i: a[tok[i]] @ w[expert(i)].T, pairs in sorted order. `a` [T, K] (or [M, K] with tok None),
    `w` [E, N, K] contiguous: bf16, or fp8 e4m3 with `w_scale` [E, N] fp32; `a` bf16, or fp8 e4m3 with `a_scale` [T]
    fp32 per row (W8A8: the prefill path's quantization, on the tensor cores). Output bf16, or fp32 when the GEMM is
    split over K (small grids; `split_k` None = auto)."""
    E, N, K = w.shape
    fp8 = w.dtype == torch.float8_e4m3fn
    assert fp8 == (w_scale is not None), "fp8 weights come with their per-channel scale"
    a8 = _a8(a, a_scale, fp8)
    bn, bk, stages, warps = _tile(block_m, tile)
    grid = (expert_ids.numel(), triton.cdiv(N, bn))
    if split_k is None:
        split_k = 1
    if split_k > 1:
        C = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    else:
        C = torch.empty(M, N, dtype=torch.bfloat16 if a8 else a.dtype, device=a.device)   # never the fp8 of W8A8 rows
    _moe_gemm_kernel[(*grid, split_k)](a, tok if tok is not None else a, w, w_scale if fp8 else a, a_scale if a8 else a,
                                       C, sorted_ids, expert_ids, M, N, K, a.stride(0), w.stride(1), w.stride(2),
                                       C.stride(0), E=E, BLOCK_M=block_m, BLOCK_N=bn, BLOCK_K=bk,
                                       ROWS_FROM_TOKENS=tok is not None, SPLIT_K=split_k, W_FP8=fp8, A_FP8=a8,
                                       num_warps=warps, num_stages=stages)
    return C


@triton.jit
def _quant_rows_kernel(X, Q, S, IDX, K, stride_x, stride_q, DIV, FP8_MAX: tl.constexpr, BLOCK_K: tl.constexpr,
                       GATHER: tl.constexpr = False):
    """Row i of Q/S from row i of X, or with GATHER from row IDX[i] // DIV (the expert-sorted pairs' tokens)."""
    row = tl.program_id(0)
    src = (tl.load(IDX + row) // DIV).to(tl.int64) if GATHER else row.to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    m = cols < K
    x = tl.load(X + src * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
    scale = amax / FP8_MAX
    q = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)
    tl.store(Q + row * stride_q + cols, q.to(tl.float8e4nv), mask=m)
    tl.store(S + row, scale)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric fp8 e4m3 per row of x [M, K] (K <= 8192) in one pass: (q [M, K] fp8, scale [M] fp32), the same values
    as qwen3_moe.quantize_rowwise, which was five launches and an fp32 copy of x."""
    M, K = x.shape
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, dtype=torch.float32, device=x.device)
    _quant_rows_kernel[(M,)](x, q, s, s, K, x.stride(0), q.stride(0), 1, FP8_MAX=448.0,
                             BLOCK_K=triton.next_power_of_2(K), num_warps=8)
    return q, s


def quant_rows_gather(x: torch.Tensor, idx: torch.Tensor, div: int) -> tuple[torch.Tensor, torch.Tensor]:
    """quant_rows(x[idx // div]) in one launch: the expert-sorted pairs' token rows gathered and quantized."""
    K = x.shape[1]
    M = idx.numel()
    q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M, dtype=torch.float32, device=x.device)
    _quant_rows_kernel[(M,)](x, q, s, idx, K, x.stride(0), q.stride(0), div, FP8_MAX=448.0,
                             BLOCK_K=triton.next_power_of_2(K), GATHER=True, num_warps=8)
    return q, s


@triton.jit
def _gather_rows_kernel(X, Y, IDX, K, stride_x, DIV, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    src = (tl.load(IDX + row) // DIV).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    for c in range(0, K, BLOCK_K):
        m = (c + cols) < K
        tl.store(Y + row.to(tl.int64) * K + c + cols, tl.load(X + src * stride_x + c + cols, mask=m), mask=m)


def gather_rows(x: torch.Tensor, idx: torch.Tensor, div: int) -> torch.Tensor:
    """x[idx // div] in one launch."""
    K = x.shape[1]
    y = torch.empty(idx.numel(), K, dtype=x.dtype, device=x.device)
    _gather_rows_kernel[(idx.numel(),)](x, y, idx, K, x.stride(0), div, BLOCK_K=min(4096, triton.next_power_of_2(K)))
    return y


_BLOCK_Z = 1024


@triton.jit
def _zero_rows(OUT, t, tm, N, BLOCK_T: tl.constexpr, BLOCK_Z: tl.constexpr):
    """OUT[t, :N] = 0 for the tokens t (mask tm): the down GEMM's fp32 accumulator, zeroed by the router before it."""
    z = tl.arange(0, BLOCK_Z)
    for c in range(0, N, BLOCK_Z):
        tl.store(OUT + t[:, None].to(tl.int64) * N + c + z[None, :], tl.zeros((BLOCK_T, BLOCK_Z), tl.float32),
                 mask=tm[:, None] & ((c + z) < N)[None, :])


@triton.jit(do_not_specialize=["T", "n_blocks", "max_blocks"])
def _route_kernel(LOGITS, SORTED_IDS, EXPERT_IDS, TOK, PAIR_W, SCRATCH, OUT, stride_l, T, n_blocks, max_blocks, N_OUT,
                  E: tl.constexpr, E_PAD: tl.constexpr, K: tl.constexpr, RENORM: tl.constexpr,
                  BLOCK_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK: tl.constexpr, BLOCK_Z: tl.constexpr):
    """The whole router for a decode-sized batch in one program: softmax and top-K per token, then the (token, slot)
    pairs grouped by expert with a counting sort (each pair's rank among its expert's pairs) and padded per expert to
    BLOCK_M rows: the same outputs as topk_softmax + sort + align_fused (four launches).
    SCRATCH int32/fp32 views: [BLOCK] pair experts, [BLOCK] pair weights, [BLOCK_T] per-token weight sums,
    [E_PAD] padded starts."""
    t = tl.arange(0, BLOCK_T)
    tm = t < T
    _zero_rows(OUT, t, tm, N_OUT, BLOCK_T, BLOCK_Z)
    cols = tl.arange(0, E_PAD)
    m2 = tm[:, None] & (cols < E)[None, :]
    x = tl.load(LOGITS + t[:, None] * stride_l + cols[None, :], mask=m2, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    p = tl.where(m2, p, -1.0)
    total = tl.zeros((BLOCK_T,), tl.float32)
    for j in tl.static_range(K):
        v = tl.max(p, axis=1)
        # A NaN row (a graph's dummy rows) matches nothing: clamp so the id stays a real expert.
        idx = tl.minimum(tl.min(tl.where(p == v[:, None], cols[None, :], E_PAD), axis=1), E - 1)
        total += v
        tl.store(SCRATCH + t * K + j, idx.to(tl.int32), mask=tm)
        tl.store(SCRATCH.to(tl.pointer_type(tl.float32)) + BLOCK + t * K + j, v, mask=tm)
        p = tl.where(cols[None, :] == idx[:, None], -1.0, p)
    tl.store(SCRATCH.to(tl.pointer_type(tl.float32)) + 2 * BLOCK + t, total, mask=tm)
    tl.debug_barrier()
    offs = tl.arange(0, BLOCK)                                  # pair i = token * K + slot
    M = T * K
    pm = offs < M
    e = tl.load(SCRATCH + offs, mask=pm, other=E)
    tok = offs // K
    w = tl.load(SCRATCH.to(tl.pointer_type(tl.float32)) + BLOCK + offs, mask=pm, other=0.0)
    if RENORM:
        w = w / tl.load(SCRATCH.to(tl.pointer_type(tl.float32)) + 2 * BLOCK + tok, mask=pm, other=1.0)
    bins = tl.arange(0, E_PAD)
    counts = tl.where(bins < E, tl.histogram(e, E_PAD), 0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    pstart = tl.cumsum(padded, 0) - padded
    tl.store(SCRATCH + 3 * BLOCK + bins, pstart)
    for s in range(0, n_blocks * BLOCK_M, BLOCK):              # padding rows read M, blocks past the end E
        tl.store(SORTED_IDS + s + offs, tl.zeros((BLOCK,), tl.int32) + M, mask=(s + offs) < n_blocks * BLOCK_M)
    for s in range(0, n_blocks, BLOCK):
        tl.store(EXPERT_IDS + s + offs, tl.zeros((BLOCK,), tl.int32) + E, mask=(s + offs) < n_blocks)
    tl.debug_barrier()
    same = (e[:, None] == e[None, :]) & (offs[None, :] < offs[:, None]) & pm[None, :]
    rank = tl.sum(same.to(tl.int32), axis=1)                   # pairs of the same expert before this one
    dest = tl.load(SCRATCH + 3 * BLOCK + e, mask=pm, other=0) + rank
    tl.store(SORTED_IDS + dest, offs.to(tl.int32), mask=pm)
    tl.store(TOK + offs, tok.to(tl.int32), mask=pm)
    tl.store(PAIR_W + offs, w, mask=pm)
    nb = padded // BLOCK_M
    first = pstart // BLOCK_M
    for j in range(0, max_blocks):                              # each expert's row blocks carry its id
        tl.store(EXPERT_IDS + first + j, bins, mask=(j < nb) & (bins < E))


def route_fused(logits: torch.Tensor, num_experts: int, top_k: int, renorm: bool,
                block_m: int, out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """topk_softmax + sort + align_fused in one launch for T tokens (decode-sized: the rank step compares every pair
    with every other): (sorted_ids, expert_ids, tok, pair_w), pairs numbered token * top_k + slot. `out` (fp32
    [T, N], the down GEMM's accumulator) is zeroed on the way."""
    T, E = logits.shape
    assert out is None or (out.is_contiguous() and out.shape[0] == T and out.dtype == torch.float32)
    M = T * top_k
    dev = logits.device
    n_blocks = (M + E * (block_m - 1)) // block_m
    sorted_ids = torch.empty(n_blocks * block_m, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(n_blocks, dtype=torch.int32, device=dev)
    tok = torch.empty(M, dtype=torch.int32, device=dev)
    pair_w = torch.empty(M, dtype=torch.float32, device=dev)
    e_pad = triton.next_power_of_2(E + 1)
    block = max(triton.next_power_of_2(M), e_pad)
    block_t = max(triton.next_power_of_2(T), 1)
    scratch = torch.empty(3 * block + e_pad, dtype=torch.int32, device=dev)
    _route_kernel[(1,)](logits, sorted_ids, expert_ids, tok, pair_w, scratch, out if out is not None else scratch,
                        logits.stride(0), T, n_blocks, triton.cdiv(M, block_m), out.shape[1] if out is not None else 0,
                        E=E, E_PAD=e_pad, K=top_k, RENORM=renorm, BLOCK_T=block_t, BLOCK_M=block_m, BLOCK=block,
                        BLOCK_Z=_BLOCK_Z, num_warps=min(16, max(4, block // 64)))
    return sorted_ids, expert_ids, tok, pair_w


@triton.jit(do_not_specialize=["T", "M", "n_blocks"])
def _route_topk_kernel(LOGITS, E_IDS, TOK, PAIR_W, SORTED_IDS, EXPERT_IDS, OUT, stride_l, T, M, n_blocks, N_OUT,
                       E: tl.constexpr, E_PAD: tl.constexpr, K: tl.constexpr, RENORM: tl.constexpr,
                       BLOCK_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_Z: tl.constexpr):
    """route_multi, step 1 (a program per BLOCK_T tokens): softmax and top-K, each pair's expert / token / weight,
    this program's share of the padding fill (rows read M, blocks E), and its tokens' rows of OUT zeroed."""
    f = tl.arange(0, BLOCK_F)
    for s in range(tl.program_id(0) * BLOCK_F, n_blocks * BLOCK_M, tl.num_programs(0) * BLOCK_F):
        tl.store(SORTED_IDS + s + f, tl.zeros((BLOCK_F,), tl.int32) + M, mask=(s + f) < n_blocks * BLOCK_M)
    for s in range(tl.program_id(0) * BLOCK_F, n_blocks, tl.num_programs(0) * BLOCK_F):
        tl.store(EXPERT_IDS + s + f, tl.zeros((BLOCK_F,), tl.int32) + E, mask=(s + f) < n_blocks)
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tm = t < T
    _zero_rows(OUT, t, tm, N_OUT, BLOCK_T, BLOCK_Z)
    cols = tl.arange(0, E_PAD)
    m2 = tm[:, None] & (cols < E)[None, :]
    x = tl.load(LOGITS + t[:, None] * stride_l + cols[None, :], mask=m2, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, axis=1)[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    p = tl.where(m2, p, -1.0)
    total = tl.zeros((BLOCK_T,), tl.float32)
    for j in tl.static_range(K):                                # the same picks and tie order as _route_kernel
        v = tl.max(p, axis=1)
        idx = tl.minimum(tl.min(tl.where(p == v[:, None], cols[None, :], E_PAD), axis=1), E - 1)
        total += v
        tl.store(E_IDS + t * K + j, idx.to(tl.int32), mask=tm)
        tl.store(PAIR_W + t * K + j, v, mask=tm)
        tl.store(TOK + t * K + j, t.to(tl.int32), mask=tm)
        p = tl.where(cols[None, :] == idx[:, None], -1.0, p)
    if RENORM:
        tl.debug_barrier()                                      # the stores above, by whichever threads made them
        for j in tl.static_range(K):
            w = tl.load(PAIR_W + t * K + j, mask=tm, other=0.0)
            tl.store(PAIR_W + t * K + j, w / total, mask=tm)


@triton.jit(do_not_specialize=["M", "max_blocks"])
def _route_offsets_kernel(E_IDS, PSTART, CURSOR, EXPERT_IDS, M, max_blocks, E: tl.constexpr, E_PAD: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK: tl.constexpr, DENSE: tl.constexpr = False):
    """route_multi, step 2 (one program): the pairs per expert (a histogram, so no buffer needs zeroing first), each
    expert's padded first row, its blocks' expert id, and the place step's cursors reset. DENSE (route_dense): no
    padding, and EXPERT_IDS takes each expert's group end (the grouped GEMM's offsets) instead."""
    bins = tl.arange(0, E_PAD)
    offs = tl.arange(0, BLOCK)
    counts = tl.zeros((E_PAD,), tl.int32)
    for s in range(0, M, BLOCK):
        counts += tl.histogram(tl.load(E_IDS + s + offs, mask=(s + offs) < M, other=E), E_PAD)
    counts = tl.where(bins < E, counts, 0)
    tl.store(CURSOR + bins, tl.zeros((E_PAD,), tl.int32))
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    pstart = tl.cumsum(padded, 0) - padded
    tl.store(PSTART + bins, pstart)
    if DENSE:
        tl.store(EXPERT_IDS + bins, (pstart + counts).to(tl.int32), mask=bins < E)
    else:
        nb = padded // BLOCK_M
        first = pstart // BLOCK_M
        for j in range(0, max_blocks):
            tl.store(EXPERT_IDS + first + j, bins, mask=(j < nb) & (bins < E))


@triton.jit(do_not_specialize=["M"])
def _route_place_kernel(E_IDS, PSTART, CURSOR, SORTED_IDS, POS, M, BLOCK: tl.constexpr, HAS_POS: tl.constexpr = False):
    """route_multi, step 3 (a program per BLOCK pairs): each pair takes the next row of its expert's run (and, with
    HAS_POS, records that row as its position)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < M
    e = tl.load(E_IDS + i, mask=m, other=0)
    rank = tl.atomic_add(CURSOR + e, tl.full((BLOCK,), 1, tl.int32), mask=m)
    dest = tl.load(PSTART + e, mask=m, other=0) + rank
    tl.store(SORTED_IDS + dest, i.to(tl.int32), mask=m)
    if HAS_POS:
        tl.store(POS + i, dest.to(tl.int32), mask=m)


def route_multi(logits: torch.Tensor, num_experts: int, top_k: int, renorm: bool,
                block_m: int, out: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """route_fused's outputs for any number of tokens, without a sort (topk_softmax + torch.sort + align_fused took
    radixSortKVInPlace alone to 5.4% of a 256-row decode step, H200 2026-10-19): a counting sort over three launches
    and no fill. Pairs within an expert's run land in an arbitrary order; each row's GEMM output does not depend on
    its position. `out` as route_fused's."""
    T, E = logits.shape
    assert out is None or (out.is_contiguous() and out.shape[0] == T and out.dtype == torch.float32)
    M = T * top_k
    dev = logits.device
    n_blocks = (M + E * (block_m - 1)) // block_m
    sorted_ids = torch.empty(n_blocks * block_m, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(n_blocks, dtype=torch.int32, device=dev)
    tok = torch.empty(M, dtype=torch.int32, device=dev)
    pair_w = torch.empty(M, dtype=torch.float32, device=dev)
    e_ids = torch.empty(M, dtype=torch.int32, device=dev)
    e_pad = triton.next_power_of_2(E + 1)
    counters = torch.empty(2 * e_pad, dtype=torch.int32, device=dev)          # cursors (reset by step 2), padded starts
    cursor, pstart = counters[:e_pad], counters[e_pad:]
    BLOCK_T = 16
    _route_topk_kernel[(triton.cdiv(T, BLOCK_T),)](logits, e_ids, tok, pair_w, sorted_ids, expert_ids,
                                                   out if out is not None else pair_w, logits.stride(0), T, M,
                                                   n_blocks, out.shape[1] if out is not None else 0, E=E, E_PAD=e_pad,
                                                   K=top_k, RENORM=renorm, BLOCK_T=BLOCK_T, BLOCK_M=block_m,
                                                   BLOCK_F=256, BLOCK_Z=_BLOCK_Z, num_warps=4)
    _route_offsets_kernel[(1,)](e_ids, pstart, cursor, expert_ids, M, triton.cdiv(M, block_m), E=E, E_PAD=e_pad,
                                BLOCK_M=block_m, BLOCK=2048, num_warps=8)
    BLOCK = 1024
    _route_place_kernel[(triton.cdiv(M, BLOCK),)](e_ids, pstart, cursor, sorted_ids, sorted_ids, M, BLOCK=BLOCK,
                                                  num_warps=4)
    return sorted_ids, expert_ids, tok, pair_w


def route_dense(logits: torch.Tensor, num_experts: int, top_k: int,
                renorm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The routing for the grouped-GEMM path, unpadded, in three launches: (sorted_ids [M] pair ids grouped by
    expert, offs [E] int32 each group's end, pos [M] each pair's row in that order, pair_w [M] fp32), pairs numbered
    token * top_k + slot. Replaces topk_softmax + torch.sort + searchsorted + the inverse-permutation scatter."""
    T, E = logits.shape
    M = T * top_k
    dev = logits.device
    sorted_ids = torch.empty(M, dtype=torch.int32, device=dev)
    pos = torch.empty(M, dtype=torch.int32, device=dev)
    offs = torch.empty(E, dtype=torch.int32, device=dev)
    tok = torch.empty(M, dtype=torch.int32, device=dev)
    pair_w = torch.empty(M, dtype=torch.float32, device=dev)
    e_ids = torch.empty(M, dtype=torch.int32, device=dev)
    e_pad = triton.next_power_of_2(E + 1)
    counters = torch.empty(2 * e_pad, dtype=torch.int32, device=dev)
    cursor, pstart = counters[:e_pad], counters[e_pad:]
    BLOCK_T = 16
    _route_topk_kernel[(triton.cdiv(T, BLOCK_T),)](logits, e_ids, tok, pair_w, sorted_ids, offs, pair_w,
                                                   logits.stride(0), T, M, 0, 0, E=E, E_PAD=e_pad, K=top_k,
                                                   RENORM=renorm, BLOCK_T=BLOCK_T, BLOCK_M=1, BLOCK_F=256,
                                                   BLOCK_Z=_BLOCK_Z, num_warps=4)
    _route_offsets_kernel[(1,)](e_ids, pstart, cursor, offs, M, 0, E=E, E_PAD=e_pad, BLOCK_M=1, BLOCK=2048,
                                DENSE=True, num_warps=8)
    BLOCK = 1024
    _route_place_kernel[(triton.cdiv(M, BLOCK),)](e_ids, pstart, cursor, sorted_ids, pos, M, BLOCK=BLOCK,
                                                  HAS_POS=True, num_warps=4)
    return sorted_ids, offs, pos, pair_w


@triton.jit(do_not_specialize=["M"])
def _scatter_add_weighted_kernel(Y, TOK, PAIR_W, OUT, M, N, stride_y, stride_o,
                                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """out[tok[i]] += pair_w[i] * y[i] for a tile of rows: the MoE epilogue of the CUTLASS path in one pass."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    rm, cm = rows < M, cols < N
    y = tl.load(Y + rows[:, None] * stride_y + cols[None, :], mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
    w = tl.load(PAIR_W + rows, mask=rm, other=0.0)
    t = tl.load(TOK + rows, mask=rm, other=0).to(tl.int64)
    tl.atomic_add(OUT + t[:, None] * stride_o + cols[None, :], y * w[:, None], mask=rm[:, None] & cm[None, :])


def scatter_add_weighted(y: torch.Tensor, tok: torch.Tensor, pair_w: torch.Tensor, out: torch.Tensor) -> None:
    """out [T, N] fp32 += each row of y [M, N] (bf16) times its routing weight, at its token. At 8k tokens the torch
    version (multiply into fp32, cast, zero-fill, index_add) was ~0.9 ms a layer for ~0.3 GB of traffic."""
    M, N = y.shape
    BLOCK_M, BLOCK_N = 32, 256
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _scatter_add_weighted_kernel[grid](y, tok, pair_w, out, M, N, y.stride(0), out.stride(0),
                                       BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)


@triton.jit
def _gather_weighted_kernel(Y, POS, W, OUT, RES, N, stride_y, stride_o, stride_r, K: tl.constexpr,
                            BLOCK_N: tl.constexpr, HAS_RES: tl.constexpr = False):
    """out[t] = sum_j w[t, j] * y[pos[t, j]]: one token's K expert rows gathered and summed, no atomics, bf16 out;
    with HAS_RES, bf16(res[t] + that bf16 sum), the residual add the layer makes next."""
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    cm = cols < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in tl.static_range(K):
        pos = tl.load(POS + t * K + j).to(tl.int64)
        w = tl.load(W + t * K + j)
        acc += w * tl.load(Y + pos * stride_y + cols, mask=cm, other=0.0).to(tl.float32)
    out = acc.to(tl.bfloat16)
    if HAS_RES:
        out = (tl.load(RES + t * stride_r + cols, mask=cm, other=0.0).to(tl.float32) + out.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + t * stride_o + cols, out, mask=cm)


def gather_weighted(y: torch.Tensor, pos: torch.Tensor, weights: torch.Tensor,
                    residual: torch.Tensor | None = None) -> torch.Tensor:
    """[T, N] bf16: each token's routed rows of y [M, N] (at sorted positions pos [T, K]) times its weights [T, K],
    summed. The MoE epilogue of the CUTLASS path as one read of y and one write of the output: the atomic
    scatter-add version ran at 456 us a layer for 8k tokens (8 fp32 atomics per output element)."""
    T, K = pos.shape
    N = y.shape[1]
    out = torch.empty(T, N, dtype=torch.bfloat16, device=y.device)
    BLOCK_N = 256
    res = residual.reshape(T, N) if residual is not None else out
    _gather_weighted_kernel[(T, triton.cdiv(N, BLOCK_N))](y, pos, weights, out, res, N, y.stride(0), out.stride(0),
                                                           res.stride(0), K=K, BLOCK_N=BLOCK_N,
                                                           HAS_RES=residual is not None, num_warps=4)
    return out
