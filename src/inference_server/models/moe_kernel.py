"""Triton grouped GEMM for the MoE experts at small M (decode).

torch._grouped_mm (CUTLASS) took ~36 us per call on an H200 for a batch-1 decode step: 8 active experts, one row
each, but a 128-group launch. That was 3.5 ms of a ~13 ms step against a ~0.9 ms weight-read floor. This kernel is
vLLM's fused_moe layout: (token, slot) pairs sorted by expert and padded per expert to BLOCK_M rows, one program per
(row block, N tile); each program reads only its expert's weight tile, so the bytes read are the active experts'
weights, once. Rows come straight from the hidden states through the token map (no gathered copy). bf16 only;
fp8 weights and large M keep the CUTLASS path."""

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


@triton.jit
def _moe_gemm_kernel(A, TOK, W, C, SORTED, EXPERT_IDS, M, N, K,
                     stride_am, stride_wn, stride_wk, stride_cm,
                     E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     ROWS_FROM_TOKENS: tl.constexpr):
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
    a_ptrs = A + src[:, None] * stride_am + offs_k[None, :]
    w_ptrs = W + e.to(tl.int64) * N * stride_wn + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_ok = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=n_ok[:, None] & k_ok[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = C + rows[:, None] * stride_cm + offs_n[None, :]
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


@triton.jit
def _moe_gemm_scatter_kernel(A, W, OUT, SORTED, EXPERT_IDS, TOK, PAIR_W, M, N, K,
                             stride_am, stride_wn, stride_wk, stride_om,
                             E: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """The down projection with its epilogue: each pair's row, times its routing weight, added into its token's
    fp32 output row (the multiply, the zero-fill-and-index_add and the cast were four launches)."""
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
    a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :]
    w_ptrs = W + e.to(tl.int64) * N * stride_wn + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_ok = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=n_ok[:, None] & k_ok[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(w))
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K * stride_wk
    acc = acc * tl.load(PAIR_W + rows, mask=valid, other=0.0)[:, None]
    t = tl.load(TOK + rows, mask=valid, other=0).to(tl.int64)
    tl.atomic_add(OUT + t[:, None] * stride_om + offs_n[None, :], acc, mask=valid[:, None] & n_ok[None, :])


def moe_gemm_scatter(a: torch.Tensor, w: torch.Tensor, sorted_ids: torch.Tensor, expert_ids: torch.Tensor, M: int,
                     tok: torch.Tensor, pair_w: torch.Tensor, out: torch.Tensor, block_m: int = 16) -> None:
    """out[tok[i]] += pair_w[i] * (a[i] @ w[expert(i)].T) over the sorted pairs; `out` [T, N] fp32, zeroed."""
    E, N, K = w.shape
    BLOCK_N, BLOCK_K = 128, 64
    grid = (expert_ids.numel(), triton.cdiv(N, BLOCK_N))
    _moe_gemm_scatter_kernel[grid](a, w, out, sorted_ids, expert_ids, tok, pair_w, M, N, K,
                                   a.stride(0), w.stride(1), w.stride(2), out.stride(0),
                                   E=E, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4, num_stages=3)


def moe_gemm(a: torch.Tensor, tok: torch.Tensor | None, w: torch.Tensor, sorted_ids: torch.Tensor,
             expert_ids: torch.Tensor, M: int, block_m: int = 16) -> torch.Tensor:
    """C [M, N] = per pair i: a[tok[i]] @ w[expert(i)].T, pairs in sorted order. `a` [T, K] (or [M, K] with tok None),
    `w` [E, N, K] bf16 contiguous."""
    E, N, K = w.shape
    C = torch.empty(M, N, dtype=a.dtype, device=a.device)
    BLOCK_N, BLOCK_K = 128, 64
    grid = (expert_ids.numel(), triton.cdiv(N, BLOCK_N))
    _moe_gemm_kernel[grid](a, tok if tok is not None else a, w, C, sorted_ids, expert_ids, M, N, K,
                           a.stride(0), w.stride(1), w.stride(2), C.stride(0),
                           E=E, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                           ROWS_FROM_TOKENS=tok is not None, num_warps=4, num_stages=3)
    return C


@triton.jit
def _route_kernel(LOGITS, SORTED_IDS, EXPERT_IDS, TOK, PAIR_W, SCRATCH, stride_l, T, n_blocks, max_blocks,
                  E: tl.constexpr, E_PAD: tl.constexpr, K: tl.constexpr, RENORM: tl.constexpr,
                  BLOCK_T: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK: tl.constexpr):
    """The whole router for a decode-sized batch in one program: softmax and top-K per token, then the (token, slot)
    pairs grouped by expert with a counting sort (each pair's rank among its expert's pairs) and padded per expert to
    BLOCK_M rows: the same outputs as topk_softmax + sort + align_fused (four launches).
    SCRATCH int32/fp32 views: [BLOCK] pair experts, [BLOCK] pair weights, [BLOCK_T] per-token weight sums,
    [E_PAD] padded starts."""
    t = tl.arange(0, BLOCK_T)
    cols = tl.arange(0, E_PAD)
    tm = t < T
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
                block_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """topk_softmax + sort + align_fused in one launch for T tokens (decode-sized: the rank step compares every pair
    with every other): (sorted_ids, expert_ids, tok, pair_w), pairs numbered token * top_k + slot."""
    T, E = logits.shape
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
    _route_kernel[(1,)](logits, sorted_ids, expert_ids, tok, pair_w, scratch, logits.stride(0), T, n_blocks,
                        triton.cdiv(M, block_m), E=E, E_PAD=e_pad, K=top_k, RENORM=renorm, BLOCK_T=block_t,
                        BLOCK_M=block_m, BLOCK=block, num_warps=min(16, max(4, block // 64)))
    return sorted_ids, expert_ids, tok, pair_w
