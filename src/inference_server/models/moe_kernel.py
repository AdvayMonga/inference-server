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
                     ROWS_FROM_TOKENS: tl.constexpr, EVEN_K: tl.constexpr, EVEN_N: tl.constexpr):
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
        if EVEN_K:
            a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
            if EVEN_N:
                w = tl.load(w_ptrs, eviction_policy="evict_first")
            else:
                w = tl.load(w_ptrs, mask=n_ok[:, None], other=0.0, eviction_policy="evict_first")
        else:
            k_ok = (k0 + offs_k) < K
            a = tl.load(a_ptrs, mask=valid[:, None] & k_ok[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_ok[:, None] & k_ok[None, :], other=0.0, eviction_policy="evict_first")
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


def moe_gemm(a: torch.Tensor, tok: torch.Tensor | None, w: torch.Tensor, sorted_ids: torch.Tensor,
             expert_ids: torch.Tensor, M: int, block_m: int = 16) -> torch.Tensor:
    """C [M, N] = per pair i: a[tok[i]] @ w[expert(i)].T, pairs in sorted order. `a` [T, K] (or [M, K] with tok None),
    `w` [E, N, K] bf16 contiguous."""
    E, N, K = w.shape
    C = torch.empty(M, N, dtype=a.dtype, device=a.device)
    BLOCK_N, BLOCK_K = 64, 128
    grid = (expert_ids.numel(), triton.cdiv(N, BLOCK_N))
    _moe_gemm_kernel[grid](a, tok if tok is not None else a, w, C, sorted_ids, expert_ids, M, N, K,
                           a.stride(0), w.stride(1), w.stride(2), C.stride(0),
                           E=E, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                           ROWS_FROM_TOKENS=tok is not None,
                           EVEN_K=(K % BLOCK_K == 0), EVEN_N=(N % BLOCK_N == 0),
                           num_warps=4, num_stages=4)
    return C
