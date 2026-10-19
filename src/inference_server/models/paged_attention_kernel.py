"""Triton paged-attention DECODE kernel (M2.3).

Replaces the gather+pad+SDPA decode path: reads each sequence's K/V directly from the
scattered physical blocks via its block table, in-kernel, with no dense [N,H,Lmax,D]
materialization and no padding. One query token per sequence (decode), FlashAttention-style
online softmax over the sequence's blocks.

GQA (Hq query heads share num_kv_heads KV heads), the caller's softmax scale, and any power-of-two
head_dim via a `D` constexpr (Qwen3-MoE: 128, GQA 8:1).

CUDA-only (Triton). Import lazily on the GPU path; do not import on CPU/MPS.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from inference_server.models import launch_table

# Tiled prefill attention. On by default; CUSTOM_BACKEND_TILED_PREFILL=0 falls back to the
# one-program-per-query kernel for A/B.
# DEFAULT OFF. The kernel is correct and 10.84x faster in isolation, but a properly replicated
# A/B (4 runs per arm, ABBA order, first discarded as warmup) found NO end-to-end benefit and a
# significant +24.3% REGRESSION on ttft_prefill_p95. An isolated win that does not reach the
# service level is not a win. Set CUSTOM_BACKEND_TILED_PREFILL=1 to re-test.
#
# Open question: that A/B necessarily ran with the prefill CUDA graph OFF, because a flag
# consumed inside a captured graph cannot be toggled at runtime. With the graph ON, dispatch is
# removed and attention is 63% of prefill, so tiling could matter more there — untested, and
# untestable by in-process toggle. Testing it needs capture-time variants.
_TILED_PREFILL = os.environ.get("CUSTOM_BACKEND_TILED_PREFILL", "0") == "1"


# Where tiling actually wins, measured on A100 (Gemma head dims; the Modal scripts are gone):
#
#   tokens   D    untiled    tiled   speedup
#      240  256    0.392     0.470     0.83x
#      512  256    1.545     1.222     1.26x
#      824  256    3.154     2.004     1.57x
#      240  512    0.328     0.686     0.48x
#      512  512    1.288     1.978     0.65x
#      824  512    2.618     4.967     0.53x
#
# The win needs BOTH enough queries to amortise the K/V load AND a head_dim small enough that
# the [BLOCK_M, D] accumulator stays in registers. At D=512 the tile must shrink to 16, which
# both halves the reuse and spills — tiling is consistently WORSE there. So route, do not
# blanket-enable.
TILED_MIN_TOKENS = 512
TILED_MAX_HEAD_DIM = 256


# (BLOCK_M, num_warps) per head_dim. The accumulator is [BLOCK_M, D] fp32 spread across the
# block's threads, so warps and tile size trade off against the register file together — leaving
# num_warps to Triton's default heuristic is what made D=512 spill.
# Swept on A100 (Modal sweep, script gone), untiled = 1.0x:
#
#   D=256 S=824   BLOCK_M=16 w=4  10.84x  | BLOCK_M=32 w=4  1.81x  | BLOCK_M=64 w=8  0.83x
#   D=256 S=512   BLOCK_M=16 w=4   8.10x  | BLOCK_M=32 w=4  1.01x
#   D=512 S=824   BLOCK_M=16 w=4   0.59x  | nothing beats untiled at any config
#
# BLOCK_M=16 beats 32 by ~6x, which is the opposite of the "bigger tile = more reuse" intuition:
# past a point the [BLOCK_M, D] accumulator costs more in occupancy than the reuse is worth.
# Hand-picking 32 is what made the first end-to-end A/B read as noise.
#
# D=512 is absent deliberately: no launch config wins there, so those layers keep the untiled
# kernel until a chunked-head-dim variant exists.
LAUNCH_BY_HEAD_DIM = {256: (16, 4)}


def prefill_launch(head_dim: int, n_tokens: int) -> tuple[int, int]:
    """(queries per program, num_warps). BLOCK_M 0 means use the untiled kernel."""
    if not _TILED_PREFILL or n_tokens < TILED_MIN_TOKENS:
        return 0, 4
    cfg = LAUNCH_BY_HEAD_DIM.get(head_dim)
    return cfg if cfg else (0, 4)


def prefill_block_m(head_dim: int, n_tokens: int) -> int:
    return prefill_launch(head_dim, n_tokens)[0]


@triton.jit
def _paged_decode_kernel(
    Q, Kp, Vp, BT, SL, Out,
    sq_n, sq_h,            # Q strides: [N, Hq, D] (last dim contiguous)
    sk_b, sk_h, sk_s,      # pool strides: [num_blocks, num_kv_heads, block_size, D]
    so_n, so_h,            # Out strides
    bt_n,                  # block_tables row stride: [N, max_blocks]
    scale,
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

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    # Runtime (per-sequence) loop bounds — NOT constexpr. A constexpr here would recompile
    # the kernel for every distinct block-count, thrashing under varied/growing seq lengths.
    for b in range(0, n_blocks):
        blk = tl.load(BT + seq * bt_n + b)
        offs = b * BLOCK_SIZE + slots
        valid = offs < L                                        # causal upper bound

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
    scale,
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

    # Even split of this row's blocks across SPLITS programs (runtime bounds).
    per = (n_blocks + SPLITS - 1) // SPLITS
    b0 = sp * per
    b1 = tl.minimum(b0 + per, n_blocks)

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    if b0 < b1:
        q = tl.load(Q + seq * sq_n + qh * sq_h + d).to(tl.float32)
        for b in range(b0, b1):
            blk = tl.load(BT + seq * bt_n + b)
            offs = b * BLOCK_SIZE + slots
            valid = offs < L
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


CONFIG = {'target_programs': 256, 'max_splits': 16, 'min_splits': 1, 'num_warps': 4, 'chunk_units': 32, 'stages': 3}
TUNABLES = {
    "target_programs": [256],
    "max_splits": [16],
    "min_splits": [1],
    "num_warps": [4],
    "chunk_units": [8, 16, 32],
    "stages": [2, 3],
}

_PD_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct PDParams {
  const void* q; const void* kp; const void* vp;
  const int* bt; const int* sl;
  void* out; float* pacc; float* pml; int* counters;
  long long bt_stride, sk_b, sk_h, sv_b, sv_h, sq_n, sq_h, so_n, so_h;
  int Hkv, group, chunk, splits, log2bs;
  float scale_log2;
};

__device__ __forceinline__ uint32_t pd_smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void pd_cp16(uint32_t dst, const void* src, int nbytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(nbytes) : "memory");
}
__device__ __forceinline__ void pd_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int NPEND> __device__ __forceinline__ void pd_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(NPEND) : "memory");
}
__device__ __forceinline__ void pd_ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ void pd_ldsm4t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a) : "memory");
}
__device__ __forceinline__ float pd_ex2(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y;
}

template <typename T> struct PDT;
template <> struct PDT<__nv_bfloat16> {
  static __device__ __forceinline__ uint32_t pack(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b); return *reinterpret_cast<uint32_t*>(&v);
  }
  static __device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    const uint32_t z = 0u;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0), "r"(b1));
  }
  static __device__ __forceinline__ __nv_bfloat16 cvt(float x) { return __float2bfloat16_rn(x); }
};
template <> struct PDT<__half> {
  static __device__ __forceinline__ uint32_t pack(float a, float b) {
    __half2 v = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&v);
  }
  static __device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    const uint32_t z = 0u;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(z), "r"(a2), "r"(z), "r"(b0), "r"(b1));
  }
  static __device__ __forceinline__ __half cvt(float x) { return __float2half_rn(x); }
};

template <typename T>
__device__ __forceinline__ void pd_store8(T* dst, const float* o, float inv) {
  uint4 v;
  v.x = PDT<T>::pack(o[0] * inv, o[1] * inv);
  v.y = PDT<T>::pack(o[2] * inv, o[3] * inv);
  v.z = PDT<T>::pack(o[4] * inv, o[5] * inv);
  v.w = PDT<T>::pack(o[6] * inv, o[7] * inv);
  *reinterpret_cast<uint4*>(dst) = v;
}

template <typename T, int STAGES>
__global__ void __launch_bounds__(128) pd_mma_kernel(PDParams p) {
  extern __shared__ __align__(128) unsigned char smem[];
  const T* __restrict__ q = reinterpret_cast<const T*>(p.q);
  const T* __restrict__ kp = reinterpret_cast<const T*>(p.kp);
  const T* __restrict__ vp = reinterpret_cast<const T*>(p.vp);
  T* __restrict__ out = reinterpret_cast<T*>(p.out);

  const int split = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int gid = lane >> 2, tig = lane & 3;
  const int group = p.group;
  const int L = p.sl[seq];
  const int n_units = L > 0 ? ((L + 15) >> 4) : 0;

  if (n_units <= 0) {
    if (split == 0) {
      const float nanv = __int_as_float(0x7fc00000);
      for (int idx = tid; idx < group * 128; idx += 128) {
        out[(long long)seq * p.so_n + (long long)(kvh * group + (idx >> 7)) * p.so_h + (idx & 127)] = PDT<T>::cvt(nanv);
      }
    }
    return;
  }
  const int chunk = p.chunk;
  const int cs = split * chunk;
  if (cs >= n_units) return;
  const int ce = min(cs + chunk, n_units);
  const int nsplit = (n_units + chunk - 1) / chunk;
  const int log2bs = p.log2bs;
  const int bsm1 = (1 << log2bs) - 1;
  const int* btrow = p.bt + (long long)seq * p.bt_stride;
  const float NEG_INF = __int_as_float(0xff800000);

  uint32_t qa[8][2];
  {
    const bool hv = gid < group;
    const T* qrow = q + (long long)seq * p.sq_n + (long long)(kvh * group + (hv ? gid : 0)) * p.sq_h;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int d0 = ks * 16 + tig * 2;
      qa[ks][0] = hv ? *reinterpret_cast<const uint32_t*>(qrow + d0) : 0u;
      qa[ks][1] = hv ? *reinterpret_cast<const uint32_t*>(qrow + d0 + 8) : 0u;
    }
  }

  const int rem = ce - cs - warp;
  const int my_n = rem > 0 ? ((rem + 3) >> 2) : 0;
  int myphys = 0;
  if (lane < my_n) {
    const int u = cs + warp + 4 * lane;
    myphys = btrow[(u << 4) >> log2bs];
  }
  const uint32_t wbase = pd_smem_u32(smem) + warp * (STAGES * 8192);

  auto load_unit = [&](int i, int st) {
    const int u = cs + warp + 4 * i;
    int phys = __shfl_sync(0xffffffffu, myphys, i & 31);
    if (i >= 32) phys = btrow[(u << 4) >> log2bs];
    const int tok0 = u << 4;
    const int off = tok0 & bsm1;
    const T* kb = kp + (long long)phys * p.sk_b + (long long)kvh * p.sk_h + (long long)off * 128;
    const T* vb = vp + (long long)phys * p.sv_b + (long long)kvh * p.sv_h + (long long)off * 128;
    const uint32_t ksm = wbase + st * 8192, vsm = ksm + 4096;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int j = lane + 32 * t;
      const int r = j >> 4, c = j & 15;
      const int nb = (tok0 + r < L) ? 16 : 0;
      const uint32_t so = r * 256 + ((c ^ (r & 7)) << 4);
      pd_cp16(ksm + so, kb + j * 8, nb);
      pd_cp16(vsm + so, vb + j * 8, nb);
    }
  };

  float m = NEG_INF, lsum = 0.f;
  float acc[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) { acc[nt][0] = 0.f; acc[nt][1] = 0.f; acc[nt][2] = 0.f; acc[nt][3] = 0.f; }

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < my_n) load_unit(s, s);
    pd_commit();
  }

  const int mi = lane >> 3, ri = lane & 7;
  const float sc2 = p.scale_log2;

  for (int i = 0; i < my_n; ++i) {
    const int nx = i + STAGES - 1;
    if (nx < my_n) load_unit(nx, nx % STAGES);
    pd_commit();
    pd_wait<STAGES - 1>();
    __syncwarp();
    const uint32_t ksa = wbase + (i % STAGES) * 8192, vsa = ksa + 4096;

    float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int r = (mi >> 1) * 8 + ri;
      const int c = ks * 2 + (mi & 1);
      uint32_t b0, b1, b2, b3;
      pd_ldsm4(ksa + r * 256 + ((c ^ (r & 7)) << 4), b0, b1, b2, b3);
      PDT<T>::mma(s0, qa[ks][0], qa[ks][1], b0, b1);
      PDT<T>::mma(s1, qa[ks][0], qa[ks][1], b2, b3);
    }

    const int t0 = ((cs + warp + 4 * i) << 4) + tig * 2;
    float x0 = (t0 < L) ? s0[0] * sc2 : NEG_INF;
    float x1 = (t0 + 1 < L) ? s0[1] * sc2 : NEG_INF;
    float x2 = (t0 + 8 < L) ? s1[0] * sc2 : NEG_INF;
    float x3 = (t0 + 9 < L) ? s1[1] * sc2 : NEG_INF;
    float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    const float mn = fmaxf(m, mx);
    const float alpha = pd_ex2(m - mn);
    const float p0 = pd_ex2(x0 - mn), p1 = pd_ex2(x1 - mn), p2 = pd_ex2(x2 - mn), p3 = pd_ex2(x3 - mn);
    lsum = lsum * alpha + ((p0 + p1) + (p2 + p3));
    m = mn;
    const uint32_t pa0 = PDT<T>::pack(p0, p1);
    const uint32_t pa2 = PDT<T>::pack(p2, p3);
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) { acc[nt][0] *= alpha; acc[nt][1] *= alpha; }

#pragma unroll
    for (int np = 0; np < 8; ++np) {
      const int r = (mi & 1) * 8 + ri;
      const int c = np * 2 + (mi >> 1);
      uint32_t b0, b1, b2, b3;
      pd_ldsm4t(vsa + r * 256 + ((c ^ (r & 7)) << 4), b0, b1, b2, b3);
      PDT<T>::mma(acc[2 * np], pa0, pa2, b0, b1);
      PDT<T>::mma(acc[2 * np + 1], pa0, pa2, b2, b3);
    }
    __syncwarp();
  }

  lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
  lsum += __shfl_xor_sync(0xffffffffu, lsum, 2);
  pd_wait<0>();
  __syncthreads();

  float* fs = reinterpret_cast<float*>(smem);
  float* sacc = fs;                 // [4][8][136]
  float* sm_m = fs + 4 * 8 * 136;   // [4][8]
  float* sm_l = sm_m + 32;          // [4][8]
  int* sflag = reinterpret_cast<int*>(sm_l + 32);
  if (tig == 0) { sm_m[warp * 8 + gid] = m; sm_l[warp * 8 + gid] = lsum; }
  __syncthreads();
  {
    const float Mg = fmaxf(fmaxf(sm_m[gid], sm_m[8 + gid]), fmaxf(sm_m[16 + gid], sm_m[24 + gid]));
    const float f = pd_ex2(m - Mg);
    float* myrow = sacc + (warp * 8 + gid) * 136;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      *reinterpret_cast<float2*>(myrow + nt * 8 + tig * 2) = make_float2(acc[nt][0] * f, acc[nt][1] * f);
    }
  }
  __syncthreads();

  const int g = tid >> 4, d0 = (tid & 15) * 8;
  float o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = 0.f;
  float M2 = NEG_INF, Lg = 0.f;
#pragma unroll
  for (int w = 0; w < 4; ++w) M2 = fmaxf(M2, sm_m[w * 8 + g]);
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    Lg += sm_l[w * 8 + g] * pd_ex2(sm_m[w * 8 + g] - M2);
    const float4 a = *reinterpret_cast<const float4*>(sacc + (w * 8 + g) * 136 + d0);
    const float4 b = *reinterpret_cast<const float4*>(sacc + (w * 8 + g) * 136 + d0 + 4);
    o[0] += a.x; o[1] += a.y; o[2] += a.z; o[3] += a.w;
    o[4] += b.x; o[5] += b.y; o[6] += b.z; o[7] += b.w;
  }

  if (nsplit == 1) {
    if (g < group) {
      T* op = out + (long long)seq * p.so_n + (long long)(kvh * group + g) * p.so_h + d0;
      pd_store8<T>(op, o, 1.0f / Lg);
    }
    return;
  }

  const long long pbase = ((long long)seq * p.Hkv + kvh) * p.splits;
  const long long pidx = pbase + split;
  {
    float4* pa = reinterpret_cast<float4*>(p.pacc + pidx * 1024 + g * 128 + d0);
    __stcg(pa, make_float4(o[0], o[1], o[2], o[3]));
    __stcg(pa + 1, make_float4(o[4], o[5], o[6], o[7]));
    if ((tid & 15) == 0) {
      __stcg(p.pml + pidx * 16 + g * 2, M2);
      __stcg(p.pml + pidx * 16 + g * 2 + 1, Lg);
    }
  }
  __threadfence();
  __syncthreads();
  const int cidx = seq * p.Hkv + kvh;
  if (tid == 0) {
    const int old = atomicAdd(&p.counters[cidx], 1);
    *sflag = (old == nsplit - 1) ? 1 : 0;
  }
  __syncthreads();
  if (*sflag == 0) return;
  __threadfence();

  float Mt = NEG_INF;
  for (int s = 0; s < nsplit; ++s) Mt = fmaxf(Mt, __ldcg(p.pml + (pbase + s) * 16 + g * 2));
  float Lt = 0.f;
  float r8[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) r8[j] = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float ms = __ldcg(p.pml + (pbase + s) * 16 + g * 2);
    const float ls = __ldcg(p.pml + (pbase + s) * 16 + g * 2 + 1);
    const float w = pd_ex2(ms - Mt);
    Lt += ls * w;
    const float4* pa = reinterpret_cast<const float4*>(p.pacc + (pbase + s) * 1024 + g * 128 + d0);
    const float4 a = __ldcg(pa), b = __ldcg(pa + 1);
    r8[0] += a.x * w; r8[1] += a.y * w; r8[2] += a.z * w; r8[3] += a.w * w;
    r8[4] += b.x * w; r8[5] += b.y * w; r8[6] += b.z * w; r8[7] += b.w * w;
  }
  if (g < group) {
    T* op = out + (long long)seq * p.so_n + (long long)(kvh * group + g) * p.so_h + d0;
    pd_store8<T>(op, r8, 1.0f / Lt);
  }
  if (tid == 0) p.counters[cidx] = 0;
}

template <typename T, int STAGES>
static void pd_launch(const PDParams& prm, int N, cudaStream_t stream) {
  const int smem = 4 * STAGES * 8192;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(pd_mma_kernel<T, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr_set = true;
  }
  dim3 grid(prm.splits, prm.Hkv, N);
  pd_mma_kernel<T, STAGES><<<grid, 128, smem, stream>>>(prm);
}

torch::Tensor pd_forward(torch::Tensor q, torch::Tensor kp, torch::Tensor vp, torch::Tensor bt,
                         torch::Tensor sl, torch::Tensor counters, double scale, int64_t chunk, int64_t stages) {
  const int N = (int)q.size(0), Hq = (int)q.size(1);
  const int Hkv = (int)kp.size(1), bs = (int)kp.size(2);
  const int group = Hq / Hkv;
  auto out = torch::empty_like(q);
  if (N == 0) return out;
  int log2bs = 0;
  while ((1 << log2bs) < bs) ++log2bs;
  const long long units_max = ((long long)bt.size(1) * bs) / 16;
  int splits = (int)((units_max + chunk - 1) / chunk);
  if (splits < 1) splits = 1;
  auto fopt = q.options().dtype(torch::kFloat32);
  torch::Tensor pacc, pml;
  float* pacc_p = nullptr;
  float* pml_p = nullptr;
  if (splits > 1) {
    pacc = torch::empty({(int64_t)N * Hkv * splits * 1024}, fopt);
    pml = torch::empty({(int64_t)N * Hkv * splits * 16}, fopt);
    pacc_p = pacc.data_ptr<float>();
    pml_p = pml.data_ptr<float>();
  }
  PDParams prm;
  prm.q = q.data_ptr(); prm.kp = kp.data_ptr(); prm.vp = vp.data_ptr();
  prm.bt = bt.data_ptr<int>(); prm.sl = sl.data_ptr<int>();
  prm.out = out.data_ptr(); prm.pacc = pacc_p; prm.pml = pml_p; prm.counters = counters.data_ptr<int>();
  prm.bt_stride = bt.stride(0);
  prm.sk_b = kp.stride(0); prm.sk_h = kp.stride(1);
  prm.sv_b = vp.stride(0); prm.sv_h = vp.stride(1);
  prm.sq_n = q.stride(0); prm.sq_h = q.stride(1);
  prm.so_n = out.stride(0); prm.so_h = out.stride(1);
  prm.Hkv = Hkv; prm.group = group; prm.chunk = (int)chunk; prm.splits = splits; prm.log2bs = log2bs;
  prm.scale_log2 = (float)(scale * 1.4426950408889634);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kp.scalar_type() == at::kBFloat16) {
    if (stages >= 3) pd_launch<__nv_bfloat16, 3>(prm, N, stream);
    else pd_launch<__nv_bfloat16, 2>(prm, N, stream);
  } else {
    if (stages >= 3) pd_launch<__half, 3>(prm, N, stream);
    else pd_launch<__half, 2>(prm, N, stream);
  }
  return out;
}
"""

_PD_CPP_SRC = ("torch::Tensor pd_forward(torch::Tensor q, torch::Tensor kp, torch::Tensor vp, "
               "torch::Tensor bt, torch::Tensor sl, torch::Tensor counters, double scale, "
               "int64_t chunk, int64_t stages);")

try:
    from torch.utils.cpp_extension import load_inline as _load_inline
    _pd_ext = _load_inline(
        name="pd_mma_decode_74f2_v1",
        cpp_sources=_PD_CPP_SRC,
        cuda_sources=_PD_CUDA_SRC,
        functions=["pd_forward"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
except Exception:
    _pd_ext = None

_PD_MAX_PARTIALS = 8192
_PD_COUNTERS = {}


def _pd_counters(device, need):
    key = (device.type, device.index)
    buf = _PD_COUNTERS.get(key)
    if buf is None or buf.numel() < need:
        buf = torch.zeros(max(1 << 16, 2 * need), dtype=torch.int32, device=device)
        _PD_COUNTERS[key] = buf
    return buf


def _pd_cuda_ok(q, k_pool, v_pool, block_tables, seq_lens):
    if _pd_ext is None or not q.is_cuda:
        return False
    N, Hq, D = q.shape
    Hkv = k_pool.shape[1]
    bs = k_pool.shape[2]
    if D != 128 or Hkv <= 0 or Hq % Hkv != 0:
        return False
    group = Hq // Hkv
    if group < 1 or group > 8:
        return False
    if k_pool.dtype not in (torch.bfloat16, torch.float16):
        return False
    if q.dtype != k_pool.dtype or v_pool.dtype != k_pool.dtype:
        return False
    if bs < 16 or (bs & (bs - 1)) != 0:
        return False
    if v_pool.shape[1] != Hkv or v_pool.shape[2] != bs or v_pool.shape[3] != D:
        return False
    for t in (k_pool, v_pool):
        if t.stride(3) != 1 or t.stride(2) != D:
            return False
        if t.stride(0) % 8 != 0 or t.stride(1) % 8 != 0 or t.data_ptr() % 16 != 0:
            return False
    if not q.is_contiguous() or q.data_ptr() % 16 != 0:
        return False
    if block_tables.dim() != 2 or seq_lens.dim() != 1:
        return False
    return True


def _pd_cuda(q, k_pool, v_pool, block_tables, seq_lens, scale):
    N, Hq, D = q.shape
    Hkv = k_pool.shape[1]
    bs = k_pool.shape[2]
    bt = block_tables
    if bt.dtype != torch.int32 or bt.stride(-1) != 1:
        bt = bt.to(torch.int32).contiguous()
    sl = seq_lens
    if sl.dtype != torch.int32 or not sl.is_contiguous():
        sl = sl.to(torch.int32).contiguous()
    units = max(1, (bt.shape[1] * bs) // 16)
    chunk = max(1, int(CONFIG.get("chunk_units", 16)))
    while N * Hkv * ((units + chunk - 1) // chunk) > _PD_MAX_PARTIALS and chunk < units:
        chunk *= 2
    cnt = _pd_counters(q.device, N * Hkv)
    return _pd_ext.pd_forward(q, k_pool, v_pool, bt, sl, cnt, float(scale), int(chunk),
                              int(CONFIG.get("stages", 2)))


@triton.jit
def _gqa_decode_splitk_kernel(
    Q, Kp, Vp, BT, SL, PM, PL, PA, Out,
    sq_n, sq_h,
    sk_b, sk_h, sk_s,
    pm_n, pm_h,
    pa_n, pa_h, pa_s,
    so_n, so_h,
    bt_n,
    scale,
    GROUP: tl.constexpr, G_PAD: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):
    # One program per (seq, kv_head, split): all GROUP query heads share each K/V load.
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    sp = tl.program_id(2)

    g = tl.arange(0, G_PAD)
    gm = g < GROUP
    heads = kvh * GROUP + g
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)

    L = tl.load(SL + seq)
    n_tiles = (L + BLOCK_N - 1) // BLOCK_N
    per = (n_tiles + SPLITS - 1) // SPLITS
    t0 = sp * per
    t1 = tl.minimum(t0 + per, n_tiles)

    q = tl.load(Q + seq * sq_n + heads[:, None] * sq_h + d[None, :],
                mask=gm[:, None], other=0.0).to(Kp.dtype.element_ty)   # [G_PAD, D]

    m_i = tl.zeros([G_PAD], dtype=tl.float32) - 1e30
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, D], dtype=tl.float32)

    for t in range(t0, t1):
        offs = t * BLOCK_N + n
        valid = offs < L
        blk = tl.load(BT + seq * bt_n + offs // BLOCK_SIZE, mask=valid, other=0)
        base = blk * sk_b + kvh * sk_h + (offs % BLOCK_SIZE) * sk_s
        k = tl.load(Kp + base[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale                          # [G_PAD, BLOCK_N]
        s = tl.where(valid[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(Vp + base[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if SPLITS == 1:
        o = acc / l_i[:, None]
        tl.store(Out + seq * so_n + heads[:, None] * so_h + d[None, :],
                 o.to(Out.dtype.element_ty), mask=gm[:, None])
    else:
        tl.store(PM + seq * pm_n + heads * pm_h + sp, m_i, mask=gm)
        tl.store(PL + seq * pm_n + heads * pm_h + sp, l_i, mask=gm)
        tl.store(PA + seq * pa_n + heads[:, None] * pa_h + sp * pa_s + d[None, :], acc,
                 mask=gm[:, None])


@triton.jit
def _splitk_combine_vec_kernel(
    PM, PL, PA, Out,
    pm_n, pm_h,
    pa_n, pa_h, pa_s,
    so_n, so_h,
    D: tl.constexpr, SPLITS: tl.constexpr,
):
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    d = tl.arange(0, D)
    sp = tl.arange(0, SPLITS)
    ms = tl.load(PM + seq * pm_n + qh * pm_h + sp)
    ls = tl.load(PL + seq * pm_n + qh * pm_h + sp)
    mmax = tl.max(ms, axis=0)
    w = tl.exp(ms - mmax)
    l_tot = tl.sum(ls * w, axis=0)
    a = tl.load(PA + seq * pa_n + qh * pa_h + sp[:, None] * pa_s + d[None, :])
    acc = tl.sum(a * w[:, None], axis=0)
    out = acc / l_tot
    tl.store(Out + seq * so_n + qh * so_h + d, out.to(Out.dtype.element_ty))


def _gqa_splits(n_seqs: int, n_kv: int) -> int:
    base = max(1, n_seqs * n_kv)
    tgt = CONFIG["target_programs"]
    s = (tgt + base - 1) // base
    s = 1 << (s - 1).bit_length() if s > 1 else 1
    return max(CONFIG["min_splits"], min(CONFIG["max_splits"], s))


def paged_decode_attention(
    q: torch.Tensor,          # [N, Hq, D] — one decode token per sequence
    k_pool: torch.Tensor,     # [num_blocks, num_kv_heads, block_size, D]
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,  # [N, MAX_BLOCKS] int32 — physical block id per logical block
    seq_lens: torch.Tensor,      # [N] int32 — real KV length per sequence
    scale: float = 1.0,
) -> torch.Tensor:
    """Decode attention reading K/V via block tables. Returns [N, Hq, D] (q's dtype)."""
    N, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    q = q.contiguous()
    group = Hq // num_kv_heads
    bs = k_pool.shape[2]

    # Hand-written CUDA path: cp.async paged gather + ldmatrix + mma.sync, fused last-CTA combine.
    if _pd_cuda_ok(q, k_pool, v_pool, block_tables, seq_lens):
        return _pd_cuda(q, k_pool, v_pool, block_tables, seq_lens, scale)

    # GQA-packed split-K: one program per (sequence, KV head, split) serves all GROUP query heads from each K/V load.
    # Found by the layer-1 loop (BlameGraph kernels/evolve, 2026-10-14) and validated for head dim <= 128 only; wider
    # heads keep the tuned launch table below.
    if (D & (D - 1)) == 0 and 16 <= D <= 128 and (bs & (bs - 1)) == 0 and k_pool.dtype in (
            torch.float16, torch.bfloat16):
        g_pad = max(16, 1 << (group - 1).bit_length())
        block_n = 64 if D <= 128 else (32 if D <= 256 else 16)
        block_n = max(block_n, bs) if bs <= 64 else block_n
        splits = _gqa_splits(N, num_kv_heads)
        out = torch.empty(N, Hq, D, dtype=q.dtype, device=q.device)
        if splits == 1:
            _gqa_decode_splitk_kernel[(N, num_kv_heads, 1)](
                q, k_pool, v_pool, block_tables, seq_lens, out, out, out, out,
                q.stride(0), q.stride(1),
                k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
                0, 0,
                0, 0, 0,
                out.stride(0), out.stride(1),
                block_tables.stride(0),
                scale,
                GROUP=group, G_PAD=g_pad, BLOCK_SIZE=bs, BLOCK_N=block_n, D=D, SPLITS=1,
                num_warps=CONFIG["num_warps"],
            )
            return out
        pm = torch.empty(N, Hq, splits, dtype=torch.float32, device=q.device)
        pl = torch.empty(N, Hq, splits, dtype=torch.float32, device=q.device)
        pa = torch.empty(N, Hq, splits, D, dtype=torch.float32, device=q.device)
        _gqa_decode_splitk_kernel[(N, num_kv_heads, splits)](
            q, k_pool, v_pool, block_tables, seq_lens, pm, pl, pa, out,
            q.stride(0), q.stride(1),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            pm.stride(0), pm.stride(1),
            pa.stride(0), pa.stride(1), pa.stride(2),
            out.stride(0), out.stride(1),
            block_tables.stride(0),
            scale,
            GROUP=group, G_PAD=g_pad, BLOCK_SIZE=bs, BLOCK_N=block_n, D=D, SPLITS=splits,
            num_warps=CONFIG["num_warps"],
        )
        _splitk_combine_vec_kernel[(N, Hq)](
            pm, pl, pa, out,
            pm.stride(0), pm.stride(1),
            pa.stride(0), pa.stride(1), pa.stride(2),
            out.stride(0), out.stride(1),
            D=D, SPLITS=splits, num_warps=4,
        )
        return out

    out = torch.empty(N, Hq, D, dtype=torch.float32, device=q.device)
    cfg = launch_table.lookup("paged_decode", N, D, k_pool.shape[2])
    splits = cfg["splits"] if cfg else decode_splits(N, Hq)
    kw = launch_table.compile_kwargs(cfg)
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
            scale,
            GROUP=Hq // num_kv_heads, BLOCK_SIZE=k_pool.shape[2], D=D, SPLITS=splits, **kw,
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
        scale,
        GROUP=Hq // num_kv_heads,
        BLOCK_SIZE=k_pool.shape[2],
        D=D,
        **kw,
    )
    return out.to(q.dtype)


@triton.jit
def _paged_prefill_kernel(
    Q, Kp, Vp, BT, PL, SL, Out,
    sq_n, sq_s, sq_h,      # Q strides: [N, S_q, Hq, D] (last dim contiguous)
    sk_b, sk_h, sk_s,      # pool strides
    so_n, so_s, so_h,      # Out strides: [N, S_q, Hq, D]
    bt_n,
    scale,
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

    n_blocks = (q_pos + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE

    m = -float("inf")
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for b in range(0, n_blocks):
        blk = tl.load(BT + seq * bt_n + b)
        offs = b * BLOCK_SIZE + slots
        valid = offs <= q_pos                      # causal (≤ q_pos)
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


@triton.jit
def _paged_prefill_tiled_kernel(
    Q, Kp, Vp, BT, PL, SL, Out,
    sq_n, sq_s, sq_h,
    sk_b, sk_h, sk_s,
    so_n, so_s, so_h,
    bt_n,
    scale,
    GROUP: tl.constexpr, BLOCK_SIZE: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """FlashAttention-style tiled paged prefill: one program per BLOCK_M query tokens.

    The untiled kernel runs one program per QUERY TOKEN, so each K/V block is re-read from HBM
    once per query — O(S^2) memory traffic for arithmetic that is only 1.5% of the prefill's
    FLOPs. Measured consequence: 63% of prefill time at 0.34% of A100 peak.

    Here a program loads each K/V block ONCE and scores it against BLOCK_M queries at a time,
    so HBM traffic falls by a factor of BLOCK_M. Softmax stays numerically identical: the same
    online (m, l) rescaling, just carried per-row across a tile instead of per-program.
    """
    seq = tl.program_id(0)
    qh = tl.program_id(1)
    tile = tl.program_id(2)
    kvh = qh // GROUP

    suffix_len = tl.load(SL + seq)
    prefix_len = tl.load(PL + seq)

    offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)          # query index within the suffix
    d = tl.arange(0, D)
    slots = tl.arange(0, BLOCK_SIZE)

    # A tile is entirely padding iff it starts past the real suffix — a scalar test, so no
    # tensor-derived control flow.
    tile_start = tile * BLOCK_M
    if tile_start >= suffix_len:
        return

    real = offs_m < suffix_len                                # right-padded tail
    q_pos = prefix_len + offs_m                               # absolute position per query
    q = tl.load(Q + seq * sq_n + offs_m[:, None] * sq_s + qh * sq_h + d[None, :],
                mask=real[:, None], other=0.0).to(tl.float32)  # [BLOCK_M, D]

    # Causal bound for the whole tile: no query here attends past the tile's last real position.
    # tl.minimum/maximum are elementwise; tl.min/max are reductions and are wrong on scalars.
    last_in_tile = tl.minimum(tile_start + BLOCK_M - 1, suffix_len - 1)
    last_pos = prefix_len + last_in_tile
    n_blocks = (last_pos + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Finite sentinel, not -inf. Tiling shares one block range across BLOCK_M queries, so an
    # earlier query in the tile WILL see a fully-masked (entirely future) block; with -inf that
    # gives exp(-inf - -inf) = NaN and poisons the whole tile. The untiled kernel never hits this
    # because it computes the block range per query.
    NEG: tl.constexpr = -1e30
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + NEG
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for b in range(0, n_blocks):
        blk = tl.load(BT + seq * bt_n + b)
        offs_n = b * BLOCK_SIZE + slots                       # absolute key positions

        kptr = Kp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
        k = tl.load(kptr).to(tl.float32)                      # [BLOCK_SIZE, D] — loaded ONCE
        s = tl.dot(q, tl.trans(k)) * scale                    # [BLOCK_M, BLOCK_SIZE]

        # causal (key <= query), per (query, key)
        valid = (offs_n[None, :] <= q_pos[:, None]) & real[:, None]
        s = tl.where(valid, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        vptr = Vp + blk * sk_b + kvh * sk_h + slots[:, None] * sk_s + d[None, :]
        v = tl.load(vptr).to(tl.float32)                      # [BLOCK_SIZE, D] — loaded ONCE
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(Out + seq * so_n + offs_m[:, None] * so_s + qh * so_h + d[None, :],
             out, mask=real[:, None])


def paged_prefill_attention(
    q: torch.Tensor,             # [N, S_q, Hq, D] — suffix query tokens (right-padded), per sequence
    k_pool: torch.Tensor,        # [num_blocks, num_kv_heads, block_size, D]
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,  # [N, MAX_BLOCKS] int32
    prefix_lens: torch.Tensor,   # [N] int32 — cached prefix length per sequence
    suffix_lens: torch.Tensor,   # [N] int32 — real suffix length per sequence
    scale: float = 1.0,
) -> torch.Tensor:
    """Multi-query paged prefill attention (no gather). Suffix K/V must already be in the blocks.
    Returns [N, S_q, Hq, D] (q's dtype); padding query rows are left undefined (caller slices)."""
    N, Sq, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    out = torch.empty(N, Sq, Hq, D, dtype=torch.float32, device=q.device)
    q = q.contiguous()

    block_m, warps = prefill_launch(D, Sq)
    if block_m:
        # The table tunes within the tiled variant; it never switches tiling on or off.
        cfg = launch_table.lookup("paged_prefill_tiled", Sq, D, k_pool.shape[2])
        if cfg:
            block_m = cfg["block_m"]
        kw = {"num_warps": warps, **launch_table.compile_kwargs(cfg)}
        grid = (N, Hq, triton.cdiv(Sq, block_m))
        _paged_prefill_tiled_kernel[grid](
            q, k_pool, v_pool, block_tables, prefix_lens, suffix_lens, out,
            q.stride(0), q.stride(1), q.stride(2),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            block_tables.stride(0),
            scale,
            GROUP=Hq // num_kv_heads, BLOCK_SIZE=k_pool.shape[2], D=D, BLOCK_M=block_m, **kw,
        )
        return out.to(q.dtype)

    kw = launch_table.compile_kwargs(launch_table.lookup("paged_prefill", Sq, D, k_pool.shape[2]))
    _paged_prefill_kernel[(N, Hq, Sq)](
        q, k_pool, v_pool, block_tables, prefix_lens, suffix_lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0),
        scale,
        GROUP=Hq // num_kv_heads,
        BLOCK_SIZE=k_pool.shape[2],
        D=D,
        **kw,
    )
    return out.to(q.dtype)

_CONFIG_BY_SIZE = [(11, {'target_programs': 256, 'max_splits': 16, 'min_splits': 1, 'num_warps': 4, 'chunk_units': 16, 'stages': 3}), (53, {'target_programs': 256, 'max_splits': 16, 'min_splits': 1, 'num_warps': 4, 'chunk_units': 32, 'stages': 3}), (63, {'target_programs': 256, 'max_splits': 16, 'min_splits': 1, 'num_warps': 4, 'chunk_units': 32, 'stages': 3}), (None, {'target_programs': 256, 'max_splits': 16, 'min_splits': 1, 'num_warps': 4, 'chunk_units': 32, 'stages': 3})]
_dispatch_impl = paged_decode_attention


def _dispatched(*args, **kwargs):
    global CONFIG
    n = args[0].shape[0]
    CONFIG = next(c for ub, c in _CONFIG_BY_SIZE if ub is None or n <= ub)
    return _dispatch_impl(*args, **kwargs)

paged_decode_attention = _dispatched
