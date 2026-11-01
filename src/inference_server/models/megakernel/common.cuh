// Shared by the decode-step megakernel (decode_step.cu) and the instruction interpreter (interp.cu): the build
// flags, the model's compile-time shape, the per-layer pointer table, the scratch layout and the device helpers.
#pragma once
#ifndef MEGA_KV8
#define MEGA_KV8 0
#endif
#ifndef MEGA_TMA
#define MEGA_TMA 0
#endif
// Blocks per SM (MEGAKERNEL_BPS): 2 doubles the warps hiding memory latency (ncu, 2026-10-20: 8 warps an SM, no
// eligible warp 89% of cycles, DRAM at 19%) at half the registers each; only where the phases' shared memory fits
// twice, so never with the weight ring.
#ifndef MEGA_BPS
#define MEGA_BPS 1
#endif
#if MEGA_TMA && MEGA_BPS > 1
#error "the weight ring takes the whole shared memory: one block per SM"
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

constexpr int H = 2048, NQ = 32, NKV = 4, D = 128, QKV = NQ * D + 2 * NKV * D, I = 768, E = 128, K = 8;
// Attention positions per item (MEGA_CHUNK): items = rows x kv groups x ceil(context / CHUNK), so a smaller chunk spreads
// a context over more blocks (at 256, a 2k context ran on 32 of 132 blocks: 20 us a layer, 2026-10-24).
#ifndef MEGA_CHUNK
#define MEGA_CHUNK 256
#endif
// Threads per block (MEGA_THREADS): the GEMV phases' stage consumption is issue-bound (8 warps dequantizing and
// FMA-ing 16 fp8 rows of 2048 per 32 KB stage took ~3 us, 2026-10-24), so 16 warps double it.
#ifndef MEGA_THREADS
#define MEGA_THREADS 256
#endif
constexpr int BS = 16, BMAX = 16, THREADS = MEGA_THREADS, WARPS = THREADS / 32, CHUNK = MEGA_CHUNK, MAXSPLIT = 32768 / CHUNK;
constexpr int HDR = 256;                                             // smem header: red[WARPS] floats, then the mbarriers at 128
static_assert(THREADS == 256 || THREADS == 512, "8 or 16 warps");
static_assert(CHUNK >= 32 && CHUNK <= 256 && THREADS % CHUNK == 0 && 32768 % CHUNK == 0, "chunk: 32..256, a divisor of 32768");
constexpr int GQA = NQ / NKV;                                        // q heads per kv head
static_assert(WARPS >= GQA, "the attention phase preps one q head per warp (the first GQA warps)");
constexpr float EPS = 1e-6f;

using bf16 = __nv_bfloat16;

// One decoder layer's tensors; 13 pointers, built on the host as an int64 [layers, 13] tensor in this order.
struct Layer {
    const bf16* in_norm; const bf16* wqkv; const bf16* q_norm; const bf16* k_norm; const bf16* wo; const bf16* post_norm;
    const bf16* wr;
    const void* gate_up; const float* gu_scale;      // [E, 2I, H] fp8 e4m3 + [E, 2I] fp32 scale, or bf16 and null
    const void* down; const float* down_scale;       // [E, H, I] likewise
    void* pool_k; void* pool_v;                      // [blocks, NKV, BS, D], bf16 or fp8 e4m3 (Args::kv8)
};
static_assert(sizeof(Layer) == 13 * 8, "Layer is 13 pointers");

struct Scratch {
    bf16* h; bf16* qkv; float* part_o; float* part_m; float* part_l; bf16* attn; bf16* o; bf16* h1; bf16* n2;
    int* ids; float* w; bf16* act; float* out2; float* rlogits;
};

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rbf(float x) { return bf2f(f2bf(x)); }      // round through bf16

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < WARPS; ++i) s += red[i];
    return s;
}

// RMSNorm of one row of n bf16 values (whole block), torch rounding: y = bf16(w * bf16(x * rsqrt(mean(x^2)+eps))).
// x and y may alias.
__device__ void rmsnorm_row(const bf16* x, const bf16* w, bf16* y, int n, float* red) {
    float ss = 0.f;
    for (int k = threadIdx.x; k < n; k += THREADS) { float v = bf2f(x[k]); ss += v * v; }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / n + EPS);
    for (int k = threadIdx.x; k < n; k += THREADS) y[k] = f2bf(bf2f(w[k]) * rbf(bf2f(x[k]) * r));
    __syncthreads();
}

template <int EPL> __device__ __forceinline__ void bf16_to_f32(const void* v, float* f) {
    const bf16* p = reinterpret_cast<const bf16*>(v);
#pragma unroll
    for (int i = 0; i < EPL; ++i) f[i] = bf2f(p[i]);
}
template <int EPL> __device__ __forceinline__ void fp8_to_f32(const void* v, float* f) {      // EPL e4m3 -> floats
    const unsigned short* p = reinterpret_cast<const unsigned short*>(v);
#pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
        const float2 t = __half22float2(__half2(r));
        f[2 * i] = t.x; f[2 * i + 1] = t.y;
    }
}

// ---- bf16 GEMV over R weight rows (length N, N % 256 == 0) against BT shared-memory rows, one warp. The loads are
// a separate step so a phase's first rows can be in flight across the barrier before it. out[r][b] warp-reduced.
template <int N, int R>
__device__ __forceinline__ void load_rows(const bf16* const* __restrict__ rows, uint4 (*wv)[N / 256]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < N / 256; ++i) wv[r][i] = *reinterpret_cast<const uint4*>(rows[r] + i * 256 + lane * 8);
}
template <int N, int R, int BT>
__device__ __forceinline__ void dot_rows(const uint4 (*wv)[N / 256], const bf16* __restrict__ xs, int xs_stride, float (*out)[BT]) {
    const int lane = threadIdx.x & 31;
    constexpr int NV = N / 256;
    float acc[R][BT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < BT; ++b) acc[r][b] = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int k0 = i * 256 + lane * 8;
        float xf[BT][8];
#pragma unroll
        for (int b = 0; b < BT; ++b) { const uint4 xv = *reinterpret_cast<const uint4*>(xs + b * xs_stride + k0); bf16_to_f32<8>(&xv, xf[b]); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float wf[8];
            bf16_to_f32<8>(&wv[r][i], wf);
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int b = 0; b < BT; ++b) acc[r][b] += wf[j] * xf[b][j];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int b = 0; b < BT; ++b) out[r][b] = warp_sum(acc[r][b]);
}

// ---- the expert GEMVs: R weight rows (fp8 e4m3 bytes or bf16) of length N dotted with one bf16 x row, one warp.
template <int VB> struct VecT;
template <> struct VecT<16> { using type = uint4; };
template <> struct VecT<8> { using type = uint2; };

// VB = bytes per lane-load (16 or 8). All R rows' loads are issued first. out[r] warp-reduced (valid on every lane).
template <int N, int R, bool FP8, int VB>
__device__ __forceinline__ void warp_dot_rows(const void* const* __restrict__ rows, const bf16* __restrict__ x, float* out) {
    constexpr int EPL = FP8 ? VB : VB / 2;                 // elements per lane-load
    constexpr int CH = 32 * EPL;                           // elements per warp chunk
    static_assert(N % CH == 0, "row length is a whole number of warp chunks");
    constexpr int NV = N / CH;
    using V = typename VecT<VB>::type;
    const int lane = threadIdx.x & 31;
    V wv[R][NV];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < NV; ++i)
            wv[r][i] = *reinterpret_cast<const V*>(reinterpret_cast<const char*>(rows[r]) + ((size_t)i * CH + lane * EPL) * (FP8 ? 1 : 2));
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        float xf[EPL];
        const bf16* xp = x + i * CH + lane * EPL;
#pragma unroll
        for (int j = 0; j < EPL / 8; ++j) { const uint4 xv = *reinterpret_cast<const uint4*>(xp + j * 8); bf16_to_f32<8>(&xv, xf + j * 8); }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float wf[EPL];
            if constexpr (FP8) fp8_to_f32<EPL>(&wv[r][i], wf); else bf16_to_f32<EPL>(&wv[r][i], wf);
#pragma unroll
            for (int j = 0; j < EPL; ++j) acc[r] += wf[j] * xf[j];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) out[r] = warp_sum(acc[r]);
}

// The down GEMV's rows (length I): RD in flight per warp, 8 elements per lane-load either way (uint2 of fp8, uint4 of bf16).
template <bool FP8> struct Down {
    static constexpr int RD = FP8 ? 8 : 4, VB = FP8 ? 8 : 16, EPL = 8, CH = 256, NV = I / 256;
    using V = typename VecT<VB>::type;
};
template <bool FP8>
__device__ __forceinline__ void load_down(const void* down, const int* ids, int w0, int limit, typename Down<FP8>::V (*wv)[Down<FP8>::NV]) {
    using DN = Down<FP8>;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < DN::RD; ++r) {
        const int w = min(w0 + r, limit - 1), pp = w / H, nn = w % H;
        const char* row = reinterpret_cast<const char*>(down) + ((size_t)ids[pp] * H + nn) * I * (FP8 ? 1 : 2);
#pragma unroll
        for (int i = 0; i < DN::NV; ++i)
            wv[r][i] = *reinterpret_cast<const typename DN::V*>(row + ((size_t)i * DN::CH + lane * DN::EPL) * (FP8 ? 1 : 2));
    }
}

// ---- TMA weight streaming (MEGAKERNEL_TMA=1, batch <= 2). A warp holding a few 16-byte loads per lane in flight
// streamed 2.1-2.8 TB/s in every GEMV phase (H200, 2026-10-19); here each block streams its own contiguous rows into a
// ring of NST shared-memory stages with cp.async.bulk (completion counted on an mbarrier per stage, ~96 KB in flight
// per SM) and its warps compute from shared memory. The ring is the top 96 KB of the 128 KB the kernel reserves,
// which no other phase touches at batch <= 2.
// MEGA_PRIME: a GEMV phase's first ring stages are issued before the grid barrier it waits at (its rows do not depend
// on the barrier), so the phase starts computing as soon as the barrier opens: o_proj's after attention (when the
// attention smem leaves the ring free, chunk <= 64), down's after gate_up, the next layer's qkv after down.
#ifndef MEGA_PRIME
#define MEGA_PRIME 0
#endif
#ifndef MEGA_NST                                                    // stages in flight and their size: NST x STAGE = 96 KB
#define MEGA_NST 3
#endif
// MEGA_RING_KB: the ring's size (96 KB default; up to 192 at batch <= 2, the shared memory those phases leave free).
// MEGA_NOMATH (timing only, wrong output): the ring phases' dot products read one element instead of the row, which
// separates the weight stream's time from the math's.
#ifndef MEGA_RING_KB
#define MEGA_RING_KB 96
#endif
#ifndef MEGA_NOMATH
#define MEGA_NOMATH 0
#endif
constexpr int NST = MEGA_NST, STAGE = MEGA_RING_KB * 1024 / NST, RING_OFF = 32768;
// Attention layout: SPLIT threads per position, each a QW-wide slice of the head dimension (q slices padded by 4 floats,
// V rows by 8 elements, so the slices a quarter-warp reads start in different banks); PVS threads per (head, 8-wide d
// group) item in PV. One thread per position spent 9-11 us of a layer's ~21 in the scores (2026-10-25, ctx 300-2000).
constexpr int SPLIT = THREADS / CHUNK, QW = D / SPLIT, QSTR = SPLIT * (QW + 4), VSTR = D + 8, PVS = THREADS / (GQA * D / 8);
static_assert(SPLIT >= 1 && SPLIT <= 8 && QW % 16 == 0 && PVS >= 1 && PVS <= 4 && (PVS & (PVS - 1)) == 0,
              "positions split into 16-element slices; 1-4 threads per PV item");
__device__ __forceinline__ int qi(int j, int d) { return j * QSTR + (d / QW) * (QW + 4) + d % QW; }   // q_s index
constexpr size_t ATTN_SMEM = HDR + GQA * QSTR * 4 + CHUNK * VSTR * 2 + GQA * CHUNK * 4 + GQA * WARPS * 4;
constexpr bool PRIME_P4 = MEGA_PRIME && ATTN_SMEM <= RING_OFF;      // the attention phase leaves the ring untouched
static_assert((MEGA_RING_KB * 1024) % NST == 0 && STAGE % 8192 == 0, "the ring is whole 8 KB rows a stage");

__device__ __forceinline__ uint32_t s2u(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* b) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(s2u(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned long long* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(s2u(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, unsigned long long* b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(s2u(dst)), "l"(src), "r"(bytes), "r"(s2u(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, uint32_t parity) {
    asm volatile("{\n\t.reg .pred P;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t@!P bra LAB_WAIT;\n\t}"
                 :: "r"(s2u(b)), "r"(parity) : "memory");
}

// nst stages through the ring: issue(s, dst, bar) (thread 0) arms `bar` with the stage's byte count and starts its
// bulk copies into `dst`; consume(s, src) runs on the whole block once they have landed. `ph` holds each slot's phase
// parity and persists across calls (every thread runs every call, so all agree).
template <typename Issue>
__device__ __forceinline__ void ring_prime(int nst, unsigned long long* bars, unsigned char* ring, Issue issue) {
    if (threadIdx.x == 0) {
        // the ring's bytes were last written by ordinary stores (the attention phase stages V there): order those
        // before the async proxy's copies
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        for (int s = 0; s < min(nst, NST); ++s) issue(s, ring + s * STAGE, bars + s);
    }
}

template <typename Issue, typename Consume>
__device__ __forceinline__ void ring_run(int nst, unsigned long long* bars, unsigned char* ring, uint32_t& ph,
                                         Issue issue, Consume consume, bool primed = false) {
    if (!primed) ring_prime(nst, bars, ring, issue);
    for (int s = 0; s < nst; ++s) {
        const int k = s % NST;
        mbar_wait(bars + k, (ph >> k) & 1u);
        ph ^= 1u << k;
        consume(s, ring + k * STAGE);
        __syncthreads();                                          // the slot is free again
        if (threadIdx.x == 0 && s + NST < nst) issue(s + NST, ring + k * STAGE, bars + k);
    }
}

// One warp: a bf16 row of N elements in shared memory dotted with BT bf16 rows (stride xs_stride) in shared memory.
template <int N, int BT>
__device__ __forceinline__ void smem_dot(const bf16* __restrict__ w, const bf16* __restrict__ xs, int xs_stride, float* out) {
    if constexpr (MEGA_NOMATH) {
#pragma unroll
        for (int b = 0; b < BT; ++b) out[b] = bf2f(w[threadIdx.x & 31]) * bf2f(xs[b * xs_stride]);
        return;
    }
    const int lane = threadIdx.x & 31;
    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; ++b) acc[b] = 0.f;
#pragma unroll 4
    for (int i = 0; i < N / 256; ++i) {
        const int k0 = i * 256 + lane * 8;
        float wf[8];
        const uint4 wv = *reinterpret_cast<const uint4*>(w + k0);
        bf16_to_f32<8>(&wv, wf);
#pragma unroll
        for (int b = 0; b < BT; ++b) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + b * xs_stride + k0);
            float xf[8];
            bf16_to_f32<8>(&xv, xf);
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[b] += wf[j] * xf[j];
        }
    }
#pragma unroll
    for (int b = 0; b < BT; ++b) out[b] = warp_sum(acc[b]);
}

// One warp: an fp8 e4m3 row of N elements in shared memory dotted with one bf16 row (unscaled; the caller scales).
template <int N>
__device__ __forceinline__ float smem_dot_fp8(const unsigned char* __restrict__ w, const bf16* __restrict__ x) {
    const int lane = threadIdx.x & 31;
    if constexpr (MEGA_NOMATH) return (float)w[lane] * bf2f(x[lane]);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < N / 512; ++i) {
        const int k0 = i * 512 + lane * 16;
        const uint4 wv = *reinterpret_cast<const uint4*>(w + k0);
        float wf[16], xf[16];
        fp8_to_f32<16>(&wv, wf);
        const uint4 x0 = *reinterpret_cast<const uint4*>(x + k0), x1 = *reinterpret_cast<const uint4*>(x + k0 + 8);
        bf16_to_f32<8>(&x0, xf);
        bf16_to_f32<8>(&x1, xf + 8);
#pragma unroll
        for (int j = 0; j < 16; ++j) acc += wf[j] * xf[j];
    }
    return warp_sum(acc);
}

// The same over 8-byte chunks (N a multiple of 256).
template <int N>
__device__ __forceinline__ float smem_dot_fp8_8(const unsigned char* __restrict__ w, const bf16* __restrict__ x) {
    const int lane = threadIdx.x & 31;
    if constexpr (MEGA_NOMATH) return (float)w[lane] * bf2f(x[lane]);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < N / 256; ++i) {
        const int k0 = i * 256 + lane * 8;
        const uint2 wv = *reinterpret_cast<const uint2*>(w + k0);
        float wf[8], xf[8];
        fp8_to_f32<8>(&wv, wf);
        const uint4 xv = *reinterpret_cast<const uint4*>(x + k0);
        bf16_to_f32<8>(&xv, xf);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += wf[j] * xf[j];
    }
    return warp_sum(acc);
}

// Stage starts over [lo, hi) in steps of at most `c`, never crossing a multiple of `seg` (a pair's rows end there):
// seg_s[0..n] with seg_s[n] = hi; returns n. One thread.
__device__ __forceinline__ int plan_stages(int lo, int hi, int c, int seg, int* seg_s, int max_n) {
    int n = 0;
    for (int f = lo; f < hi && n < max_n; ++n) {
        seg_s[n] = f;
        f = min(min(f + c, hi), (f / seg + 1) * seg);
    }
    seg_s[n] = hi;
    return n;
}

// The block's slice [lo, hi) of n rows split as evenly as the grid allows.
__device__ __forceinline__ void block_slice(int n, int& lo, int& hi) {
    const int per = (n + gridDim.x - 1) / gridDim.x;
    lo = min(n, blockIdx.x * per);
    hi = min(n, lo + per);
}

// ---- host side
static int num_sms() {
    static int n = 0;
    if (!n) { cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); }
    return n;
}

template <int BT> constexpr size_t smem_bytes() {           // what the BT-row instantiation touches (the ring included when built)
    size_t m = ATTN_SMEM;
    for (size_t v : {(size_t)HDR + 2 * BT * H * 2, (size_t)HDR + BT * H * 2 + BT * K * 8, (size_t)HDR + (size_t)BT * NQ * D * 2,
                     (MEGA_TMA && BT <= 2) ? (size_t)RING_OFF + NST * STAGE : (size_t)0})
        m = v > m ? v : m;
    return m;
}
static size_t smem_for(int B) {
    return B <= 1 ? smem_bytes<1>() : B <= 2 ? smem_bytes<2>() : B <= 4 ? smem_bytes<4>() : B <= 8 ? smem_bytes<8>() : smem_bytes<16>();
}
// The scratch buffers carved from the three tensors the host allocates (sizes below).
inline Scratch carve_scratch(float* f, bf16* s, int* i) {
    Scratch sc;
    sc.part_o = f; f += (size_t)BMAX * NQ * MAXSPLIT * D;
    sc.part_m = f; f += (size_t)BMAX * NQ * MAXSPLIT;
    sc.part_l = f; f += (size_t)BMAX * NQ * MAXSPLIT;
    sc.w = f; f += BMAX * K;
    sc.out2 = f; f += (size_t)BMAX * H;
    sc.rlogits = f;
    sc.h = s; s += (size_t)BMAX * H;
    sc.qkv = s; s += (size_t)BMAX * QKV;
    sc.attn = s; s += (size_t)BMAX * NQ * D;
    sc.o = s; s += (size_t)BMAX * H;
    sc.h1 = s; s += (size_t)BMAX * H;
    sc.n2 = s; s += (size_t)BMAX * H;
    sc.act = s;
    sc.ids = i;
    return sc;
}
size_t scratch_f32_elems() { return (size_t)BMAX * NQ * MAXSPLIT * D + 2 * (size_t)BMAX * NQ * MAXSPLIT + BMAX * K + (size_t)BMAX * H + (size_t)BMAX * E; }
size_t scratch_bf16_elems() { return (size_t)BMAX * (QKV + NQ * D + 4 * H) + (size_t)BMAX * K * I; }
size_t scratch_i32_elems() { return (size_t)BMAX * K + 2 * BMAX; }   // ids, then the argmax keys (64-bit)

