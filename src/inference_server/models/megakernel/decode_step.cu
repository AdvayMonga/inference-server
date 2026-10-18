// One persistent kernel for a whole Qwen3-30B-A3B decode step, batch <= 16 (MEGAKERNEL=1): embedding, the 48
// decoder layers, the final norm and lm_head, in one cooperative launch (one block per SM), phases separated by
// grid-wide barriers. v1 (decode_layer.cu, 2026-10-18) was one launch per layer: 68 us a layer at batch 1 against a
// ~28 us byte floor, plus 47 launch gaps and the torch lm_head. Here, per layer:
//   P1 residual h = (L == 0 ? embed[token] : h1 + out2) in every block, RMSNorm, QKV GEMV      -> qkv [B, 5120]
//   P2a per (row, kv group): q heads normed+roped+scaled (fp32); k normed+roped and v written into the paged pool
//   P2b attention per (row, kv group, 256-position chunk); one chunk -> attn directly, else partial o/m/l
//   P3 (only when some row spans several chunks) combine the chunks                               -> attn [B, 4096]
//   P4 o_proj GEMV                                                                                -> o [B, 2048]
//   P5 h1 = bf16(h + o), n2 = RMSNorm(h1) in every block; router logits over the grid             -> logits [B, 128]
//   P6 every block: softmax + top-8 of the logits (redundant, 1 warp per row); gate_up GEMV per (pair, n) with the
//      expert weights streamed as fp8 e4m3 (one fp32 scale per output channel) or bf16          -> act [B*8, 768]
//   P7 down GEMV per (pair, n), fp32 atomics                                                      -> out2 [B, 2048]
// then PF: RMSNorm(h1 + out2) and the lm_head GEMV                                                -> logits [B, V]
// Numerics follow the torch path: fp32 accumulation, bf16 rounding where it rounds; fp8 experts are weight-only
// (activations stay bf16/fp32), x . (q * scale) accumulated in fp32. Dimensions are compile-time: the engine is
// specialised to this model.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

constexpr int H = 2048, NQ = 32, NKV = 4, D = 128, QKV = NQ * D + 2 * NKV * D, I = 768, E = 128, K = 8;
constexpr int BS = 16, BMAX = 16, THREADS = 256, WARPS = THREADS / 32, CHUNK = 256, MAXSPLIT = 128;
constexpr int GQA = NQ / NKV;                                        // q heads per kv head
constexpr float EPS = 1e-6f;

using bf16 = __nv_bfloat16;

// One decoder layer's tensors; 13 pointers, built on the host as an int64 [layers, 13] tensor in this order.
struct Layer {
    const bf16* in_norm; const bf16* wqkv; const bf16* q_norm; const bf16* k_norm; const bf16* wo; const bf16* post_norm;
    const bf16* wr;
    const void* gate_up; const float* gu_scale;      // [E, 2I, H] fp8 e4m3 + [E, 2I] fp32 scale, or bf16 and null
    const void* down; const float* down_scale;       // [E, H, I] likewise
    bf16* pool_k; bf16* pool_v;                      // [blocks, NKV, BS, D]
};
static_assert(sizeof(Layer) == 13 * 8, "Layer is 13 pointers");

struct Scratch {
    bf16* h; bf16* qkv; float* q; float* part_o; float* part_m; float* part_l; bf16* attn; bf16* o; bf16* h1; bf16* n2;
    int* ids; float* w; bf16* act; float* out2; float* rlogits;
};
struct Args {
    const int64_t* tokens; const bf16* embed; const int* pos; const int* bt; int bt_cols;   // bt [layers, B, cols]
    const Layer* layers; int n_layers; const bf16* final_norm; const bf16* lm_head; int V; bf16* logits;
    int B; float scale; const float* inv_freq; Scratch sc; unsigned long long* times;      // times [16] ns, debug
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

// R bf16 weight rows (length N, N % 256 == 0) dotted with the BT shared-memory rows (B of them real), one warp: every
// load of the R rows is issued before any arithmetic. out[r][b] valid on lane 0 (warp-reduced).
template <int N, int R, int BT>
__device__ __forceinline__ void warp_gemv_rows(const bf16* const* __restrict__ wrows, const bf16* __restrict__ xs,
                                               int xs_stride, int B, float (*out)[BT]) {
    const int lane = threadIdx.x & 31;
    constexpr int NV = N / 256;
    uint4 wv[R][NV];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < NV; ++i) wv[r][i] = *reinterpret_cast<const uint4*>(wrows[r] + i * 256 + lane * 8);
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
        for (int b = 0; b < BT; ++b) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + b * xs_stride + k0);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
            for (int j = 0; j < 8; ++j) xf[b][j] = bf2f(xp[j]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const bf16* wp = reinterpret_cast<const bf16*>(&wv[r][i]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float wf = bf2f(wp[j]);
#pragma unroll
                for (int b = 0; b < BT; ++b) acc[r][b] += wf * xf[b][j];
            }
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

template <int EPL> __device__ __forceinline__ void fp8_to_f32(const void* v, float* f) {      // EPL e4m3 -> floats
    const unsigned short* p = reinterpret_cast<const unsigned short*>(v);
#pragma unroll
    for (int i = 0; i < EPL / 2; ++i) {
        const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(p[i], __NV_E4M3);
        const float2 t = __half22float2(__half2(r));
        f[2 * i] = t.x; f[2 * i + 1] = t.y;
    }
}
template <int EPL> __device__ __forceinline__ void bf16_to_f32(const void* v, float* f) {
    const bf16* p = reinterpret_cast<const bf16*>(v);
#pragma unroll
    for (int i = 0; i < EPL; ++i) f[i] = bf2f(p[i]);
}

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
        for (int j = 0; j < EPL / 8; ++j) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xp + j * 8);
            bf16_to_f32<8>(&xv, xf + j * 8);
        }
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

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define STAMP(i) do { if (a.times && L == 0 && blockIdx.x == 0 && threadIdx.x == 0) a.times[i] = gtimer(); } while (0)
#define STAMP_ALWAYS(i) do { if (a.times && blockIdx.x == 0 && threadIdx.x == 0) a.times[i] = gtimer(); } while (0)

template <int BT, bool FP8>
__global__ void __launch_bounds__(THREADS, 1) decode_step_kernel(Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    cg::grid_group grid = cg::this_grid();
    const int B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gwarp = blockIdx.x * WARPS + warp, nwarps = gridDim.x * WARPS;
    float* red = reinterpret_cast<float*>(smem_raw);                  // [WARPS] reduction scratch
    unsigned char* smem = smem_raw + 64;
    int maxlen = 0;
    for (int b = 0; b < B; ++b) maxlen = max(maxlen, a.pos[b] + 1);
    const int nsplit = (maxlen + CHUNK - 1) / CHUNK;                 // the same in every block

    for (int L = 0; L < a.n_layers; ++L) {
        const Layer lw = a.layers[L];
        const int* bt = a.bt + (size_t)L * B * a.bt_cols;
        STAMP(0);
        // ---- P1: the residual h (embedding at layer 0, else h1 + out2 of the previous layer) into shared memory in
        //      every block, block b writing row b out for P5; RMSNorm in place; the QKV GEMV over rows n.
        {
            bf16* xn = reinterpret_cast<bf16*>(smem);                  // [BT][H] (rows past B zero)
            for (int b = 0; b < BT; ++b) {
                if (b < B) {
                    if (L == 0) {
                        const bf16* src = a.embed + (size_t)a.tokens[b] * H;
                        for (int k = tid; k < H; k += THREADS) xn[b * H + k] = src[k];
                    } else {
                        for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(bf2f(a.sc.h1[b * H + k]) + a.sc.out2[b * H + k]);
                    }
                    if (blockIdx.x == b) for (int k = tid; k < H; k += THREADS) a.sc.h[b * H + k] = xn[b * H + k];
                    __syncthreads();
                    rmsnorm_row(xn + b * H, lw.in_norm, xn + b * H, H, red);
                } else {
                    for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(0.f);
                }
            }
            __syncthreads();
            constexpr int R1 = (BT <= 4) ? 2 : 1;                      // rows in flight per warp
            float out[R1][BT];
            for (int n0 = gwarp * R1; n0 < QKV; n0 += nwarps * R1) {
                const bf16* rows[R1];
#pragma unroll
                for (int r = 0; r < R1; ++r) rows[r] = lw.wqkv + (size_t)min(n0 + r, QKV - 1) * H;
                warp_gemv_rows<H, R1, BT>(rows, xn, H, B, out);
                if (lane == 0)
#pragma unroll
                    for (int r = 0; r < R1; ++r) if (n0 + r < QKV) for (int b = 0; b < B; ++b) a.sc.qkv[b * QKV + n0 + r] = f2bf(out[r][b]);
            }
        }
        STAMP(1);
        grid.sync();

        // ---- P2a: per (row b, kv group g): warps 0-7 each norm+rope one q head of the group (scaled, to fp32 scratch);
        //      warp 0 then norms+ropes k and writes k and v of the new token into the paged pool. A lane holds the
        //      elements lane, lane+32, lane+64, lane+96, so the rope pairs (i, i+64) never leave the lane.
        for (int item = blockIdx.x; item < B * NKV; item += gridDim.x) {
            const int g = item % NKV, b = item / NKV, pos = a.pos[b];
            float cs[2], sn[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) { const float f = pos * a.inv_freq[lane + 32 * u]; cs[u] = rbf(cosf(f)); sn[u] = rbf(sinf(f)); }
            {
                const int hq = g * GQA + warp;
                const bf16* qrow = a.sc.qkv + b * QKV + hq * D;
                float x[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) x[u] = bf2f(qrow[lane + 32 * u]);
                const float r = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / D + EPS);
                float n[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) n[u] = rbf(bf2f(lw.q_norm[lane + 32 * u]) * rbf(x[u] * r));
                float* qg = a.sc.q + ((size_t)b * NQ + hq) * D;
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    qg[lane + 32 * u] = rbf(n[u] * cs[u] - n[u + 2] * sn[u]) * a.scale;
                    qg[lane + 32 * u + D / 2] = rbf(n[u + 2] * cs[u] + n[u] * sn[u]) * a.scale;
                }
            }
            if (warp == 0) {
                const bf16* krow = a.sc.qkv + b * QKV + NQ * D + g * D;
                const bf16* vrow = a.sc.qkv + b * QKV + NQ * D + NKV * D + g * D;
                float x[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) x[u] = bf2f(krow[lane + 32 * u]);
                const float r = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / D + EPS);
                float n[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) n[u] = rbf(bf2f(lw.k_norm[lane + 32 * u]) * rbf(x[u] * r));
                const int bid = bt[b * a.bt_cols + pos / BS], slot = pos % BS;
                bf16* kdst = lw.pool_k + (((size_t)bid * NKV + g) * BS + slot) * D;
                bf16* vdst = lw.pool_v + (((size_t)bid * NKV + g) * BS + slot) * D;
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    kdst[lane + 32 * u] = f2bf(n[u] * cs[u] - n[u + 2] * sn[u]);
                    kdst[lane + 32 * u + D / 2] = f2bf(n[u + 2] * cs[u] + n[u] * sn[u]);
                }
#pragma unroll
                for (int u = 0; u < 4; ++u) vdst[lane + 32 * u] = vrow[lane + 32 * u];
            }
        }
        STAMP(2);
        grid.sync();

        // ---- P2b: attention per (row b, kv group g, 256-position chunk c). Thread t: scores of position t0 + t against
        //      the group's 8 q heads (k/v rows loaded with every load in flight), chunk softmax stats, then PV with each
        //      thread owning 8 consecutive d of one head over half the positions (16-byte smem loads, 8 FMAs per load).
        {
            float* q_s = reinterpret_cast<float*>(smem);                                   // [GQA][D]
            bf16* v_s = reinterpret_cast<bf16*>(smem + GQA * D * 4);                       // [CHUNK][D]
            float* p_s = reinterpret_cast<float*>(smem + GQA * D * 4 + CHUNK * D * 2);     // [GQA][CHUNK]
            float* red2 = p_s + GQA * CHUNK;                                               // [GQA][WARPS]
            float* pv_s = red2 + GQA * WARPS;                                              // [THREADS/2][8]
            for (int item = blockIdx.x; item < B * NKV * nsplit; item += gridDim.x) {
                const int c = item % nsplit, g = (item / nsplit) % NKV, b = item / (nsplit * NKV);
                const int pos = a.pos[b], kv_len = pos + 1, t0 = c * CHUNK;
                if (t0 >= kv_len) continue;
                const int tn = min(CHUNK, kv_len - t0);
                for (int i = tid; i < GQA * D; i += THREADS) q_s[i] = a.sc.q[((size_t)b * NQ + g * GQA) * D + i];
                __syncthreads();
                float s[GQA];
#pragma unroll
                for (int j = 0; j < GQA; ++j) s[j] = -INFINITY;
                if (tid < tn) {
                    const int t = t0 + tid;
                    const int bid = bt[b * a.bt_cols + t / BS], slot = t % BS;
                    const bf16* krow = lw.pool_k + (((size_t)bid * NKV + g) * BS + slot) * D;
                    const bf16* vrow = lw.pool_v + (((size_t)bid * NKV + g) * BS + slot) * D;
                    uint4 kv4[D / 8];
#pragma unroll
                    for (int i = 0; i < D / 8; ++i) kv4[i] = *reinterpret_cast<const uint4*>(krow + i * 8);
#pragma unroll
                    for (int i = 0; i < D / 8; ++i) *reinterpret_cast<uint4*>(v_s + tid * D + i * 8) = *reinterpret_cast<const uint4*>(vrow + i * 8);
#pragma unroll
                    for (int j = 0; j < GQA; ++j) s[j] = 0.f;
#pragma unroll
                    for (int i = 0; i < D / 8; ++i) {
                        float kf[8];
                        bf16_to_f32<8>(&kv4[i], kf);
#pragma unroll
                        for (int j = 0; j < GQA; ++j) {
                            const float* q = q_s + j * D + i * 8;
#pragma unroll
                            for (int jj = 0; jj < 8; ++jj) s[j] += q[jj] * kf[jj];
                        }
                    }
                }
                __syncthreads();
                float m[GQA], l[GQA];
#pragma unroll
                for (int j = 0; j < GQA; ++j) { const float wm = warp_max(s[j]); if (lane == 0) red2[j * WARPS + warp] = wm; }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < GQA; ++j) {
                    float mm = red2[j * WARPS];
#pragma unroll
                    for (int w = 1; w < WARPS; ++w) mm = fmaxf(mm, red2[j * WARPS + w]);
                    m[j] = mm;
                }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < GQA; ++j) {
                    const float pv = (tid < tn) ? __expf(s[j] - m[j]) : 0.f;
                    p_s[j * CHUNK + tid] = pv;
                    const float ws = warp_sum(pv);
                    if (lane == 0) red2[j * WARPS + warp] = ws;
                }
                __syncthreads();
#pragma unroll
                for (int j = 0; j < GQA; ++j) {
                    float ss = 0.f;
#pragma unroll
                    for (int w = 0; w < WARPS; ++w) ss += red2[j * WARPS + w];
                    l[j] = ss;
                }
                // PV: item = (head j, 8-wide d group dg), two threads per item over interleaved positions
                {
                    const int it = tid & 127, half = tid >> 7, j = it / (D / 8), d0 = (it % (D / 8)) * 8;
                    float acc[8];
#pragma unroll
                    for (int u = 0; u < 8; ++u) acc[u] = 0.f;
                    for (int t = half; t < tn; t += 2) {
                        const float p = p_s[j * CHUNK + t];
                        const uint4 vv = *reinterpret_cast<const uint4*>(v_s + t * D + d0);
                        float vf[8];
                        bf16_to_f32<8>(&vv, vf);
#pragma unroll
                        for (int u = 0; u < 8; ++u) acc[u] += p * vf[u];
                    }
                    if (half == 1) {
#pragma unroll
                        for (int u = 0; u < 8; ++u) pv_s[it * 8 + u] = acc[u];
                    }
                    __syncthreads();
                    if (half == 0) {
                        const int hq = g * GQA + j;
                        if (nsplit == 1) {
                            bf16* dst = a.sc.attn + b * (NQ * D) + hq * D + d0;
                            const float inv = 1.f / l[j];
#pragma unroll
                            for (int u = 0; u < 8; ++u) dst[u] = f2bf((acc[u] + pv_s[it * 8 + u]) * inv);
                        } else {
                            float* dst = a.sc.part_o + (((size_t)b * NQ + hq) * MAXSPLIT + c) * D + d0;
#pragma unroll
                            for (int u = 0; u < 8; ++u) dst[u] = acc[u] + pv_s[it * 8 + u];
                        }
                    }
                }
                if (nsplit > 1 && tid < GQA) {
                    const int hq = g * GQA + tid;
                    a.sc.part_m[((size_t)b * NQ + hq) * MAXSPLIT + c] = m[tid];
                    a.sc.part_l[((size_t)b * NQ + hq) * MAXSPLIT + c] = l[tid];
                }
                __syncthreads();
            }
        }
        STAMP(3);
        grid.sync();

        // ---- P3 (long contexts only): combine the chunks per (row, head) -> attn [B, NQ*D] bf16
        if (nsplit > 1) {
            for (int item = blockIdx.x; item < B * NQ; item += gridDim.x) {
                const int hq = item % NQ, b = item / NQ;
                const int n = (a.pos[b] + 1 + CHUNK - 1) / CHUNK;
                const float* pm = a.sc.part_m + ((size_t)b * NQ + hq) * MAXSPLIT;
                const float* pl = a.sc.part_l + ((size_t)b * NQ + hq) * MAXSPLIT;
                float M = -INFINITY;
                for (int c = 0; c < n; ++c) M = fmaxf(M, pm[c]);
                float Lsum = 0.f;
                for (int c = 0; c < n; ++c) Lsum += pl[c] * __expf(pm[c] - M);
                if (tid < D) {
                    float acc = 0.f;
                    for (int c = 0; c < n; ++c)
                        acc += a.sc.part_o[(((size_t)b * NQ + hq) * MAXSPLIT + c) * D + tid] * __expf(pm[c] - M);
                    a.sc.attn[b * (NQ * D) + hq * D + tid] = f2bf(acc / Lsum);
                }
            }
            grid.sync();
        }
        STAMP(4);

        // ---- P4: o_proj GEMV: o[b, n] = attn[b, :] . wo[n, :]
        {
            bf16* xs = reinterpret_cast<bf16*>(smem);                  // [BT][NQ*D]
            for (int i = tid; i < BT * NQ * D / 8; i += THREADS)
                reinterpret_cast<uint4*>(xs)[i] = (i < B * NQ * D / 8) ? reinterpret_cast<const uint4*>(a.sc.attn)[i] : make_uint4(0, 0, 0, 0);
            __syncthreads();
            constexpr int R4 = (BT <= 2) ? 2 : 1;
            float out[R4][BT];
            for (int n0 = gwarp * R4; n0 < H; n0 += nwarps * R4) {
                const bf16* rows[R4];
#pragma unroll
                for (int r = 0; r < R4; ++r) rows[r] = lw.wo + (size_t)min(n0 + r, H - 1) * (NQ * D);
                warp_gemv_rows<NQ * D, R4, BT>(rows, xs, NQ * D, B, out);
                if (lane == 0)
#pragma unroll
                    for (int r = 0; r < R4; ++r) if (n0 + r < H) for (int b = 0; b < B; ++b) a.sc.o[b * H + n0 + r] = f2bf(out[r][b]);
            }
        }
        STAMP(5);
        grid.sync();

        // ---- P5: every block: h1 = bf16(h + o) and n2 = RMSNorm(h1) for all rows into shared memory (block b also
        //      writes row b out and zeroes out2); router logits over the whole grid, one (row, expert) per warp.
        {
            bf16* n2s = reinterpret_cast<bf16*>(smem);                 // [BT][H]
            bf16* h1s = n2s + BT * H;                                  // [BT][H]
            for (int b = 0; b < B; ++b) {
                for (int k = tid; k < H; k += THREADS) h1s[b * H + k] = f2bf(bf2f(a.sc.h[b * H + k]) + bf2f(a.sc.o[b * H + k]));
                __syncthreads();
                rmsnorm_row(h1s + b * H, lw.post_norm, n2s + b * H, H, red);
                if (blockIdx.x == b) {
                    for (int k = tid; k < H; k += THREADS) { a.sc.h1[b * H + k] = h1s[b * H + k]; a.sc.n2[b * H + k] = n2s[b * H + k]; a.sc.out2[b * H + k] = 0.f; }
                }
            }
            __syncthreads();
            for (int w = gwarp; w < B * E; w += nwarps) {
                const int b = w / E, e = w % E;
                const bf16* wr = lw.wr + (size_t)e * H;
                const bf16* x = n2s + b * H;
                uint4 wv[H / 256];
#pragma unroll
                for (int i = 0; i < H / 256; ++i) wv[i] = *reinterpret_cast<const uint4*>(wr + i * 256 + lane * 8);
                float acc = 0.f;
#pragma unroll
                for (int i = 0; i < H / 256; ++i) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(x + i * 256 + lane * 8);
                    const bf16* wp = reinterpret_cast<const bf16*>(&wv[i]);
                    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc += bf2f(xp[j]) * bf2f(wp[j]);
                }
                acc = warp_sum(acc);
                if (lane == 0) a.sc.rlogits[b * E + e] = acc;
            }
        }
        STAMP(6);
        grid.sync();

        // ---- P6: top-8 per row in every block (one warp per row, 4 experts per lane: softmax in fp32, iterated
        //      argmax, lowest index on ties, renormalised weights; block 0 writes them out for P7), then the gate_up GEMV
        //      per (pair, n): act[pair][n] = bf16(silu(bf16(gate)) * bf16(up)).
        {
            bf16* xs = reinterpret_cast<bf16*>(smem);                  // n2 rows [BT][H]
            int* ids_s = reinterpret_cast<int*>(xs + BT * H);          // [BT*K]
            float* w_s = reinterpret_cast<float*>(ids_s + BT * K);     // [BT*K]
            for (int b = warp; b < B; b += WARPS) {
                float pr[4];
                float mx = -INFINITY;
#pragma unroll
                for (int u = 0; u < 4; ++u) { pr[u] = a.sc.rlogits[b * E + lane + 32 * u]; mx = fmaxf(mx, pr[u]); }
                mx = warp_max(mx);
                float sum = 0.f;
#pragma unroll
                for (int u = 0; u < 4; ++u) { pr[u] = __expf(pr[u] - mx); sum += pr[u]; }
                sum = warp_sum(sum);
#pragma unroll
                for (int u = 0; u < 4; ++u) pr[u] /= sum;
                float total = 0.f;
                for (int j = 0; j < K; ++j) {
                    float bv = fmaxf(fmaxf(pr[0], pr[1]), fmaxf(pr[2], pr[3]));
                    bv = warp_max(bv);
                    int idx = E;
#pragma unroll
                    for (int u = 3; u >= 0; --u) if (pr[u] == bv) idx = lane + 32 * u;
                    for (int o = 16; o > 0; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffff, idx, o));
                    if (lane == 0) { ids_s[b * K + j] = idx; w_s[b * K + j] = bv; }
                    total += bv;
#pragma unroll
                    for (int u = 0; u < 4; ++u) if (lane + 32 * u == idx) pr[u] = -1.f;
                }
                __syncwarp();
                if (lane < K) {
                    w_s[b * K + lane] /= total;
                    if (blockIdx.x == 0) { a.sc.ids[b * K + lane] = ids_s[b * K + lane]; a.sc.w[b * K + lane] = w_s[b * K + lane]; }
                }
            }
            for (int i = tid; i < B * H / 8; i += THREADS)
                reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(a.sc.n2)[i];
            __syncthreads();
            constexpr int NPW = FP8 ? 2 : 1;                           // n values per warp item (fp8 rows are half the bytes)
            constexpr int R6 = 2 * NPW;                                // gate and up rows per n
            const int pairs = B * K, items = pairs * (I / NPW);
            for (int w = gwarp; w < items; w += nwarps) {
                const int p = w / (I / NPW), n = (w % (I / NPW)) * NPW, b = p / K, e = ids_s[p];
                const void* rows[R6];
                const size_t ebase = (size_t)e * (2 * I);
#pragma unroll
                for (int q = 0; q < NPW; ++q) {
                    rows[2 * q] = reinterpret_cast<const char*>(lw.gate_up) + (ebase + n + q) * H * (FP8 ? 1 : 2);
                    rows[2 * q + 1] = reinterpret_cast<const char*>(lw.gate_up) + (ebase + I + n + q) * H * (FP8 ? 1 : 2);
                }
                float out[R6];
                warp_dot_rows<H, R6, FP8, 16>(rows, xs + b * H, out);
                if (lane == 0) {
#pragma unroll
                    for (int q = 0; q < NPW; ++q) {
                        float g = out[2 * q], u = out[2 * q + 1];
                        if (FP8) { g *= lw.gu_scale[ebase + n + q]; u *= lw.gu_scale[ebase + I + n + q]; }
                        const float gb = rbf(g), ub = rbf(u);
                        a.sc.act[(size_t)p * I + n + q] = f2bf(gb / (1.f + __expf(-gb)) * ub);
                    }
                }
            }
        }
        STAMP(7);
        grid.sync();

        // ---- P7: down GEMV per (pair, n): out2[b][n] += w[pair] * dot(act[pair], down[e][n])
        {
            const int pairs = B * K;
            constexpr int RD = FP8 ? 8 : 4;                            // rows in flight per warp (768 B / 1.5 KB each)
            constexpr int VB = FP8 ? 8 : 16;
            for (int w0 = gwarp * RD; w0 < pairs * H; w0 += nwarps * RD) {
                const void* rows[RD];
                int pp[RD], nn[RD];
#pragma unroll
                for (int r = 0; r < RD; ++r) {
                    const int w = min(w0 + r, pairs * H - 1);
                    pp[r] = w / H; nn[r] = w % H;
                    rows[r] = reinterpret_cast<const char*>(lw.down) + ((size_t)a.sc.ids[pp[r]] * H + nn[r]) * I * (FP8 ? 1 : 2);
                }
                // the RD rows may belong to different pairs (different x): dot them one at a time, loads still first
                float out[RD];
                {
                    constexpr int EPL = FP8 ? VB : VB / 2, CH = 32 * EPL, NV = I / CH;
                    using V = typename VecT<VB>::type;
                    V wv[RD][NV];
#pragma unroll
                    for (int r = 0; r < RD; ++r)
#pragma unroll
                        for (int i = 0; i < NV; ++i)
                            wv[r][i] = *reinterpret_cast<const V*>(reinterpret_cast<const char*>(rows[r]) + ((size_t)i * CH + lane * EPL) * (FP8 ? 1 : 2));
#pragma unroll
                    for (int r = 0; r < RD; ++r) {
                        const bf16* x = a.sc.act + (size_t)pp[r] * I;
                        float acc = 0.f;
#pragma unroll
                        for (int i = 0; i < NV; ++i) {
                            float xf[EPL], wf[EPL];
                            const bf16* xp = x + i * CH + lane * EPL;
#pragma unroll
                            for (int j = 0; j < EPL / 8; ++j) { const uint4 xv = *reinterpret_cast<const uint4*>(xp + j * 8); bf16_to_f32<8>(&xv, xf + j * 8); }
                            if constexpr (FP8) fp8_to_f32<EPL>(&wv[r][i], wf); else bf16_to_f32<EPL>(&wv[r][i], wf);
#pragma unroll
                            for (int j = 0; j < EPL; ++j) acc += wf[j] * xf[j];
                        }
                        out[r] = warp_sum(acc);
                    }
                }
                if (lane == 0) {
#pragma unroll
                    for (int r = 0; r < RD; ++r) if (w0 + r < pairs * H) {
                        float v = out[r];
                        if (FP8) v *= lw.down_scale[(size_t)a.sc.ids[pp[r]] * H + nn[r]];
                        atomicAdd(a.sc.out2 + (pp[r] / K) * H + nn[r], v * a.sc.w[pp[r]]);
                    }
                }
            }
        }
        STAMP(8);
        grid.sync();
    }
    STAMP_ALWAYS(12);

    // ---- PF: final norm of (h1 + out2) in every block, then the lm_head GEMV -> logits [B, V] bf16
    {
        bf16* xn = reinterpret_cast<bf16*>(smem);                      // [BT][H]
        for (int b = 0; b < BT; ++b) {
            if (b < B) {
                for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(bf2f(a.sc.h1[b * H + k]) + a.sc.out2[b * H + k]);
                __syncthreads();
                rmsnorm_row(xn + b * H, a.final_norm, xn + b * H, H, red);
            } else {
                for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(0.f);
            }
        }
        __syncthreads();
        constexpr int RF = (BT <= 4) ? 2 : 1;
        float out[RF][BT];
        for (int n0 = gwarp * RF; n0 < a.V; n0 += nwarps * RF) {
            const bf16* rows[RF];
#pragma unroll
            for (int r = 0; r < RF; ++r) rows[r] = a.lm_head + (size_t)min(n0 + r, a.V - 1) * H;
            warp_gemv_rows<H, RF, BT>(rows, xn, H, B, out);
            if (lane == 0)
#pragma unroll
                for (int r = 0; r < RF; ++r) if (n0 + r < a.V) for (int b = 0; b < B; ++b) a.logits[(size_t)b * a.V + n0 + r] = f2bf(out[r][b]);
        }
    }
    STAMP_ALWAYS(13);
}

// ------------------------------------------------------------------------------------------------ host side
static int num_sms() {
    static int n = 0;
    if (!n) { cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); }
    return n;
}

constexpr size_t SMEM_BYTES = 64 + (size_t)BMAX * NQ * D * 2;   // the largest phase: attn rows for o_proj (128 KB)
static_assert(64 + GQA * D * 4 + CHUNK * D * 2 + GQA * CHUNK * 4 + GQA * WARPS * 4 + (THREADS / 2) * 8 * 4 <= SMEM_BYTES, "attention smem");
static_assert(64 + 2 * BMAX * H * 2 <= SMEM_BYTES, "P5 smem");
static_assert(64 + BMAX * H * 2 + BMAX * K * 8 <= SMEM_BYTES, "P6 smem");

template <bool FP8> static void* pick(int B) {
    return B <= 1 ? (void*)decode_step_kernel<1, FP8> : B <= 2 ? (void*)decode_step_kernel<2, FP8> : B <= 4 ? (void*)decode_step_kernel<4, FP8>
         : B <= 8 ? (void*)decode_step_kernel<8, FP8> : (void*)decode_step_kernel<16, FP8>;
}

torch::Tensor decode_step(torch::Tensor tokens, torch::Tensor pos, torch::Tensor bt, torch::Tensor layers, torch::Tensor embed,
                          torch::Tensor final_norm, torch::Tensor lm_head, torch::Tensor inv_freq, torch::Tensor scratch_f32,
                          torch::Tensor scratch_bf16, torch::Tensor scratch_i32, double scale, bool fp8, torch::Tensor times) {
    TORCH_CHECK(tokens.is_cuda() && tokens.dtype() == torch::kInt64 && tokens.dim() == 1, "tokens [B] int64");
    const int B = tokens.size(0);
    TORCH_CHECK(B >= 1 && B <= BMAX, "batch 1..16");
    TORCH_CHECK(pos.dtype() == torch::kInt32 && pos.numel() == B, "pos [B] int32");
    TORCH_CHECK(bt.dtype() == torch::kInt32 && bt.dim() == 3 && bt.size(1) == B && bt.is_contiguous(), "bt [layers, B, cols] int32");
    TORCH_CHECK(layers.dtype() == torch::kInt64 && layers.dim() == 2 && layers.size(1) == 13 && layers.is_cuda() && layers.is_contiguous(), "layers [L, 13] int64");
    TORCH_CHECK(bt.size(0) == layers.size(0), "one block table per layer");
    TORCH_CHECK(bt.size(2) * BS <= MAXSPLIT * CHUNK, "context longer than MAXSPLIT * CHUNK");   // no host sync here
    TORCH_CHECK(embed.dtype() == torch::kBFloat16 && embed.size(1) == H && lm_head.dtype() == torch::kBFloat16 && lm_head.size(1) == H, "embed/lm_head [V, H] bf16");
    const int V = lm_head.size(0);
    auto logits = torch::empty({B, V}, tokens.options().dtype(torch::kBFloat16));
    Args a;
    a.tokens = tokens.data_ptr<int64_t>(); a.embed = reinterpret_cast<const bf16*>(embed.data_ptr());
    a.pos = pos.data_ptr<int>(); a.bt = bt.data_ptr<int>(); a.bt_cols = bt.size(2);
    a.layers = reinterpret_cast<const Layer*>(layers.data_ptr<int64_t>()); a.n_layers = layers.size(0);
    a.final_norm = reinterpret_cast<const bf16*>(final_norm.data_ptr()); a.lm_head = reinterpret_cast<const bf16*>(lm_head.data_ptr());
    a.V = V; a.logits = reinterpret_cast<bf16*>(logits.data_ptr()); a.B = B; a.scale = (float)scale; a.inv_freq = inv_freq.data_ptr<float>();
    float* f = scratch_f32.data_ptr<float>();
    bf16* s = reinterpret_cast<bf16*>(scratch_bf16.data_ptr());
    a.sc.q = f; f += (size_t)BMAX * NQ * D;
    a.sc.part_o = f; f += (size_t)BMAX * NQ * MAXSPLIT * D;
    a.sc.part_m = f; f += (size_t)BMAX * NQ * MAXSPLIT;
    a.sc.part_l = f; f += (size_t)BMAX * NQ * MAXSPLIT;
    a.sc.w = f; f += BMAX * K;
    a.sc.out2 = f; f += (size_t)BMAX * H;
    a.sc.rlogits = f;
    a.sc.h = s; s += (size_t)BMAX * H;
    a.sc.qkv = s; s += (size_t)BMAX * QKV;
    a.sc.attn = s; s += (size_t)BMAX * NQ * D;
    a.sc.o = s; s += (size_t)BMAX * H;
    a.sc.h1 = s; s += (size_t)BMAX * H;
    a.sc.n2 = s; s += (size_t)BMAX * H;
    a.sc.act = s;
    a.sc.ids = scratch_i32.data_ptr<int>();
    static bool attr_set = false;
    if (!attr_set) {
        for (int b : {1, 2, 4, 8, 16}) {
            cudaFuncSetAttribute(pick<true>(b), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            cudaFuncSetAttribute(pick<false>(b), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        }
        attr_set = true;
    }
    a.times = times.numel() ? reinterpret_cast<unsigned long long*>(times.data_ptr<int64_t>()) : nullptr;
    void* args[] = {&a};
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaLaunchCooperativeKernel(fp8 ? pick<true>(B) : pick<false>(B), dim3(num_sms()), dim3(THREADS), args, SMEM_BYTES, stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
    return logits;
}

size_t scratch_f32_elems() { return (size_t)BMAX * NQ * D + (size_t)BMAX * NQ * MAXSPLIT * D + 2 * (size_t)BMAX * NQ * MAXSPLIT + BMAX * K + (size_t)BMAX * H + (size_t)BMAX * E; }
size_t scratch_bf16_elems() { return (size_t)BMAX * (QKV + NQ * D + 4 * H) + (size_t)BMAX * K * I; }
size_t scratch_i32_elems() { return (size_t)BMAX * K; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_step", &decode_step);
    m.def("scratch_f32_elems", &scratch_f32_elems);
    m.def("scratch_bf16_elems", &scratch_bf16_elems);
    m.def("scratch_i32_elems", &scratch_i32_elems);
}
