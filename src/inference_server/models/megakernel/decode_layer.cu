// One persistent kernel per Qwen3-30B-A3B decoder layer, decode step, batch <= 16 (MEGAKERNEL=1).
//
// The eager decode step ran ~20 kernels a layer (5.4 ms of kernels in a 5.9 ms step at batch 1), each small one
// paying its ~2-3 us floor and none overlapping the weight stream. Here one cooperative kernel (one block per SM)
// does the whole layer in phases separated by grid-wide barriers; the GEMVs stream each weight row once through a
// warp, the attention is split over 256-position chunks and combined flash-decoding style, the experts run per
// (row, expert) pair. Numerics follow the torch/Triton path: fp32 accumulation, bf16 rounding where it rounds.
//
// Phases (grid.sync between each):
//   P1 RMSNorm(h) in every block (shared memory) + QKV GEMV           -> qkv [B, 5120] bf16
//   P2a per (row, kv group): q heads normed+roped+scaled from qkv -> q (fp32); k normed+roped and v of the new
//      token written into the paged pool
//   P2b attention per (row, kv group, 256-position chunk)           -> partial o/m/l per chunk (fp32)
//   P3 combine chunks per (row, head)                                -> attn_out [B, 4096] bf16
//   P4 o_proj GEMV                                                   -> o [B, 2048] bf16
//   P5 per row: h1 = bf16(h + o), n2 = RMSNorm(h1), router logits, softmax, top-8 (lowest index on ties),
//      renormalised weights; out2 zeroed                             -> h1, n2, ids, w
//   P6 experts gate_up GEMV per (pair, row n): act = bf16(silu(bf16(gate)) * bf16(up))   -> act [B*8, 768] bf16
//   P7 experts down GEMV per (pair, row n): out2[b, n] += w * dot (fp32 atomics)
//   P8 per row: h_out = bf16(h1 + out2)
// Dimensions are compile-time (the engine is specialised to this model).
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

constexpr int H = 2048, NQ = 32, NKV = 4, D = 128, QKV = NQ * D + 2 * NKV * D, I = 768, E = 128, K = 8;
constexpr int BS = 16, BMAX = 16, THREADS = 256, WARPS = THREADS / 32, CHUNK = 256, MAXSPLIT = 128;
constexpr int GQA = NQ / NKV;                                        // q heads per kv head
constexpr float EPS = 1e-6f;

using bf16 = __nv_bfloat16;

struct Weights {
    const bf16* in_norm;  const bf16* wqkv;  const bf16* q_norm;  const bf16* k_norm;  const bf16* wo;
    const bf16* post_norm; const bf16* wr;   const bf16* gate_up;  const bf16* down;    const float* inv_freq;
};
struct Scratch {
    bf16* qkv; float* q; float* part_o; float* part_m; float* part_l; bf16* attn; bf16* o; bf16* h1; bf16* n2;
    int* ids; float* w; bf16* act; float* out2; float* logits;
};
struct Args {
    const bf16* h; const int* pos; const int* bt; int bt_cols; bf16* pool_k; bf16* pool_v;   // pool [blocks, NKV, BS, D]
    bf16* h_out; int B; float scale; Weights wt; Scratch sc; unsigned long long* times;   // [16] ns, debug
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
// block-wide sum over THREADS values (uses red[WARPS])
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
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float m = red[0];
    for (int i = 1; i < WARPS; ++i) m = fmaxf(m, red[i]);
    return m;
}

// RMSNorm of one row of `n` bf16 values (whole block), torch rounding: y = bf16(w * bf16(x * rsqrt(mean(x^2)+eps))).
__device__ void rmsnorm_row(const bf16* x, const bf16* w, bf16* y, int n, float* red) {
    float ss = 0.f;
    for (int k = threadIdx.x; k < n; k += THREADS) { float v = bf2f(x[k]); ss += v * v; }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / n + EPS);
    for (int k = threadIdx.x; k < n; k += THREADS) y[k] = f2bf(bf2f(w[k]) * rbf(bf2f(x[k]) * r));
    __syncthreads();
}

// R weight rows (length N, N % 256 == 0) dotted with the BT shared-memory rows (B of them real), one warp: every
// load of the R rows is issued before any arithmetic (a warp needs several rows in flight to stream at bandwidth
// with 8 warps per SM). out[r][b] valid on lane 0.
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

__device__ __forceinline__ int block_of(const Args& a, int b, int t) { return a.bt[b * a.bt_cols + t / BS]; }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define STAMP(i) do { if (a.times && blockIdx.x == 0 && threadIdx.x == 0) a.times[i] = gtimer(); } while (0)

template <int BT>
__global__ void __launch_bounds__(THREADS, 1) decode_layer_kernel(Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    cg::grid_group grid = cg::this_grid();
    const int B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gwarp = blockIdx.x * WARPS + warp, nwarps = gridDim.x * WARPS;
    __shared__ int ne_shared;                                        // experts with pairs (P6/P7)
    float* red = reinterpret_cast<float*>(smem_raw);                  // [WARPS] reduction scratch
    unsigned char* smem = smem_raw + 64;

    STAMP(0);
    // ---- P1: xn = RMSNorm(h) for every row (each block, into shared memory), then the QKV GEMV over rows n.
    {
        bf16* xn = reinterpret_cast<bf16*>(smem);                      // [BT][H] (rows past B zero)
        for (int b = 0; b < BT; ++b) {
            if (b < B) rmsnorm_row(a.h + b * H, a.wt.in_norm, xn + b * H, H, red);
            else for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(0.f);
        }
        __syncthreads();
        constexpr int R1 = (BT <= 4) ? 2 : 1;                          // rows in flight per warp
        float out[R1][BT];
        for (int n0 = gwarp * R1; n0 < QKV; n0 += nwarps * R1) {
            const bf16* rows[R1];
#pragma unroll
            for (int r = 0; r < R1; ++r) rows[r] = a.wt.wqkv + (size_t)min(n0 + r, QKV - 1) * H;
            warp_gemv_rows<H, R1, BT>(rows, xn, H, B, out);
            if (lane == 0)
#pragma unroll
                for (int r = 0; r < R1; ++r) if (n0 + r < QKV) for (int b = 0; b < B; ++b) a.sc.qkv[b * QKV + n0 + r] = f2bf(out[r][b]);
        }
    }
    STAMP(1);
    grid.sync();

    // ---- P2a: per (row b, kv group g): warps 0-7 each norm+rope one q head of the group (scaled, to fp32 scratch);
    //          warp 0 then norms+ropes k and writes k and v of the new token into the paged pool. A lane holds the
    //          elements lane, lane+32, lane+64, lane+96, so the rope pairs (i, i+64) never leave the lane.
    for (int item = blockIdx.x; item < B * NKV; item += gridDim.x) {
        const int g = item % NKV, b = item / NKV, pos = a.pos[b];
        float cs[2], sn[2];                                      // cos/sin for i = lane, lane + 32 (bf16 tables)
#pragma unroll
        for (int u = 0; u < 2; ++u) { const float f = pos * a.wt.inv_freq[lane + 32 * u]; cs[u] = rbf(cosf(f)); sn[u] = rbf(sinf(f)); }
        {   // q head `warp` of this group
            const int hq = g * GQA + warp;
            const bf16* qrow = a.sc.qkv + b * QKV + hq * D;
            float x[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) x[u] = bf2f(qrow[lane + 32 * u]);
            const float r = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / D + EPS);
            float n[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) n[u] = rbf(bf2f(a.wt.q_norm[lane + 32 * u]) * rbf(x[u] * r));
            float* qg = a.sc.q + ((size_t)b * NQ + hq) * D;
#pragma unroll
            for (int u = 0; u < 2; ++u) {                        // pairs (u, u + 2): elements i and i + 64
                qg[lane + 32 * u] = rbf(n[u] * cs[u] - n[u + 2] * sn[u]) * a.scale;
                qg[lane + 32 * u + D / 2] = rbf(n[u + 2] * cs[u] + n[u] * sn[u]) * a.scale;
            }
        }
        if (warp == 0) {                                         // k (normed, roped) and v of the new token
            const bf16* krow = a.sc.qkv + b * QKV + NQ * D + g * D;
            const bf16* vrow = a.sc.qkv + b * QKV + NQ * D + NKV * D + g * D;
            float x[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) x[u] = bf2f(krow[lane + 32 * u]);
            const float r = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / D + EPS);
            float n[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) n[u] = rbf(bf2f(a.wt.k_norm[lane + 32 * u]) * rbf(x[u] * r));
            const int bid = block_of(a, b, pos), slot = pos % BS;
            bf16* kdst = a.pool_k + (((size_t)bid * NKV + g) * BS + slot) * D;
            bf16* vdst = a.pool_v + (((size_t)bid * NKV + g) * BS + slot) * D;
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

    // ---- P2b: attention per (row b, kv group g, 256-position chunk c): scores, chunk softmax stats, partial o.
    {
        float* q_s = reinterpret_cast<float*>(smem);                   // [GQA][D]
        bf16* v_s = reinterpret_cast<bf16*>(smem + GQA * D * 4);       // [CHUNK][D]
        float* p_s = reinterpret_cast<float*>(smem + GQA * D * 4 + CHUNK * D * 2);   // [GQA][CHUNK]
        int maxlen = 0;
        for (int b = 0; b < B; ++b) maxlen = max(maxlen, a.pos[b] + 1);
        const int nsplit = (maxlen + CHUNK - 1) / CHUNK;
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
            const int t = t0 + tid;
            if (tid < tn) {
                const int bid = block_of(a, b, t), slot = t % BS;
                const bf16* krow = a.pool_k + (((size_t)bid * NKV + g) * BS + slot) * D;
                const bf16* vrow = a.pool_v + (((size_t)bid * NKV + g) * BS + slot) * D;
#pragma unroll
                for (int j = 0; j < GQA; ++j) s[j] = 0.f;
#pragma unroll 2
                for (int i = 0; i < D / 8; ++i) {
                    const uint4 kv4 = *reinterpret_cast<const uint4*>(krow + i * 8);
                    const bf16* kp = reinterpret_cast<const bf16*>(&kv4);
                    *reinterpret_cast<uint4*>(v_s + tid * D + i * 8) = *reinterpret_cast<const uint4*>(vrow + i * 8);
                    float kf[8];
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) kf[jj] = bf2f(kp[jj]);
#pragma unroll
                    for (int j = 0; j < GQA; ++j) {
                        const float* q = q_s + j * D + i * 8;
#pragma unroll
                        for (int jj = 0; jj < 8; ++jj) s[j] += q[jj] * kf[jj];
                    }
                }
            }
            __syncthreads();
            float* red2 = p_s + GQA * CHUNK;                           // [GQA][WARPS] partials
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
            for (int i = 0; i < GQA * D / THREADS; ++i) {
                const int idx = tid + i * THREADS, j = idx / D, d = idx % D;
                float acc = 0.f;
                for (int tt = 0; tt < tn; ++tt) acc += p_s[j * CHUNK + tt] * bf2f(v_s[tt * D + d]);
                a.sc.part_o[(((size_t)b * NQ + g * GQA + j) * MAXSPLIT + c) * D + d] = acc;
            }
            if (tid < GQA) {
                const int hq = g * GQA + tid;
                a.sc.part_m[((size_t)b * NQ + hq) * MAXSPLIT + c] = m[tid];
                a.sc.part_l[((size_t)b * NQ + hq) * MAXSPLIT + c] = l[tid];
            }
            __syncthreads();
        }
    }
    STAMP(3);
    grid.sync();

    // ---- P3: combine the chunks per (row, head) -> attn [B, NQ*D] bf16
    {
        int maxlen = 0;
        for (int b = 0; b < B; ++b) maxlen = max(maxlen, a.pos[b] + 1);
        const int nsplit = (maxlen + CHUNK - 1) / CHUNK;
        for (int item = blockIdx.x; item < B * NQ; item += gridDim.x) {
            const int hq = item % NQ, b = item / NQ;
            const int n = (a.pos[b] + 1 + CHUNK - 1) / CHUNK;             // this row's chunks
            const float* pm = a.sc.part_m + ((size_t)b * NQ + hq) * MAXSPLIT;
            const float* pl = a.sc.part_l + ((size_t)b * NQ + hq) * MAXSPLIT;
            float M = -INFINITY;
            for (int c = 0; c < n; ++c) M = fmaxf(M, pm[c]);
            float L = 0.f;
            for (int c = 0; c < n; ++c) L += pl[c] * __expf(pm[c] - M);
            if (tid < D) {
                float acc = 0.f;
                for (int c = 0; c < n; ++c)
                    acc += a.sc.part_o[(((size_t)b * NQ + hq) * MAXSPLIT + c) * D + tid] * __expf(pm[c] - M);
                a.sc.attn[b * (NQ * D) + hq * D + tid] = f2bf(acc / L);
            }
            (void)nsplit;
        }
    }
    STAMP(4);
    grid.sync();

    // ---- P4: o_proj GEMV: o[b, n] = attn[b, :] . wo[n, :]
    {
        bf16* xs = reinterpret_cast<bf16*>(smem);                      // [BT][NQ*D]
        for (int i = tid; i < BT * NQ * D / 8; i += THREADS)
            reinterpret_cast<uint4*>(xs)[i] = (i < B * NQ * D / 8) ? reinterpret_cast<const uint4*>(a.sc.attn)[i] : make_uint4(0, 0, 0, 0);
        __syncthreads();
        constexpr int R4 = (BT <= 2) ? 2 : 1;
        float out[R4][BT];
        for (int n0 = gwarp * R4; n0 < H; n0 += nwarps * R4) {
            const bf16* rows[R4];
#pragma unroll
            for (int r = 0; r < R4; ++r) rows[r] = a.wt.wo + (size_t)min(n0 + r, H - 1) * (NQ * D);
            warp_gemv_rows<NQ * D, R4, BT>(rows, xs, NQ * D, B, out);
            if (lane == 0)
#pragma unroll
                for (int r = 0; r < R4; ++r) if (n0 + r < H) for (int b = 0; b < B; ++b) a.sc.o[b * H + n0 + r] = f2bf(out[r][b]);
        }
    }
    STAMP(5);
    grid.sync();

    // ---- P5a: every block: h1 = bf16(h + o) and n2 = RMSNorm(h1) for all rows into shared memory (blocks 0..B-1 also
    //          write them out and zero out2); router logits over the whole grid, one (row, expert) per warp.
    {
        bf16* n2s = reinterpret_cast<bf16*>(smem);                     // [BT][H]
        bf16* h1s = n2s + BT * H;                                      // [BT][H]
        for (int b = 0; b < B; ++b) {
            for (int k = tid; k < H; k += THREADS) h1s[b * H + k] = f2bf(bf2f(a.h[b * H + k]) + bf2f(a.sc.o[b * H + k]));
            __syncthreads();
            rmsnorm_row(h1s + b * H, a.wt.post_norm, n2s + b * H, H, red);
            if (blockIdx.x == b) {
                for (int k = tid; k < H; k += THREADS) { a.sc.h1[b * H + k] = h1s[b * H + k]; a.sc.n2[b * H + k] = n2s[b * H + k]; a.sc.out2[b * H + k] = 0.f; }
            }
        }
        __syncthreads();
        for (int w = gwarp; w < B * E; w += nwarps) {
            const int b = w / E, e = w % E;
            const bf16* wr = a.wt.wr + (size_t)e * H;
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
            if (lane == 0) a.sc.logits[b * E + e] = acc;
        }
    }
    STAMP(6);
    grid.sync();

    // ---- P5b: top-8 per row by one warp (4 experts per lane): softmax in fp32, iterated argmax, lowest index on ties.
    for (int b = gwarp; b < B; b += nwarps) {
        float pr[4];
        float mx = -INFINITY;
#pragma unroll
        for (int u = 0; u < 4; ++u) { pr[u] = a.sc.logits[b * E + lane + 32 * u]; mx = fmaxf(mx, pr[u]); }
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
            for (int u = 3; u >= 0; --u) if (pr[u] == bv) idx = lane + 32 * u;      // this lane's lowest match
            for (int o = 16; o > 0; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffff, idx, o));
            if (lane == 0) { a.sc.ids[b * K + j] = idx; a.sc.w[b * K + j] = bv; }
            total += bv;
#pragma unroll
            for (int u = 0; u < 4; ++u) if (lane + 32 * u == idx) pr[u] = -1.f;
        }
        if (lane < K) a.sc.w[b * K + lane] /= total;
    }
    STAMP(7);
    grid.sync();

    // ---- P6: experts gate_up, grouped by expert: a weight row is read once for every pair that routed to it
    //          (every block builds the grouping from ids: pairs sorted by expert, the experts with work).
    {
        bf16* xs = reinterpret_cast<bf16*>(smem);                      // n2 rows [B][H]
        int* cnt = reinterpret_cast<int*>(smem + (size_t)BT * H * 2);  // [E] pairs per expert
        int* start = cnt + E;                                          // [E] first slot of the expert's pairs
        int* fill = start + E;                                         // [E] placement cursor
        int* elist = fill + E;                                         // [E] experts with pairs, in index order
        int* gpairs = elist + E;                                       // [B*K] pair ids grouped by expert
        const int pairs = B * K;
        for (int i = tid; i < B * H / 8; i += THREADS)
            reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(a.sc.n2)[i];
        for (int e = tid; e < E; e += THREADS) { cnt[e] = 0; fill[e] = 0; }
        __syncthreads();
        for (int p = tid; p < pairs; p += THREADS) atomicAdd(&cnt[a.sc.ids[p]], 1);
        __syncthreads();
        if (tid == 0) {
            int acc = 0, k = 0;
            for (int e = 0; e < E; ++e) { start[e] = acc; acc += cnt[e]; if (cnt[e]) elist[k++] = e; }
            ne_shared = k;
        }
        __syncthreads();
        for (int p = tid; p < pairs; p += THREADS) { const int e = a.sc.ids[p]; gpairs[start[e] + atomicAdd(&fill[e], 1)] = p; }
        __syncthreads();
        const int ne_ = ne_shared;
        for (int w = gwarp; w < ne_ * I; w += nwarps) {
            const int e = elist[w / I], n = w % I, c = cnt[e], s0 = start[e];
            const bf16* wg = a.wt.gate_up + ((size_t)e * (2 * I) + n) * H;
            const bf16* wu = a.wt.gate_up + ((size_t)e * (2 * I) + I + n) * H;
            uint4 gv[H / 256], uv[H / 256];
#pragma unroll
            for (int i = 0; i < H / 256; ++i) { gv[i] = *reinterpret_cast<const uint4*>(wg + i * 256 + lane * 8); uv[i] = *reinterpret_cast<const uint4*>(wu + i * 256 + lane * 8); }
#pragma unroll
            for (int q = 0; q < BT; ++q) {
                if (q < c) {
                    const int p = gpairs[s0 + q];
                    const bf16* x = xs + (p / K) * H;
                    float g = 0.f, u = 0.f;
#pragma unroll
                    for (int i = 0; i < H / 256; ++i) {
                        const uint4 xv = *reinterpret_cast<const uint4*>(x + i * 256 + lane * 8);
                        const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                        const bf16* gp = reinterpret_cast<const bf16*>(&gv[i]);
                        const bf16* up = reinterpret_cast<const bf16*>(&uv[i]);
#pragma unroll
                        for (int j = 0; j < 8; ++j) { const float xf = bf2f(xp[j]); g += xf * bf2f(gp[j]); u += xf * bf2f(up[j]); }
                    }
                    g = warp_sum(g); u = warp_sum(u);
                    if (lane == 0) { const float gb = rbf(g), ub = rbf(u); a.sc.act[(size_t)p * I + n] = f2bf(gb / (1.f + __expf(-gb)) * ub); }
                }
            }
        }
    }
    STAMP(8);
    grid.sync();

    // ---- P7: experts down, grouped by expert (the grouping arrays survive in shared memory): out2[b][n] +=
    //          w[pair] * dot(act[pair], down[e][n]); RD rows in flight per warp.
    {
        int* cnt = reinterpret_cast<int*>(smem + (size_t)BT * H * 2);
        int* start = cnt + E; int* elist = start + 2 * E; int* gpairs = elist + E;
        const int ne_ = ne_shared;
        constexpr int RD = 4;
        for (int w0 = gwarp * RD; w0 < ne_ * H; w0 += nwarps * RD) {
            uint4 wv[RD][I / 256];
            int ee[RD], nn[RD];
#pragma unroll
            for (int r = 0; r < RD; ++r) {
                const int w = min(w0 + r, ne_ * H - 1);
                ee[r] = elist[w / H]; nn[r] = w % H;
                const bf16* wd = a.wt.down + ((size_t)ee[r] * H + nn[r]) * I;
#pragma unroll
                for (int i = 0; i < I / 256; ++i) wv[r][i] = *reinterpret_cast<const uint4*>(wd + i * 256 + lane * 8);
            }
#pragma unroll
            for (int r = 0; r < RD; ++r) {
                if (w0 + r < ne_ * H) {
                    const int c = cnt[ee[r]], s0 = start[ee[r]];
                    for (int q = 0; q < c; ++q) {
                        const int p = gpairs[s0 + q];
                        const bf16* x = a.sc.act + (size_t)p * I;
                        float acc = 0.f;
#pragma unroll
                        for (int i = 0; i < I / 256; ++i) {
                            const uint4 xv = *reinterpret_cast<const uint4*>(x + i * 256 + lane * 8);
                            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                            const bf16* wp = reinterpret_cast<const bf16*>(&wv[r][i]);
#pragma unroll
                            for (int j = 0; j < 8; ++j) acc += bf2f(xp[j]) * bf2f(wp[j]);
                        }
                        acc = warp_sum(acc);
                        if (lane == 0) atomicAdd(a.sc.out2 + (p / K) * H + nn[r], acc * a.sc.w[p]);
                    }
                }
            }
        }
    }
    STAMP(9);
    grid.sync();

    // ---- P8: h_out = bf16(h1 + out2)
    for (int i = blockIdx.x * THREADS + tid; i < B * H; i += gridDim.x * THREADS)
        a.h_out[i] = f2bf(bf2f(a.sc.h1[i]) + a.sc.out2[i]);
    STAMP(10);
}

// ------------------------------------------------------------------------------------------------ host side
static int num_sms() {
    static int n = 0;
    if (!n) { cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); }
    return n;
}

constexpr size_t SMEM_BYTES = 64 + (size_t)BMAX * NQ * D * 2;   // the largest phase: attn rows for o_proj (128 KB)

torch::Tensor decode_layer(torch::Tensor h, torch::Tensor pos, torch::Tensor bt, torch::Tensor pool_k, torch::Tensor pool_v,
                           torch::Tensor in_norm, torch::Tensor wqkv, torch::Tensor q_norm, torch::Tensor k_norm,
                           torch::Tensor wo, torch::Tensor post_norm, torch::Tensor wr, torch::Tensor gate_up,
                           torch::Tensor down, torch::Tensor inv_freq, torch::Tensor scratch_f32, torch::Tensor scratch_bf16,
                           torch::Tensor scratch_i32, double scale, torch::Tensor times) {
    TORCH_CHECK(h.is_cuda() && h.dtype() == torch::kBFloat16 && h.dim() == 2 && h.size(1) == H, "h [B, 2048] bf16");
    const int B = h.size(0);
    TORCH_CHECK(B >= 1 && B <= BMAX, "batch 1..16");
    TORCH_CHECK(pos.dtype() == torch::kInt32 && bt.dtype() == torch::kInt32, "pos/bt int32");
    auto h_out = torch::empty_like(h);
    Args a;
    a.h = reinterpret_cast<const bf16*>(h.data_ptr()); a.pos = pos.data_ptr<int>(); a.bt = bt.data_ptr<int>();
    a.bt_cols = bt.size(1); a.pool_k = reinterpret_cast<bf16*>(pool_k.data_ptr()); a.pool_v = reinterpret_cast<bf16*>(pool_v.data_ptr());
    a.h_out = reinterpret_cast<bf16*>(h_out.data_ptr()); a.B = B; a.scale = (float)scale;
    a.wt = {reinterpret_cast<const bf16*>(in_norm.data_ptr()), reinterpret_cast<const bf16*>(wqkv.data_ptr()),
            reinterpret_cast<const bf16*>(q_norm.data_ptr()), reinterpret_cast<const bf16*>(k_norm.data_ptr()),
            reinterpret_cast<const bf16*>(wo.data_ptr()), reinterpret_cast<const bf16*>(post_norm.data_ptr()),
            reinterpret_cast<const bf16*>(wr.data_ptr()), reinterpret_cast<const bf16*>(gate_up.data_ptr()),
            reinterpret_cast<const bf16*>(down.data_ptr()), inv_freq.data_ptr<float>()};
    // scratch carve-up (sized for BMAX)
    float* f = scratch_f32.data_ptr<float>();
    bf16* s = reinterpret_cast<bf16*>(scratch_bf16.data_ptr());
    a.sc.q = f; f += (size_t)BMAX * NQ * D;
    a.sc.part_o = f; f += (size_t)BMAX * NQ * MAXSPLIT * D;
    a.sc.part_m = f; f += (size_t)BMAX * NQ * MAXSPLIT;
    a.sc.part_l = f; f += (size_t)BMAX * NQ * MAXSPLIT;
    a.sc.w = f; f += BMAX * K;
    a.sc.out2 = f; f += (size_t)BMAX * H;
    a.sc.logits = f;
    a.sc.qkv = s; s += (size_t)BMAX * QKV;
    a.sc.attn = s; s += (size_t)BMAX * NQ * D;
    a.sc.o = s; s += (size_t)BMAX * H;
    a.sc.h1 = s; s += (size_t)BMAX * H;
    a.sc.n2 = s; s += (size_t)BMAX * H;
    a.sc.act = s;
    a.sc.ids = scratch_i32.data_ptr<int>();
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(decode_layer_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        cudaFuncSetAttribute(decode_layer_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        cudaFuncSetAttribute(decode_layer_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        cudaFuncSetAttribute(decode_layer_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        cudaFuncSetAttribute(decode_layer_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        attr_set = true;
    }
    TORCH_CHECK(bt.size(1) * BS <= MAXSPLIT * CHUNK, "context longer than MAXSPLIT * CHUNK");   // no host sync here
    a.times = times.numel() ? reinterpret_cast<unsigned long long*>(times.data_ptr<int64_t>()) : nullptr;
    const int grid = num_sms();
    void* args[] = {&a};
    auto stream = at::cuda::getCurrentCUDAStream();
    void* kern = B <= 1 ? (void*)decode_layer_kernel<1> : B <= 2 ? (void*)decode_layer_kernel<2> : B <= 4 ? (void*)decode_layer_kernel<4>
               : B <= 8 ? (void*)decode_layer_kernel<8> : (void*)decode_layer_kernel<16>;
    cudaError_t err = cudaLaunchCooperativeKernel(kern, dim3(grid), dim3(THREADS), args, SMEM_BYTES, stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
    return h_out;
}

size_t scratch_f32_elems() { return (size_t)BMAX * NQ * D + (size_t)BMAX * NQ * MAXSPLIT * D + 2 * (size_t)BMAX * NQ * MAXSPLIT + BMAX * K + (size_t)BMAX * H + (size_t)BMAX * E; }
size_t scratch_bf16_elems() { return (size_t)BMAX * (QKV + NQ * D + 3 * H) + (size_t)BMAX * K * I; }
size_t scratch_i32_elems() { return (size_t)BMAX * K; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_layer", &decode_layer);
    m.def("scratch_f32_elems", &scratch_f32_elems);
    m.def("scratch_bf16_elems", &scratch_bf16_elems);
    m.def("scratch_i32_elems", &scratch_i32_elems);
}
