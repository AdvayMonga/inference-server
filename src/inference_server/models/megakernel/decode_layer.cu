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
    int* ids; float* w; bf16* act; float* out2;
};
struct Args {
    const bf16* h; const int* pos; const int* bt; int bt_cols; bf16* pool_k; bf16* pool_v;   // pool [blocks, NKV, BS, D]
    bf16* h_out; int B; float scale; Weights wt; Scratch sc;
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

// dot of a bf16 weight row (length n, n % 256 == 0) with up to B bf16 rows held in shared memory, one warp:
// lane handles elements lane*8 + 256*i. Results per b in `out` (lane 0 valid).
template <int N>
__device__ __forceinline__ void warp_gemv_row(const bf16* __restrict__ wrow, const bf16* __restrict__ xs,
                                              int xs_stride, int B, float* out) {
    const int lane = threadIdx.x & 31;
    float acc[BMAX];
#pragma unroll
    for (int b = 0; b < BMAX; ++b) acc[b] = 0.f;
#pragma unroll 4
    for (int i = 0; i < N / 256; ++i) {
        const int k0 = i * 256 + lane * 8;
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + k0);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
        for (int b = 0; b < B; ++b) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + b * xs_stride + k0);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) a += bf2f(wp[j]) * bf2f(xp[j]);
            acc[b] += a;
        }
    }
    for (int b = 0; b < B; ++b) out[b] = warp_sum(acc[b]);
}

__device__ __forceinline__ int block_of(const Args& a, int b, int t) { return a.bt[b * a.bt_cols + t / BS]; }

__global__ void __launch_bounds__(THREADS, 1) decode_layer_kernel(Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    cg::grid_group grid = cg::this_grid();
    const int B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gwarp = blockIdx.x * WARPS + warp, nwarps = gridDim.x * WARPS;
    float* red = reinterpret_cast<float*>(smem_raw);                  // [WARPS] reduction scratch
    unsigned char* smem = smem_raw + 64;

    // ---- P1: xn = RMSNorm(h) for every row (each block, into shared memory), then the QKV GEMV over rows n.
    {
        bf16* xn = reinterpret_cast<bf16*>(smem);                      // [B][H]
        for (int b = 0; b < B; ++b) rmsnorm_row(a.h + b * H, a.wt.in_norm, xn + b * H, H, red);
        float out[BMAX];
        for (int n = gwarp; n < QKV; n += nwarps) {
            warp_gemv_row<H>(a.wt.wqkv + (size_t)n * H, xn, H, B, out);
            if (lane == 0) for (int b = 0; b < B; ++b) a.sc.qkv[b * QKV + n] = f2bf(out[b]);
        }
    }
    grid.sync();

    // ---- P2a: per (row b, kv group g): q heads normed, roped, scaled -> q_g (fp32 scratch); k/v of the new token
    //          normed/roped and written into the paged pool.
    for (int item = blockIdx.x; item < B * NKV; item += gridDim.x) {
        const int g = item % NKV, b = item / NKV, pos = a.pos[b];
        const float inv_freq = (tid < D / 2) ? a.wt.inv_freq[tid] : 0.f;
        const float cosv = rbf(cosf(pos * inv_freq)), sinv = rbf(sinf(pos * inv_freq));   // the bf16 tables
        __shared__ float qn[D];
        for (int j = 0; j < GQA; ++j) {
            const int hq = g * GQA + j;
            const bf16* qrow = a.sc.qkv + b * QKV + hq * D;
            const float x = (tid < D) ? bf2f(qrow[tid]) : 0.f;
            const float r = rsqrtf(block_sum(x * x, red) / D + EPS);
            if (tid < D) qn[tid] = rbf(bf2f(a.wt.q_norm[tid]) * rbf(x * r));
            __syncthreads();
            if (tid < D / 2) {
                const float x1 = qn[tid], x2 = qn[tid + D / 2];
                float* qg = a.sc.q + ((size_t)b * NQ + hq) * D;
                qg[tid] = rbf(x1 * cosv - x2 * sinv) * a.scale;
                qg[tid + D / 2] = rbf(x2 * cosv + x1 * sinv) * a.scale;
            }
            __syncthreads();
        }
        const bf16* krow = a.sc.qkv + b * QKV + NQ * D + g * D;
        const bf16* vrow = a.sc.qkv + b * QKV + NQ * D + NKV * D + g * D;
        const float x = (tid < D) ? bf2f(krow[tid]) : 0.f;
        const float r = rsqrtf(block_sum(x * x, red) / D + EPS);
        if (tid < D) qn[tid] = rbf(bf2f(a.wt.k_norm[tid]) * rbf(x * r));
        __syncthreads();
        const int bid = block_of(a, b, pos), slot = pos % BS;
        bf16* kdst = a.pool_k + (((size_t)bid * NKV + g) * BS + slot) * D;
        bf16* vdst = a.pool_v + (((size_t)bid * NKV + g) * BS + slot) * D;
        if (tid < D / 2) {
            const float x1 = qn[tid], x2 = qn[tid + D / 2];
            kdst[tid] = f2bf(x1 * cosv - x2 * sinv);
            kdst[tid + D / 2] = f2bf(x2 * cosv + x1 * sinv);
        }
        if (tid < D) vdst[tid] = vrow[tid];
        __syncthreads();
    }
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
            float m[GQA], l[GQA];
#pragma unroll
            for (int j = 0; j < GQA; ++j) {
                m[j] = block_max(s[j], red);
                const float pv = (tid < tn) ? __expf(s[j] - m[j]) : 0.f;
                p_s[j * CHUNK + tid] = pv;
                l[j] = block_sum(pv, red);
            }
            __syncthreads();
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
    grid.sync();

    // ---- P4: o_proj GEMV: o[b, n] = attn[b, :] . wo[n, :]
    {
        bf16* xs = reinterpret_cast<bf16*>(smem);                      // [B][NQ*D]
        for (int i = tid; i < B * NQ * D / 8; i += THREADS)
            reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(a.sc.attn)[i];
        __syncthreads();
        float out[BMAX];
        for (int n = gwarp; n < H; n += nwarps) {
            warp_gemv_row<NQ * D>(a.wt.wo + (size_t)n * (NQ * D), xs, NQ * D, B, out);
            if (lane == 0) for (int b = 0; b < B; ++b) a.sc.o[b * H + n] = f2bf(out[b]);
        }
    }
    grid.sync();

    // ---- P5: per row: residual, post norm, router, top-k; out2 zeroed
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        bf16* h1 = a.sc.h1 + b * H;
        bf16* n2 = a.sc.n2 + b * H;
        for (int k = tid; k < H; k += THREADS) {
            h1[k] = f2bf(bf2f(a.h[b * H + k]) + bf2f(a.sc.o[b * H + k]));
            a.sc.out2[b * H + k] = 0.f;
        }
        __syncthreads();
        bf16* n2s = reinterpret_cast<bf16*>(smem);                     // [H]
        rmsnorm_row(h1, a.wt.post_norm, n2s, H, red);
        for (int k = tid; k < H; k += THREADS) n2[k] = n2s[k];
        __shared__ float logits[E];
        __shared__ float probs[E];
        // router: 8 warps, 16 experts each; lanes split k
        for (int e = warp; e < E; e += WARPS) {
            const bf16* wr = a.wt.wr + (size_t)e * H;
            float acc = 0.f;
            for (int k = lane; k < H; k += 32) acc += bf2f(n2s[k]) * bf2f(wr[k]);
            acc = warp_sum(acc);
            if (lane == 0) logits[e] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            float mx = -INFINITY;
            for (int e = 0; e < E; ++e) mx = fmaxf(mx, logits[e]);
            float sum = 0.f;
            for (int e = 0; e < E; ++e) { probs[e] = __expf(logits[e] - mx); sum += probs[e]; }
            for (int e = 0; e < E; ++e) probs[e] /= sum;
            float total = 0.f;
            for (int j = 0; j < K; ++j) {                                 // iterated argmax, lowest index on ties
                int best = 0; float bv = -1.f;
                for (int e = 0; e < E; ++e) if (probs[e] > bv) { bv = probs[e]; best = e; }
                a.sc.ids[b * K + j] = best; a.sc.w[b * K + j] = bv; total += bv; probs[best] = -1.f;
            }
            for (int j = 0; j < K; ++j) a.sc.w[b * K + j] /= total;      // norm_topk_prob
        }
        __syncthreads();
    }
    grid.sync();

    // ---- P6: gate_up GEMV per (pair, n): act[pair][n] = bf16(silu(bf16(gate)) * bf16(up))
    {
        bf16* xs = reinterpret_cast<bf16*>(smem);                      // n2 rows [B][H]
        for (int i = tid; i < B * H / 8; i += THREADS)
            reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(a.sc.n2)[i];
        __syncthreads();
        const int pairs = B * K;
        for (int w = gwarp; w < pairs * I; w += nwarps) {
            const int p = w / I, n = w % I, b = p / K, e = a.sc.ids[p];
            const bf16* wg = a.wt.gate_up + ((size_t)e * (2 * I) + n) * H;
            const bf16* wu = a.wt.gate_up + ((size_t)e * (2 * I) + I + n) * H;
            // one row's dot: lanes split k
            float g = 0.f, u = 0.f;
            const bf16* x = xs + b * H;
            for (int k = lane * 8; k < H; k += 256) {
                const uint4 xv = *reinterpret_cast<const uint4*>(x + k);
                const uint4 gv = *reinterpret_cast<const uint4*>(wg + k);
                const uint4 uv = *reinterpret_cast<const uint4*>(wu + k);
                const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                const bf16* gp = reinterpret_cast<const bf16*>(&gv);
                const bf16* up = reinterpret_cast<const bf16*>(&uv);
#pragma unroll
                for (int j = 0; j < 8; ++j) { const float xf = bf2f(xp[j]); g += xf * bf2f(gp[j]); u += xf * bf2f(up[j]); }
            }
            g = warp_sum(g); u = warp_sum(u);
            if (lane == 0) {
                const float gb = rbf(g), ub = rbf(u);
                a.sc.act[(size_t)p * I + n] = f2bf(gb / (1.f + __expf(-gb)) * ub);
            }
        }
    }
    grid.sync();

    // ---- P7: down GEMV per (pair, n): out2[b][n] += w[pair] * dot(act[pair], down[e][n])
    {
        const int pairs = B * K;
        for (int w = gwarp; w < pairs * H; w += nwarps) {
            const int p = w / H, n = w % H, b = p / K, e = a.sc.ids[p];
            const bf16* wd = a.wt.down + ((size_t)e * H + n) * I;
            const bf16* x = a.sc.act + (size_t)p * I;
            float acc = 0.f;
            for (int k = lane * 8; k < I; k += 256) {
                const uint4 xv = *reinterpret_cast<const uint4*>(x + k);
                const uint4 wv = *reinterpret_cast<const uint4*>(wd + k);
                const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc += bf2f(xp[j]) * bf2f(wp[j]);
            }
            acc = warp_sum(acc);
            if (lane == 0) atomicAdd(a.sc.out2 + b * H + n, acc * a.sc.w[p]);
        }
    }
    grid.sync();

    // ---- P8: h_out = bf16(h1 + out2)
    for (int i = blockIdx.x * THREADS + tid; i < B * H; i += gridDim.x * THREADS)
        a.h_out[i] = f2bf(bf2f(a.sc.h1[i]) + a.sc.out2[i]);
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
                           torch::Tensor scratch_i32, double scale) {
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
    a.sc.out2 = f;
    a.sc.qkv = s; s += (size_t)BMAX * QKV;
    a.sc.attn = s; s += (size_t)BMAX * NQ * D;
    a.sc.o = s; s += (size_t)BMAX * H;
    a.sc.h1 = s; s += (size_t)BMAX * H;
    a.sc.n2 = s; s += (size_t)BMAX * H;
    a.sc.act = s;
    a.sc.ids = scratch_i32.data_ptr<int>();
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(decode_layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
        attr_set = true;
    }
    int maxlen = 0;   // the attention item count must fit the grid (one pass; see the kernel note)
    {
        auto pos_h = pos.to(torch::kCPU);
        for (int b = 0; b < B; ++b) maxlen = std::max(maxlen, pos_h.data_ptr<int>()[b] + 1);
    }
    const int nsplit = (maxlen + CHUNK - 1) / CHUNK;
    TORCH_CHECK(nsplit <= MAXSPLIT, "context longer than MAXSPLIT * CHUNK");
    const int grid = num_sms();
    void* args[] = {&a};
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaLaunchCooperativeKernel((void*)decode_layer_kernel, dim3(grid), dim3(THREADS), args, SMEM_BYTES, stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
    return h_out;
}

size_t scratch_f32_elems() { return (size_t)BMAX * NQ * D + (size_t)BMAX * NQ * MAXSPLIT * D + 2 * (size_t)BMAX * NQ * MAXSPLIT + BMAX * K + (size_t)BMAX * H; }
size_t scratch_bf16_elems() { return (size_t)BMAX * (QKV + NQ * D + 3 * H) + (size_t)BMAX * K * I; }
size_t scratch_i32_elems() { return (size_t)BMAX * K; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_layer", &decode_layer);
    m.def("scratch_f32_elems", &scratch_f32_elems);
    m.def("scratch_bf16_elems", &scratch_bf16_elems);
    m.def("scratch_i32_elems", &scratch_i32_elems);
}
