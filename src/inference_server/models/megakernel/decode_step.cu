// One persistent kernel for a whole Qwen3-30B-A3B decode step, batch <= 16 (MEGAKERNEL=1): embedding, the 48
// decoder layers, the final norm and lm_head, in one cooperative launch (one block per SM), phases separated by
// grid-wide barriers. v1 (decode_layer.cu, 2026-10-18) was one launch per layer: 68 us a layer at batch 1 against a
// ~28 us byte floor, plus 47 launch gaps and the torch lm_head. The phases are latency-bound at small batch (a few
// items per warp, each a DRAM round trip, then a barrier), so each GEMV phase's first weight rows are loaded into
// registers before the barrier that precedes it, and the attention phase prepares its own q/k/v. Per layer:
//   P1 residual h = (L == 0 ? embed[token] : h1 + out2) in every block, RMSNorm, QKV GEMV      -> qkv [B, 5120]
//   P2 attention per (row, kv group, 256-position chunk): the block norms+ropes the group's q heads itself, the
//      chunk holding the new token writes its k/v into the paged pool first; one chunk -> attn, else partial o/m/l
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
// Built per (MEGA_KV8, MEGA_TMA): the KV pools' dtype and the weight ring are decided at build time (megakernel/_ext),
// so neither path carries the other's code. Measured 2026-10-20 with both as runtime flags: the base path (bf16 KV,
// no ring) was 5.21 ms a step against 4.48 for the kernel without them (B=1, 37-token context, fp8 experts).
#include "common.cuh"

struct Args {
    const int64_t* tokens; const bf16* embed; const int64_t* pos; const int64_t* bt; int bt_cols;   // bt [B, cols], every layer
    const Layer* layers; int n_layers; const bf16* final_norm; const bf16* lm_head; int V; bf16* logits;
    unsigned long long* keys; int64_t* tokens_out;      // greedy argmax: packed (ordered bf16, ~index) per row, the token ids
    int B; float scale; const float* inv_freq; Scratch sc; unsigned long long* times;      // times [16] ns, debug
    int stamp_layer;                                                                       // the layer `times` stamps
    int times_blocks;                                                                      // times is [blocks, 16]
    int kv8;                                                                               // the pools hold fp8 e4m3
    int tma;                                                                               // MEGAKERNEL_TMA (batch <= 2)
};

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// Stamps: block 0's, or every block's ([blocks, 16], a.times_blocks) to see each phase's skew across the grid.
#define STAMP(i) do { if (a.times && L == a.stamp_layer && threadIdx.x == 0 && (a.times_blocks || blockIdx.x == 0)) a.times[(a.times_blocks ? blockIdx.x * 16 : 0) + (i)] = gtimer(); } while (0)
#define STAMP_ALWAYS(i) do { if (a.times && threadIdx.x == 0 && (a.times_blocks || blockIdx.x == 0)) a.times[(a.times_blocks ? blockIdx.x * 16 : 0) + (i)] = gtimer(); } while (0)

template <int BT, bool FP8>
__global__ void __launch_bounds__(THREADS, MEGA_BPS) decode_step_kernel(Args a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    cg::grid_group grid = cg::this_grid();
    const int B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gwarp = blockIdx.x * WARPS + warp, nwarps = gridDim.x * WARPS;
    float* red = reinterpret_cast<float*>(smem_raw);                  // [WARPS] reduction scratch
    unsigned char* smem = smem_raw + HDR;
    int maxlen = 0;
    for (int b = 0; b < B; ++b) maxlen = max(maxlen, (int)a.pos[b] + 1);
    const int nsplit = (maxlen + CHUNK - 1) / CHUNK;                 // the same in every block

    constexpr int RA = (BT <= 4) ? 2 : 1;                            // rows per warp item, the H-long GEMVs (qkv, lm_head)
    constexpr int R4 = (BT <= 2) ? 2 : 1;                            // o_proj
    using DN = Down<FP8>;
    constexpr bool tma = MEGA_TMA && (BT <= 2);                      // weights streamed through the shared-memory ring
    unsigned long long* bars = reinterpret_cast<unsigned long long*>(smem_raw + 128);  // [NST], after red[WARPS]
    unsigned char* ring = smem_raw + RING_OFF;
    uint32_t ph = 0;
    if constexpr (tma) {
        if (tid == 0) {
            for (int k = 0; k < NST; ++k) mbar_init(bars + k);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        __syncthreads();
    }
    bool primed1 = false;                                            // the layer's qkv stages issued by the previous layer (MEGA_PRIME)
    uint4 wvA[RA][H / 256];                                          // the next H-long GEMV's first rows, loaded before the barrier
    if (!tma) {
        const bf16* rows[RA];
#pragma unroll
        for (int r = 0; r < RA; ++r) rows[r] = a.layers[0].wqkv + (size_t)min(gwarp * RA + r, QKV - 1) * H;
        load_rows<H, RA>(rows, wvA);
    }

    for (int L = 0; L < a.n_layers; ++L) {
        const Layer lw = a.layers[L];
        const int64_t* bt = a.bt;                                     // one block table for every layer
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
            if (tma) {                                                 // 8 rows a stage, one per warp
                constexpr int RPS = STAGE / (H * 2);
                int lo, hi;
                block_slice(QKV, lo, hi);
                ring_run((hi - lo + RPS - 1) / RPS, bars, ring, ph,
                    [&](int st, unsigned char* dst, unsigned long long* bar) {
                        const int r0 = lo + st * RPS, n = min(RPS, hi - r0);
                        mbar_expect(bar, n * H * 2);
                        bulk_load(dst, lw.wqkv + (size_t)r0 * H, n * H * 2, bar);
                    },
                    [&](int st, unsigned char* src) {
                        const int r = lo + st * RPS + warp;
                        if (warp < RPS && r < hi) {
                            float o[BT];
                            smem_dot<H, BT>(reinterpret_cast<const bf16*>(src) + warp * H, xn, H, o);
                            if (lane == 0) for (int b = 0; b < B; ++b) a.sc.qkv[b * QKV + r] = f2bf(o[b]);
                        }
                    }, primed1);
                primed1 = false;
            } else {
            float out[RA][BT];
            for (int n0 = gwarp * RA; n0 < QKV; n0 += nwarps * RA) {
                if (n0 != gwarp * RA) {
                    const bf16* rows[RA];
#pragma unroll
                    for (int r = 0; r < RA; ++r) rows[r] = lw.wqkv + (size_t)min(n0 + r, QKV - 1) * H;
                    load_rows<H, RA>(rows, wvA);
                }
                dot_rows<H, RA, BT>(wvA, xn, H, out);
                if (lane == 0)
#pragma unroll
                    for (int r = 0; r < RA; ++r) if (n0 + r < QKV) for (int b = 0; b < B; ++b) a.sc.qkv[b * QKV + n0 + r] = f2bf(out[r][b]);
            }
            }
        }
        STAMP(1);
        grid.sync();

        // ---- P2: attention per (row b, kv group g, 256-position chunk c). First the block prepares the group: warp w
        //      norms+ropes+scales q head g*8+w into shared memory (a lane holds elements lane, +32, +64, +96, so the rope
        //      pairs (i, i+64) never leave the lane); in the chunk holding the new token, warp 0 also norms+ropes k and
        //      writes k and v into the paged pool. Then thread t scores position t0 + t against the 8 heads (k/v rows
        //      with every load in flight), chunk softmax stats, and PV with each thread owning 8 consecutive d of one
        //      head over half the positions (16-byte smem loads, 8 FMAs per load).
        {
            float* q_s = reinterpret_cast<float*>(smem);                                   // [GQA][QSTR], qi()
            bf16* v_s = reinterpret_cast<bf16*>(smem + GQA * QSTR * 4);                    // [CHUNK][VSTR]
            float* p_s = reinterpret_cast<float*>(smem + GQA * QSTR * 4 + CHUNK * VSTR * 2);   // [GQA][CHUNK]
            float* red2 = p_s + GQA * CHUNK;                                               // [GQA][WARPS]
            for (int item = blockIdx.x; item < B * NKV * nsplit; item += gridDim.x) {
                const int c = item % nsplit, g = (item / nsplit) % NKV, b = item / (nsplit * NKV);
                const int pos = (int)a.pos[b], kv_len = pos + 1, t0 = c * CHUNK;
                if (t0 >= kv_len) continue;
                STAMP(8);                                             // sub-steps (debug): an item's start
                const int tn = min(CHUNK, kv_len - t0);
                if (warp < GQA) {                                    // warp w preps q head g*8+w
                    float cs[2], sn[2];
#pragma unroll
                    for (int u = 0; u < 2; ++u) { const float f = pos * a.inv_freq[lane + 32 * u]; cs[u] = rbf(cosf(f)); sn[u] = rbf(sinf(f)); }
                    const bf16* qrow = a.sc.qkv + b * QKV + (g * GQA + warp) * D;
                    float x[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) x[u] = bf2f(qrow[lane + 32 * u]);
                    const float r = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / D + EPS);
                    float n[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) n[u] = rbf(bf2f(lw.q_norm[lane + 32 * u]) * rbf(x[u] * r));
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        q_s[qi(warp, lane + 32 * u)] = rbf(n[u] * cs[u] - n[u + 2] * sn[u]) * a.scale;
                        q_s[qi(warp, lane + 32 * u + D / 2)] = rbf(n[u + 2] * cs[u] + n[u] * sn[u]) * a.scale;
                    }
                    if (warp == 0 && c == pos / CHUNK) {                  // this chunk holds the new token: its k and v
                        const bf16* krow = a.sc.qkv + b * QKV + NQ * D + g * D;
                        const bf16* vrow = a.sc.qkv + b * QKV + NQ * D + NKV * D + g * D;
#pragma unroll
                        for (int u = 0; u < 4; ++u) x[u] = bf2f(krow[lane + 32 * u]);
                        const float rk = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / D + EPS);
#pragma unroll
                        for (int u = 0; u < 4; ++u) n[u] = rbf(bf2f(lw.k_norm[lane + 32 * u]) * rbf(x[u] * rk));
                        const int bid = (int)bt[b * a.bt_cols + pos / BS], slot = pos % BS;
                        const size_t off = (((size_t)bid * NKV + g) * BS + slot) * D;
                        bf16 kb[4];
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            kb[u] = f2bf(n[u] * cs[u] - n[u + 2] * sn[u]);
                            kb[u + 2] = f2bf(n[u + 2] * cs[u] + n[u] * sn[u]);
                        }
                        if (MEGA_KV8) {                                        // KV_CACHE_DTYPE=fp8
                            __nv_fp8_e4m3* kdst = reinterpret_cast<__nv_fp8_e4m3*>(lw.pool_k) + off;
                            __nv_fp8_e4m3* vdst = reinterpret_cast<__nv_fp8_e4m3*>(lw.pool_v) + off;
#pragma unroll
                            for (int u = 0; u < 2; ++u) { kdst[lane + 32 * u] = __nv_fp8_e4m3(bf2f(kb[u])); kdst[lane + 32 * u + D / 2] = __nv_fp8_e4m3(bf2f(kb[u + 2])); }
#pragma unroll
                            for (int u = 0; u < 4; ++u) vdst[lane + 32 * u] = __nv_fp8_e4m3(bf2f(vrow[lane + 32 * u]));
                        } else {
                            bf16* kdst = reinterpret_cast<bf16*>(lw.pool_k) + off;
                            bf16* vdst = reinterpret_cast<bf16*>(lw.pool_v) + off;
#pragma unroll
                            for (int u = 0; u < 2; ++u) { kdst[lane + 32 * u] = kb[u]; kdst[lane + 32 * u + D / 2] = kb[u + 2]; }
#pragma unroll
                            for (int u = 0; u < 4; ++u) vdst[lane + 32 * u] = vrow[lane + 32 * u];
                        }
                    }
                }
                __syncthreads();
                STAMP(9);                                             // q normed and roped, the new k/v written
                // Scores: thread (position p, slice q4) dots its QW elements of K with the 8 heads' q, the SPLIT
                // threads of a position (adjacent lanes) then add up; it also copies its slice of V into v_s.
                const int p = tid / SPLIT, q4 = tid % SPLIT, d0q = q4 * QW;
                float s[GQA];
#pragma unroll
                for (int j = 0; j < GQA; ++j) s[j] = 0.f;
                if (p < tn) {
                    const int t = t0 + p;
                    const int bid = (int)bt[b * a.bt_cols + t / BS], slot = t % BS;
                    const size_t off = (((size_t)bid * NKV + g) * BS + slot) * D + d0q;
                    if (MEGA_KV8) {                                            // fp8 rows: widened as they land
                        const uint4* krow = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_fp8_e4m3*>(lw.pool_k) + off);
                        const uint4* vrow = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_fp8_e4m3*>(lw.pool_v) + off);
                        uint4 kv4[QW / 16], vv4[QW / 16];
#pragma unroll
                        for (int i = 0; i < QW / 16; ++i) { kv4[i] = krow[i]; vv4[i] = vrow[i]; }
#pragma unroll
                        for (int i = 0; i < QW / 16; ++i) {
                            float vf[16];
                            fp8_to_f32<16>(&vv4[i], vf);
#pragma unroll
                            for (int jj = 0; jj < 16; ++jj) v_s[p * VSTR + d0q + i * 16 + jj] = f2bf(vf[jj]);
                            float kf[16];
                            fp8_to_f32<16>(&kv4[i], kf);
#pragma unroll
                            for (int j = 0; j < GQA; ++j) {
                                const float* q = q_s + j * QSTR + q4 * (QW + 4) + i * 16;
#pragma unroll
                                for (int jj = 0; jj < 16; ++jj) s[j] += q[jj] * kf[jj];
                            }
                        }
                    } else {
                        const bf16* krow = reinterpret_cast<const bf16*>(lw.pool_k) + off;
                        const bf16* vrow = reinterpret_cast<const bf16*>(lw.pool_v) + off;
                        uint4 kv4[QW / 8];
#pragma unroll
                        for (int i = 0; i < QW / 8; ++i) kv4[i] = *reinterpret_cast<const uint4*>(krow + i * 8);
#pragma unroll
                        for (int i = 0; i < QW / 8; ++i)
                            *reinterpret_cast<uint4*>(v_s + p * VSTR + d0q + i * 8) = *reinterpret_cast<const uint4*>(vrow + i * 8);
#pragma unroll
                        for (int i = 0; i < QW / 8; ++i) {
                            float kf[8];
                            bf16_to_f32<8>(&kv4[i], kf);
#pragma unroll
                            for (int j = 0; j < GQA; ++j) {
                                const float* q = q_s + j * QSTR + q4 * (QW + 4) + i * 8;
#pragma unroll
                                for (int jj = 0; jj < 8; ++jj) s[j] += q[jj] * kf[jj];
                            }
                        }
                    }
                }
#pragma unroll
                for (int j = 0; j < GQA; ++j) {                                // the position's slices add up
#pragma unroll
                    for (int o = 1; o < SPLIT; o <<= 1) s[j] += __shfl_xor_sync(0xffffffff, s[j], o);
                    if (p >= tn) s[j] = -INFINITY;
                }
                __syncthreads();
                STAMP(10);                                            // the chunk's scores
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
                    const float pv = (p < tn && q4 == 0) ? __expf(s[j] - m[j]) : 0.f;   // one thread a position
                    if (q4 == 0 && p < CHUNK) p_s[j * CHUNK + p] = pv;
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
                {   // PV: item = (head j, 8-wide d group), PVS adjacent lanes per item over interleaved positions
                    const int it = tid / PVS, pq = tid % PVS, j = it / (D / 8), d0 = (it % (D / 8)) * 8;
                    float acc[8];
#pragma unroll
                    for (int u = 0; u < 8; ++u) acc[u] = 0.f;
                    for (int t = pq; t < tn; t += PVS) {
                        const float pr = p_s[j * CHUNK + t];
                        const uint4 vv = *reinterpret_cast<const uint4*>(v_s + t * VSTR + d0);
                        float vf[8];
                        bf16_to_f32<8>(&vv, vf);
#pragma unroll
                        for (int u = 0; u < 8; ++u) acc[u] += pr * vf[u];
                    }
#pragma unroll
                    for (int u = 0; u < 8; ++u)
#pragma unroll
                        for (int o = 1; o < PVS; o <<= 1) acc[u] += __shfl_xor_sync(0xffffffff, acc[u], o);
                    if (pq == 0) {
                        const int hq = g * GQA + j;
                        if (nsplit == 1) {
                            bf16* dst = a.sc.attn + b * (NQ * D) + hq * D + d0;
                            const float inv = 1.f / l[j];
#pragma unroll
                            for (int u = 0; u < 8; ++u) dst[u] = f2bf(acc[u] * inv);
                        } else {
                            float* dst = a.sc.part_o + (((size_t)b * NQ + hq) * MAXSPLIT + c) * D + d0;
#pragma unroll
                            for (int u = 0; u < 8; ++u) dst[u] = acc[u];
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
        // o_proj's first stages, issued now when the attention phase left the ring untouched (MEGA_PRIME, chunk <= 64)
        int lo4 = 0, hi4 = 0;
        bool primed4 = false;
        auto issue4 = [&](int st, unsigned char* dst, unsigned long long* bar) {
            constexpr int RPS = STAGE / (NQ * D * 2);
            const int r0 = lo4 + st * RPS, n = min(RPS, hi4 - r0);
            mbar_expect(bar, n * NQ * D * 2);
            bulk_load(dst, lw.wo + (size_t)r0 * (NQ * D), n * NQ * D * 2, bar);
        };
        if constexpr (tma && PRIME_P4) {
            constexpr int RPS = STAGE / (NQ * D * 2);
            block_slice(H, lo4, hi4);
            __syncthreads();                                          // every warp is past its attention item's smem use
            ring_prime((hi4 - lo4 + RPS - 1) / RPS, bars, ring, issue4);
            primed4 = true;
        }
        STAMP(2);

        // ---- P3 (long contexts only): combine the chunks per (row, head) -> attn [B, NQ*D] bf16
        if (nsplit > 1) {
            grid.sync();
            for (int item = blockIdx.x; item < B * NQ; item += gridDim.x) {
                const int hq = item % NQ, b = item / NQ;
                const int n = ((int)a.pos[b] + 1 + CHUNK - 1) / CHUNK;
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
        }
        STAMP(3);
        uint4 wv4[R4][NQ * D / 256];                                  // o_proj's first rows, in flight across the barrier
        if (!tma) {
            const bf16* rows[R4];
#pragma unroll
            for (int r = 0; r < R4; ++r) rows[r] = lw.wo + (size_t)min(gwarp * R4 + r, H - 1) * (NQ * D);
            load_rows<NQ * D, R4>(rows, wv4);
        }
        grid.sync();

        // ---- P4: o_proj GEMV: o[b, n] = attn[b, :] . wo[n, :]
        {
            bf16* xs = reinterpret_cast<bf16*>(smem);                  // [BT][NQ*D]
            for (int i = tid; i < BT * NQ * D / 8; i += THREADS)
                reinterpret_cast<uint4*>(xs)[i] = (i < B * NQ * D / 8) ? reinterpret_cast<const uint4*>(a.sc.attn)[i] : make_uint4(0, 0, 0, 0);
            __syncthreads();
            float out[R4][BT];
            if (tma) {                                                 // 4 rows of 8 KB a stage, two warps a row
                constexpr int RPS = STAGE / (NQ * D * 2);
                float* part = reinterpret_cast<float*>(smem + 16384);   // [WARPS][BT], past xs (16 KB at batch 2)
                if (!primed4) block_slice(H, lo4, hi4);
                const int lo = lo4, hi = hi4;
                ring_run((hi - lo + RPS - 1) / RPS, bars, ring, ph, issue4,
                    [&](int st, unsigned char* src) {
                        const int j = warp >> 1, half = warp & 1, r = lo + st * RPS + j;
                        float o[BT];
                        if (j < RPS && r < hi) {                      // 16 warps: the rest idle (RPS rows a stage)
                            smem_dot<NQ * D / 2, BT>(reinterpret_cast<const bf16*>(src) + j * (NQ * D) + half * (NQ * D / 2),
                                                     xs + half * (NQ * D / 2), NQ * D, o);
                            if (lane == 0) for (int b = 0; b < BT; ++b) part[warp * BT + b] = o[b];
                        }
                        __syncthreads();
                        if (j < RPS && r < hi && half == 0 && lane == 0)
                            for (int b = 0; b < B; ++b) a.sc.o[b * H + r] = f2bf(part[warp * BT + b] + part[(warp + 1) * BT + b]);
                    }, primed4);
            } else
            for (int n0 = gwarp * R4; n0 < H; n0 += nwarps * R4) {
                if (n0 != gwarp * R4) {
                    const bf16* rows[R4];
#pragma unroll
                    for (int r = 0; r < R4; ++r) rows[r] = lw.wo + (size_t)min(n0 + r, H - 1) * (NQ * D);
                    load_rows<NQ * D, R4>(rows, wv4);
                }
                dot_rows<NQ * D, R4, BT>(wv4, xs, NQ * D, out);
                if (lane == 0)
#pragma unroll
                    for (int r = 0; r < R4; ++r) if (n0 + r < H) for (int b = 0; b < B; ++b) a.sc.o[b * H + n0 + r] = f2bf(out[r][b]);
            }
        }
        STAMP(4);
        uint4 wv5[1][H / 256];                                        // this warp's first router row
        {
            const bf16* rows[1] = {lw.wr + (size_t)(min(gwarp, B * E - 1) % E) * H};
            load_rows<H, 1>(rows, wv5);
        }
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
            float out[1][1];
            for (int w = gwarp; w < B * E; w += nwarps) {
                const int b = w / E, e = w % E;
                if (w != gwarp) { const bf16* rows[1] = {lw.wr + (size_t)e * H}; load_rows<H, 1>(rows, wv5); }
                dot_rows<H, 1, 1>(wv5, n2s + b * H, H, out);
                if (lane == 0) a.sc.rlogits[b * E + e] = out[0][0];
            }
        }
        STAMP(5);
        grid.sync();

        // ---- P6: top-8 per row in every block (one warp per row, 4 experts per lane: softmax in fp32, iterated
        //      argmax, lowest index on ties, renormalised weights; block 0 writes them out for P7), then the gate_up GEMV
        //      per (pair, n): act[pair][n] = bf16(silu(bf16(gate)) * bf16(up)).
        bf16* xs6 = reinterpret_cast<bf16*>(smem);                     // n2 rows [BT][H]
        int* ids_s = reinterpret_cast<int*>(xs6 + BT * H);             // [BT*K], read by P7's prefetch too
        float* w_s = reinterpret_cast<float*>(ids_s + BT * K);         // [BT*K]
        const int pairs = B * K;
        {
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
                reinterpret_cast<uint4*>(xs6)[i] = reinterpret_cast<const uint4*>(a.sc.n2)[i];
            __syncthreads();
            STAMP(11);                                                // top-8 done and the rows in shared memory
            constexpr int NPW = FP8 ? 2 : 1;                           // n values per warp item (fp8 rows are half the bytes)
            constexpr int R6 = 2 * NPW;                                // gate and up rows per n
            const int items = pairs * (I / NPW);
            if (tma) {                                                 // C (gate, up) row pairs a stage, within one pair
                constexpr int RB = H * (FP8 ? 1 : 2), C = STAGE / (2 * RB);
                int* seg_s = reinterpret_cast<int*>(smem + 16384);
                float* gu_s = reinterpret_cast<float*>(smem + 20480);  // [2 * C]
                int lo, hi;
                block_slice(pairs * I, lo, hi);
                if (tid == 0) seg_s[1023] = plan_stages(lo, hi, C, I, seg_s, 1000);
                __syncthreads();
                ring_run(seg_s[1023], bars, ring, ph,
                    [&](int st, unsigned char* dst, unsigned long long* bar) {
                        const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / I, n0 = f % I;
                        const size_t gate = ((size_t)ids_s[pp] * (2 * I) + n0) * RB;
                        mbar_expect(bar, 2 * cnt * RB);
                        bulk_load(dst, reinterpret_cast<const char*>(lw.gate_up) + gate, cnt * RB, bar);
                        bulk_load(dst + C * RB, reinterpret_cast<const char*>(lw.gate_up) + gate + (size_t)I * RB, cnt * RB, bar);
                    },
                    [&](int st, unsigned char* src) {
                        const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / I, n0 = f % I, b = pp / K, e = ids_s[pp];
                        for (int ri = warp; ri < 2 * C; ri += WARPS) {
                            const int j = ri % C, up = ri / C;
                            if (j >= cnt) continue;
                            const unsigned char* row = src + (up * C + j) * RB;
                            float v;
                            if constexpr (FP8) {
                                v = smem_dot_fp8<H>(row, xs6 + b * H) * lw.gu_scale[(size_t)e * (2 * I) + up * I + n0 + j];
                            } else {
                                smem_dot<H, 1>(reinterpret_cast<const bf16*>(row), xs6 + b * H, H, &v);
                            }
                            if (lane == 0) gu_s[ri] = v;
                        }
                        __syncthreads();
                        if (tid < cnt) {
                            const float gb = rbf(gu_s[tid]), ub = rbf(gu_s[C + tid]);
                            a.sc.act[(size_t)pp * I + n0 + tid] = f2bf(gb / (1.f + __expf(-gb)) * ub);
                        }
                    });
            } else
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
                warp_dot_rows<H, R6, FP8, 16>(rows, xs6 + b * H, out);
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
        STAMP(6);
        typename DN::V wv7[DN::RD][DN::NV];                           // the down GEMV's first rows (ids known in this block)
        if (!tma) load_down<FP8>(lw.down, ids_s, gwarp * DN::RD, pairs * H, wv7);
        // the down GEMV's stages are planned here (its experts are known) and, with MEGA_PRIME, the first ones issued
        constexpr int RB7 = I * (FP8 ? 1 : 2), C7 = (STAGE / RB7 < (FP8 ? 32 : 16)) ? STAGE / RB7 : (FP8 ? 32 : 16);   // rows a stage, within its bytes
        int* seg7 = reinterpret_cast<int*>(smem + 16384);
        int n7 = 0;
        auto issue7 = [&](int st, unsigned char* dst, unsigned long long* bar) {
            const int f = seg7[st], cnt = seg7[st + 1] - f, pp = f / H, n0 = f % H;
            mbar_expect(bar, cnt * RB7);
            bulk_load(dst, reinterpret_cast<const char*>(lw.down) + ((size_t)ids_s[pp] * H + n0) * RB7, cnt * RB7, bar);
        };
        if (tma) {
            int lo, hi;
            block_slice(pairs * H, lo, hi);
            __syncthreads();                                          // gate_up's last stage consumed by every warp
            if (tid == 0) seg7[1023] = plan_stages(lo, hi, C7, H, seg7, 1000);
            __syncthreads();
            n7 = seg7[1023];
            if (MEGA_PRIME) ring_prime(n7, bars, ring, issue7);
        }
        grid.sync();

        // ---- P7: down GEMV per (pair, n): out2[b][n] += w[pair] * dot(act[pair], down[e][n])
        if (tma) {                                                     // C rows of one pair a stage
            constexpr int RB = RB7;
            int* seg_s = seg7;
            ring_run(n7, bars, ring, ph, issue7,
                [&](int st, unsigned char* src) {
                    const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / H, n0 = f % H, e = ids_s[pp];
                    const bf16* x = a.sc.act + (size_t)pp * I;
                    for (int j = warp; j < cnt; j += WARPS) {
                        float v;
                        if constexpr (FP8) {
                            v = smem_dot_fp8_8<I>(src + j * RB, x) * lw.down_scale[(size_t)e * H + n0 + j];
                        } else {
                            smem_dot<I, 1>(reinterpret_cast<const bf16*>(src + j * RB), x, I, &v);
                        }
                        if (lane == 0) atomicAdd(a.sc.out2 + (pp / K) * H + n0 + j, v * w_s[pp]);
                    }
                }, MEGA_PRIME != 0);
        } else {
            for (int w0 = gwarp * DN::RD; w0 < pairs * H; w0 += nwarps * DN::RD) {
                if (w0 != gwarp * DN::RD) load_down<FP8>(lw.down, a.sc.ids, w0, pairs * H, wv7);
                float out[DN::RD];
#pragma unroll
                for (int r = 0; r < DN::RD; ++r) {
                    const int w = min(w0 + r, pairs * H - 1), pp = w / H;
                    const bf16* x = a.sc.act + (size_t)pp * I;
                    float acc = 0.f;
#pragma unroll
                    for (int i = 0; i < DN::NV; ++i) {
                        float xf[DN::EPL], wf[DN::EPL];
                        const uint4 xv = *reinterpret_cast<const uint4*>(x + i * DN::CH + lane * DN::EPL);
                        bf16_to_f32<8>(&xv, xf);
                        if constexpr (FP8) fp8_to_f32<DN::EPL>(&wv7[r][i], wf); else bf16_to_f32<DN::EPL>(&wv7[r][i], wf);
#pragma unroll
                        for (int j = 0; j < DN::EPL; ++j) acc += wf[j] * xf[j];
                    }
                    out[r] = warp_sum(acc);
                }
                if (lane == 0) {
#pragma unroll
                    for (int r = 0; r < DN::RD; ++r) if (w0 + r < pairs * H) {
                        const int w = w0 + r, pp = w / H, nn = w % H;
                        float v = out[r];
                        if (FP8) v *= lw.down_scale[(size_t)a.sc.ids[pp] * H + nn];
                        atomicAdd(a.sc.out2 + (pp / K) * H + nn, v * a.sc.w[pp]);
                    }
                }
            }
        }
        STAMP(7);
        {   // the next layer's first qkv rows, or the lm_head's, in flight across the barrier
            const bool last = L + 1 == a.n_layers;
            if (last && blockIdx.x == 0 && tid < B) a.keys[tid] = 0ull;
            const bf16* base = last ? a.lm_head : a.layers[L + 1].wqkv;
            const int nrows = last ? a.V : QKV;
            const bf16* rows[RA];
#pragma unroll
            for (int r = 0; r < RA; ++r) rows[r] = base + (size_t)min(gwarp * RA + r, nrows - 1) * H;
            if (!tma) load_rows<H, RA>(rows, wvA);
            if (tma && MEGA_PRIME && !last) {                         // the next layer's qkv stages: its rows need no barrier
                constexpr int RPS = STAGE / (H * 2);
                int lo, hi;
                block_slice(QKV, lo, hi);
                const bf16* wq = a.layers[L + 1].wqkv;
                __syncthreads();                                      // down's last stage consumed by every warp
                ring_prime((hi - lo + RPS - 1) / RPS, bars, ring, [&](int st, unsigned char* dst, unsigned long long* bar) {
                    const int r0 = lo + st * RPS, n = min(RPS, hi - r0);
                    mbar_expect(bar, n * H * 2);
                    bulk_load(dst, wq + (size_t)r0 * H, n * H * 2, bar);
                });
                primed1 = true;
            }
        }
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
        float out[RA][BT];
        unsigned long long best[BT];                                   // packed (ordered bf16 value, ~index): max = argmax, lowest index on ties
#pragma unroll
        for (int b = 0; b < BT; ++b) best[b] = 0ull;
        if (tma) {
            constexpr int RPS = STAGE / (H * 2);
            int lo, hi;
            block_slice(a.V, lo, hi);
            ring_run((hi - lo + RPS - 1) / RPS, bars, ring, ph,
                [&](int st, unsigned char* dst, unsigned long long* bar) {
                    const int r0 = lo + st * RPS, n = min(RPS, hi - r0);
                    mbar_expect(bar, n * H * 2);
                    bulk_load(dst, a.lm_head + (size_t)r0 * H, n * H * 2, bar);
                },
                [&](int st, unsigned char* src) {
                    const int r = lo + st * RPS + warp;
                    if (warp < RPS && r < hi) {
                        float o[BT];
                        smem_dot<H, BT>(reinterpret_cast<const bf16*>(src) + warp * H, xn, H, o);
                        if (lane == 0)
                            for (int b = 0; b < B; ++b) {
                                const bf16 v = f2bf(o[b]);
                                a.logits[(size_t)b * a.V + r] = v;
                                const unsigned short bits = __bfloat16_as_ushort(v);
                                const unsigned int ordered = (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
                                best[b] = max(best[b], ((unsigned long long)ordered << 32) | (0xFFFFFFFFu - (unsigned)r));
                            }
                    }
                });
        } else
        for (int n0 = gwarp * RA; n0 < a.V; n0 += nwarps * RA) {
            if (n0 != gwarp * RA) {
                const bf16* rows[RA];
#pragma unroll
                for (int r = 0; r < RA; ++r) rows[r] = a.lm_head + (size_t)min(n0 + r, a.V - 1) * H;
                load_rows<H, RA>(rows, wvA);
            }
            dot_rows<H, RA, BT>(wvA, xn, H, out);
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < RA; ++r) if (n0 + r < a.V) {
                    for (int b = 0; b < B; ++b) {
                        const bf16 v = f2bf(out[r][b]);
                        a.logits[(size_t)b * a.V + n0 + r] = v;
                        const unsigned short bits = __bfloat16_as_ushort(v);
                        const unsigned int ordered = (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
                        const unsigned long long key = ((unsigned long long)ordered << 32) | (0xFFFFFFFFu - (unsigned)(n0 + r));
                        best[b] = max(best[b], key);
                    }
                }
            }
        }
        if (lane == 0) for (int b = 0; b < B; ++b) atomicMax(a.keys + b, best[b]);
    }
    grid.sync();
    if (blockIdx.x == 0 && tid < B) a.tokens_out[tid] = (int64_t)(0xFFFFFFFFu - (unsigned)(a.keys[tid] & 0xFFFFFFFFull));
    STAMP_ALWAYS(13);
}

// ------------------------------------------------------------------------------------------------ host side
constexpr size_t SMEM_BYTES = HDR + (size_t)BMAX * NQ * D * 2;   // the largest phase at BMAX: attn rows for o_proj (128 KB)
static_assert(ATTN_SMEM <= SMEM_BYTES, "attention smem");
static_assert(HDR + 2 * BMAX * H * 2 <= SMEM_BYTES, "P5 smem");
static_assert(RING_OFF + NST * STAGE <= 232448 && HDR + 2 * 4096 * 2 + 4096 <= RING_OFF, "TMA ring vs the batch-2 phases and the 227 KB a block may have");
static_assert(HDR + BMAX * H * 2 + BMAX * K * 8 <= SMEM_BYTES, "P6 smem");
static_assert(128 + NST * 8 <= HDR && WARPS * 4 <= 128, "smem header holds red[WARPS] and the mbarriers");

template <bool FP8> static void* pick(int B) {
    return B <= 1 ? (void*)decode_step_kernel<1, FP8> : B <= 2 ? (void*)decode_step_kernel<2, FP8> : B <= 4 ? (void*)decode_step_kernel<4, FP8>
         : B <= 8 ? (void*)decode_step_kernel<8, FP8> : (void*)decode_step_kernel<16, FP8>;
}

std::tuple<torch::Tensor, torch::Tensor> decode_step(torch::Tensor tokens, torch::Tensor pos, torch::Tensor bt, torch::Tensor layers, torch::Tensor embed,
                          torch::Tensor final_norm, torch::Tensor lm_head, torch::Tensor inv_freq, torch::Tensor scratch_f32,
                          torch::Tensor scratch_bf16, torch::Tensor scratch_i32, double scale, bool fp8, torch::Tensor times,
                          int64_t stamp_layer, bool kv8, bool tma) {
    TORCH_CHECK(tokens.is_cuda() && tokens.dtype() == torch::kInt64 && tokens.dim() == 1 && tokens.is_contiguous(), "tokens [B] int64");
    const int B = tokens.size(0);
    TORCH_CHECK(B >= 1 && B <= BMAX, "batch 1..16");
    TORCH_CHECK(pos.dtype() == torch::kInt64 && pos.numel() == B && pos.is_contiguous(), "pos [B] int64");
    TORCH_CHECK(bt.dtype() == torch::kInt64 && bt.dim() == 2 && bt.size(0) == B && bt.is_contiguous(), "bt [B, cols] int64");
    TORCH_CHECK(layers.dtype() == torch::kInt64 && layers.dim() == 2 && layers.size(1) == 13 && layers.is_cuda() && layers.is_contiguous(), "layers [L, 13] int64");
    TORCH_CHECK(bt.size(1) * BS <= MAXSPLIT * CHUNK, "context longer than MAXSPLIT * CHUNK");   // no host sync here
    TORCH_CHECK(embed.dtype() == torch::kBFloat16 && embed.size(1) == H && lm_head.dtype() == torch::kBFloat16 && lm_head.size(1) == H, "embed/lm_head [V, H] bf16");
    const int V = lm_head.size(0);
    auto logits = torch::empty({B, V}, tokens.options().dtype(torch::kBFloat16));
    auto next = torch::empty({B}, tokens.options());
    Args a;
    a.tokens = tokens.data_ptr<int64_t>(); a.embed = reinterpret_cast<const bf16*>(embed.data_ptr());
    a.pos = pos.data_ptr<int64_t>(); a.bt = bt.data_ptr<int64_t>(); a.bt_cols = bt.size(1);
    a.keys = reinterpret_cast<unsigned long long*>(scratch_i32.data_ptr<int>() + BMAX * K); a.tokens_out = next.data_ptr<int64_t>();
    a.layers = reinterpret_cast<const Layer*>(layers.data_ptr<int64_t>()); a.n_layers = layers.size(0);
    a.final_norm = reinterpret_cast<const bf16*>(final_norm.data_ptr()); a.lm_head = reinterpret_cast<const bf16*>(lm_head.data_ptr());
    a.V = V; a.logits = reinterpret_cast<bf16*>(logits.data_ptr()); a.B = B; a.scale = (float)scale; a.inv_freq = inv_freq.data_ptr<float>();
    a.sc = carve_scratch(scratch_f32.data_ptr<float>(), reinterpret_cast<bf16*>(scratch_bf16.data_ptr()), scratch_i32.data_ptr<int>());
    static bool attr_set = false;
    if (!attr_set) {
        for (int b : {1, 2, 4, 8, 16}) {
            cudaFuncSetAttribute(pick<true>(b), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(b));
            cudaFuncSetAttribute(pick<false>(b), cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_for(b));
        }
        attr_set = true;
    }
    const size_t smem = smem_for(B);
    int smem_per_sm = 0;
    cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    const int bps = (MEGA_BPS > 1 && (int)(MEGA_BPS * (smem + 1024)) <= smem_per_sm) ? MEGA_BPS : 1;   // 1 KB a block is the driver's
    a.times = times.numel() ? reinterpret_cast<unsigned long long*>(times.data_ptr<int64_t>()) : nullptr;
    a.times_blocks = times.numel() >= 16 * num_sms() * MEGA_BPS ? 1 : 0;
    a.stamp_layer = (int)stamp_layer;
    TORCH_CHECK(kv8 == (MEGA_KV8 != 0) && tma == (MEGA_TMA != 0), "this build of the kernel is for kv8=", MEGA_KV8, " tma=", MEGA_TMA);
    a.kv8 = kv8 ? 1 : 0;
    a.tma = tma ? 1 : 0;
    void* args[] = {&a};
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaLaunchCooperativeKernel(fp8 ? pick<true>(B) : pick<false>(B), dim3(num_sms() * bps), dim3(THREADS), args, smem, stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
    return {logits, next};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_step", &decode_step);
    m.def("scratch_f32_elems", &scratch_f32_elems);
    m.def("scratch_bf16_elems", &scratch_bf16_elems);
    m.def("scratch_i32_elems", &scratch_i32_elems);
}
