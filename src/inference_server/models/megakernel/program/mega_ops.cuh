// The default program's instructions for the interpreter (interp.cu): today's megakernel phases (decode_step.cu) as
// instructions over a range of rows or items, with the same arithmetic in the same order. Opcodes and args match
// program/__init__.py. Each op calls wait() once, from code every thread runs, before it reads anything another
// instruction wrote (an op with nothing to do may skip it); what comes before (weight rows into registers, ring stages
// issued) overlaps the wait.
#pragma once

enum { OP_QKV = 1, OP_ATTN, OP_COMBINE, OP_OPROJ, OP_POST, OP_TOPK, OP_GATEUP, OP_DOWN, OP_LMHEAD, OP_ARGMAX };

template <int BT> constexpr size_t program_smem() { return smem_bytes<BT>(); }   // decode_step.cu's phases, the same buffers

// xn [BT][H] in shared memory = RMSNorm(the residual) per row, rows past B zero; the residual is the embedding (layer
// 0) or h1 + out2, and row `hrow` of it is also written to h.
template <int BT>
__device__ __forceinline__ void residual_norm(const Ctx& c, bf16* xn, bool embed, const bf16* w, int hrow) {
    const Args& a = c.a;
    const int tid = threadIdx.x;
    for (int b = 0; b < BT; ++b) {
        if (b < a.B) {
            if (embed) {
                const bf16* src = a.embed + (size_t)a.tokens[b] * H;
                for (int k = tid; k < H; k += THREADS) xn[b * H + k] = src[k];
            } else {
                for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(bf2f(a.sc.h1[b * H + k]) + a.sc.out2[b * H + k]);
            }
            if (hrow == b) for (int k = tid; k < H; k += THREADS) a.sc.h[b * H + k] = xn[b * H + k];
            __syncthreads();
            rmsnorm_row(xn + b * H, w, xn + b * H, H, c.red);
        } else {
            for (int k = tid; k < H; k += THREADS) xn[b * H + k] = f2bf(0.f);
        }
    }
    __syncthreads();
}

// The argmax key of a bf16 logit at vocab row r: (ordered value, ~r), so the max is the argmax with the lowest index on ties.
__device__ __forceinline__ unsigned long long argmax_key(bf16 v, int r) {
    const unsigned short bits = __bfloat16_as_ushort(v);
    const unsigned int ordered = (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
    return ((unsigned long long)ordered << 32) | (0xFFFFFFFFu - (unsigned)r);
}

// A GEMV over rows [lo, hi) of a [rows, H] bf16 matrix against xn [BT][H] in shared memory (built by `prep` once the
// wait is over); emit(r, out[BT]) on lane 0 of the warp that owns row r. The QKV and LMHEAD shape: through the ring at
// batch <= 2, else RA rows per warp in registers, the first ones loaded before the wait.
template <int BT, typename W, typename Prep, typename Emit>
__device__ __forceinline__ void gemv_h(Ctx& c, const bf16* wm, int lo, int hi, W& wait, Prep prep, Emit emit) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bf16* xn = reinterpret_cast<bf16*>(c.smem);
    if constexpr (MEGA_TMA && BT <= 2) {                               // 8 rows a stage, one per warp
        constexpr int RPS = STAGE / (H * 2);
        const int nst = (hi - lo + RPS - 1) / RPS;
        auto issue = [&](int st, unsigned char* dst, unsigned long long* bar) {
            const int r0 = lo + st * RPS, n = min(RPS, hi - r0);
            mbar_expect(bar, n * H * 2);
            bulk_load(dst, wm + (size_t)r0 * H, n * H * 2, bar);
        };
        if (MEGA_PRIME) ring_prime(nst, c.bars, c.ring, issue);
        wait();
        prep(xn);
        ring_run(nst, c.bars, c.ring, c.ph, issue, [&](int st, unsigned char* src) {
            const int r = lo + st * RPS + warp;
            if (warp < RPS && r < hi) {
                float o[BT];
                smem_dot<H, BT>(reinterpret_cast<const bf16*>(src) + warp * H, xn, H, o);
                if (lane == 0) emit(r, o);
            }
        }, MEGA_PRIME != 0);
    } else {
        constexpr int RA = (BT <= 4) ? 2 : 1;
        uint4 wv[RA][H / 256];
        const int first = lo + warp * RA;
        const bf16* rows[RA];
        if (first < hi) {
#pragma unroll
            for (int r = 0; r < RA; ++r) rows[r] = wm + (size_t)min(first + r, hi - 1) * H;
            load_rows<H, RA>(rows, wv);
        }
        wait();
        prep(xn);
        float out[RA][BT];
        for (int n0 = first; n0 < hi; n0 += WARPS * RA) {
            if (n0 != first) {
#pragma unroll
                for (int r = 0; r < RA; ++r) rows[r] = wm + (size_t)min(n0 + r, hi - 1) * H;
                load_rows<H, RA>(rows, wv);
            }
            dot_rows<H, RA, BT>(wv, xn, H, out);
            if (lane == 0)
#pragma unroll
                for (int r = 0; r < RA; ++r) if (n0 + r < hi) emit(n0 + r, out[r]);
        }
    }
}

// QKV: residual and input norm of every row, then rows [arg0, arg1) of wqkv; arg2: the row of h this block writes (-1 none).
template <int BT, typename W>
__device__ __forceinline__ void op_qkv(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const Layer lw = a.layers[in.layer];
    gemv_h<BT>(c, lw.wqkv, in.arg[0], in.arg[1], wait,
               [&](bf16* xn) { residual_norm<BT>(c, xn, in.layer == 0, lw.in_norm, in.arg[2]); },
               [&](int r, const float* o) { for (int b = 0; b < a.B; ++b) a.sc.qkv[b * QKV + r] = f2bf(o[b]); });
}

// LMHEAD: final norm of h1 + out2, rows [arg0, arg1) of lm_head into the logits, each warp's best key per row into keys.
template <int BT, typename W>
__device__ __forceinline__ void op_lmhead(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    unsigned long long best[BT];
#pragma unroll
    for (int b = 0; b < BT; ++b) best[b] = 0ull;
    gemv_h<BT>(c, a.lm_head, in.arg[0], in.arg[1], wait,
               [&](bf16* xn) { residual_norm<BT>(c, xn, false, a.final_norm, -1); },
               [&](int r, const float* o) {
                   for (int b = 0; b < a.B; ++b) {
                       const bf16 v = f2bf(o[b]);
                       a.logits[(size_t)b * a.V + r] = v;
                       best[b] = max(best[b], argmax_key(v, r));
                   }
               });
    if ((threadIdx.x & 31) == 0) for (int b = 0; b < a.B; ++b) atomicMax(a.keys + b, best[b]);
}

// ARGMAX: the greedy token of every row from its key, and the keys zeroed for the next step.
template <typename W>
__device__ __forceinline__ void op_argmax(Ctx& c, const Instr&, W& wait) {
    const Args& a = c.a;
    wait();
    if (threadIdx.x < a.B) {
        a.tokens_out[threadIdx.x] = (int64_t)(0xFFFFFFFFu - (unsigned)(a.keys[threadIdx.x] & 0xFFFFFFFFull));
        a.keys[threadIdx.x] = 0ull;
    }
}

// ATTN: attention items arg0, arg0 + arg1, ... of (row b, kv group g, chunk c): the block norms+ropes+scales the group's
// q heads, the chunk holding the new token norms+ropes its k and writes k and v into the paged pool, then scores,
// the chunk's softmax and PV; one chunk -> attn, else partial o/m/l for COMBINE (decode_step.cu's P2).
template <int BT, typename W>
__device__ __forceinline__ void op_attn(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const Layer lw = a.layers[in.layer];
    const int64_t* bt = a.bt;
    const int B = a.B, nsplit = c.nsplit, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned char* smem = c.smem;
    wait();
    float* q_s = reinterpret_cast<float*>(smem);                                   // [GQA][QSTR], qi()
    bf16* v_s = reinterpret_cast<bf16*>(smem + GQA * QSTR * 4);                    // [CHUNK][VSTR]
    float* p_s = reinterpret_cast<float*>(smem + GQA * QSTR * 4 + CHUNK * VSTR * 2);   // [GQA][CHUNK]
    float* red2 = p_s + GQA * CHUNK;                                               // [GQA][WARPS]
    for (int item = in.arg[0]; item < B * NKV * nsplit; item += in.arg[1]) {
        const int c_ = item % nsplit, g = (item / nsplit) % NKV, b = item / (nsplit * NKV);
        const int pos = (int)a.pos[b], kv_len = pos + 1, t0 = c_ * CHUNK;
        if (t0 >= kv_len) continue;
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
            if (warp == 0 && c_ == pos / CHUNK) {                  // this chunk holds the new token: its k and v
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
        // Scores: thread (position p, slice q4) dots its QW elements of K with the 8 heads' q, the SPLIT threads of a
        // position (adjacent lanes) then add up; it also copies its slice of V into v_s.
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
                    float* dst = a.sc.part_o + (((size_t)b * NQ + hq) * MAXSPLIT + c_) * D + d0;
#pragma unroll
                    for (int u = 0; u < 8; ++u) dst[u] = acc[u];
                }
            }
        }
        if (nsplit > 1 && tid < GQA) {
            const int hq = g * GQA + tid;
            a.sc.part_m[((size_t)b * NQ + hq) * MAXSPLIT + c_] = m[tid];
            a.sc.part_l[((size_t)b * NQ + hq) * MAXSPLIT + c_] = l[tid];
        }
        __syncthreads();
    }
}

// COMBINE: (row, head) items arg0, arg0 + arg1, ... of rows spanning several chunks -> attn; nothing (and no wait)
// when every row fits one chunk (decode_step.cu's P3).
template <typename W>
__device__ __forceinline__ void op_combine(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    if (c.nsplit <= 1) return;
    wait();
    const int tid = threadIdx.x;
    for (int item = in.arg[0]; item < a.B * NQ; item += in.arg[1]) {
        const int hq = item % NQ, b = item / NQ;
        const int n = ((int)a.pos[b] + 1 + CHUNK - 1) / CHUNK;
        const float* pm = a.sc.part_m + ((size_t)b * NQ + hq) * MAXSPLIT;
        const float* pl = a.sc.part_l + ((size_t)b * NQ + hq) * MAXSPLIT;
        float M = -INFINITY;
        for (int c_ = 0; c_ < n; ++c_) M = fmaxf(M, pm[c_]);
        float Lsum = 0.f;
        for (int c_ = 0; c_ < n; ++c_) Lsum += pl[c_] * __expf(pm[c_] - M);
        if (tid < D) {
            float acc = 0.f;
            for (int c_ = 0; c_ < n; ++c_)
                acc += a.sc.part_o[(((size_t)b * NQ + hq) * MAXSPLIT + c_) * D + tid] * __expf(pm[c_] - M);
            a.sc.attn[b * (NQ * D) + hq * D + tid] = f2bf(acc / Lsum);
        }
    }
}

// OPROJ: rows [arg0, arg1) of wo: o[b, n] = attn[b, :] . wo[n, :] (decode_step.cu's P4).
template <int BT, typename W>
__device__ __forceinline__ void op_oproj(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const Layer lw = a.layers[in.layer];
    const int lo = in.arg[0], hi = in.arg[1], B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    bf16* xs = reinterpret_cast<bf16*>(c.smem);                       // [BT][NQ*D]
    auto load_x = [&] {
        for (int i = tid; i < BT * NQ * D / 8; i += THREADS)
            reinterpret_cast<uint4*>(xs)[i] = (i < B * NQ * D / 8) ? reinterpret_cast<const uint4*>(a.sc.attn)[i] : make_uint4(0, 0, 0, 0);
        __syncthreads();
    };
    if constexpr (MEGA_TMA && BT <= 2) {                               // 4 rows of 8 KB a stage, two warps a row
        constexpr int RPS = STAGE / (NQ * D * 2);
        const int nst = (hi - lo + RPS - 1) / RPS;
        auto issue = [&](int st, unsigned char* dst, unsigned long long* bar) {
            const int r0 = lo + st * RPS, n = min(RPS, hi - r0);
            mbar_expect(bar, n * NQ * D * 2);
            bulk_load(dst, lw.wo + (size_t)r0 * (NQ * D), n * NQ * D * 2, bar);
        };
        if (MEGA_PRIME) ring_prime(nst, c.bars, c.ring, issue);
        wait();
        load_x();
        float* part = reinterpret_cast<float*>(c.smem + 16384);       // [WARPS][BT], past xs (16 KB at batch 2)
        ring_run(nst, c.bars, c.ring, c.ph, issue, [&](int st, unsigned char* src) {
            const int j = warp >> 1, half = warp & 1, r = lo + st * RPS + j;
            float o[BT];
            if (j < RPS && r < hi) {                                  // 16 warps: the rest idle (RPS rows a stage)
                smem_dot<NQ * D / 2, BT>(reinterpret_cast<const bf16*>(src) + j * (NQ * D) + half * (NQ * D / 2),
                                         xs + half * (NQ * D / 2), NQ * D, o);
                if (lane == 0) for (int b = 0; b < BT; ++b) part[warp * BT + b] = o[b];
            }
            __syncthreads();
            if (j < RPS && r < hi && half == 0 && lane == 0)
                for (int b = 0; b < B; ++b) a.sc.o[b * H + r] = f2bf(part[warp * BT + b] + part[(warp + 1) * BT + b]);
        }, MEGA_PRIME != 0);
    } else {
        constexpr int R4 = (BT <= 2) ? 2 : 1;
        uint4 wv[R4][NQ * D / 256];
        const int first = lo + warp * R4;
        const bf16* rows[R4];
        if (first < hi) {
#pragma unroll
            for (int r = 0; r < R4; ++r) rows[r] = lw.wo + (size_t)min(first + r, hi - 1) * (NQ * D);
            load_rows<NQ * D, R4>(rows, wv);
        }
        wait();
        load_x();
        float out[R4][BT];
        for (int n0 = first; n0 < hi; n0 += WARPS * R4) {
            if (n0 != first) {
#pragma unroll
                for (int r = 0; r < R4; ++r) rows[r] = lw.wo + (size_t)min(n0 + r, hi - 1) * (NQ * D);
                load_rows<NQ * D, R4>(rows, wv);
            }
            dot_rows<NQ * D, R4, BT>(wv, xs, NQ * D, out);
            if (lane == 0)
#pragma unroll
                for (int r = 0; r < R4; ++r) if (n0 + r < hi) for (int b = 0; b < B; ++b) a.sc.o[b * H + n0 + r] = f2bf(out[r][b]);
        }
    }
}

// POST: h1 = bf16(h + o) and n2 = RMSNorm(h1) of every row in shared memory, row arg0 of them written out with out2
// zeroed (-1 none), then router logits for (row, expert) items [arg1, arg2), one a warp (decode_step.cu's P5).
template <int BT, typename W>
__device__ __forceinline__ void op_post(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const Layer lw = a.layers[in.layer];
    const int hrow = in.arg[0], lo = in.arg[1], hi = in.arg[2], tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint4 wv[1][H / 256];
    const int first = lo + warp;
    if (first < hi) { const bf16* rows[1] = {lw.wr + (size_t)(first % E) * H}; load_rows<H, 1>(rows, wv); }
    wait();
    bf16* n2s = reinterpret_cast<bf16*>(c.smem);                      // [BT][H]
    bf16* h1s = n2s + BT * H;                                          // [BT][H]
    for (int b = 0; b < a.B; ++b) {
        for (int k = tid; k < H; k += THREADS) h1s[b * H + k] = f2bf(bf2f(a.sc.h[b * H + k]) + bf2f(a.sc.o[b * H + k]));
        __syncthreads();
        rmsnorm_row(h1s + b * H, lw.post_norm, n2s + b * H, H, c.red);
        if (hrow == b) {
            for (int k = tid; k < H; k += THREADS) { a.sc.h1[b * H + k] = h1s[b * H + k]; a.sc.n2[b * H + k] = n2s[b * H + k]; a.sc.out2[b * H + k] = 0.f; }
        }
    }
    __syncthreads();
    float out[1][1];
    for (int w = first; w < hi; w += WARPS) {
        const int b = w / E, e = w % E;
        if (w != first) { const bf16* rows[1] = {lw.wr + (size_t)e * H}; load_rows<H, 1>(rows, wv); }
        dot_rows<H, 1, 1>(wv, n2s + b * H, H, out);
        if (lane == 0) a.sc.rlogits[b * E + e] = out[0][0];
    }
}

// TOPK, the router: softmax and top-8 of every row's logits (one warp a row: fp32, iterated argmax, lowest index on
// ties, renormalised weights), then the experts of this block's expert instructions written into them: the arg1
// instructions from arg0 past this one, each holding its first pair p0 in arg[2], get arg[3..6] = the experts of
// pairs p0 and p0 + 1 and their weights (float bits). arg2 = 1: also ids and w to scratch.
template <int BT, typename W>
__device__ __forceinline__ void op_topk(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const int B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wait();
    int* ids_s = reinterpret_cast<int*>(c.smem);                      // [BT*K]
    float* w_s = reinterpret_cast<float*>(ids_s + BT * K);            // [BT*K]
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
            if (in.arg[2]) { a.sc.ids[b * K + lane] = ids_s[b * K + lane]; a.sc.w[b * K + lane] = w_s[b * K + lane]; }
        }
    }
    __syncthreads();
    const int last = B * K - 1;
    for (int s = tid; s < in.arg[1]; s += THREADS) {
        Instr* slot = a.instrs + c.at + in.arg[0] + s;
        const int p0 = slot->arg[2], p1 = min(p0 + 1, last);
        slot->arg[3] = ids_s[p0]; slot->arg[4] = ids_s[p1];
        slot->arg[5] = __float_as_int(w_s[p0]); slot->arg[6] = __float_as_int(w_s[p1]);
    }
}

// GATEUP (+ silu_mul): flattened rows [arg0, arg1) of (pair, n), pairs arg2 and arg2 + 1 with experts arg3, arg4:
// act[pair][n] = bf16(silu(bf16(gate)) * bf16(up)) (decode_step.cu's P6 after the top-8).
template <int BT, bool FP8, typename W>
__device__ __forceinline__ void op_gateup(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const Layer lw = a.layers[in.layer];
    const int lo = in.arg[0], hi = in.arg[1], p0 = in.arg[2], e0 = in.arg[3], e1 = in.arg[4];
    const int B = a.B, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    auto expert = [&](int pp) { return pp == p0 ? e0 : e1; };
    bf16* xs6 = reinterpret_cast<bf16*>(c.smem);                       // n2 rows [BT][H]
    auto load_x = [&] {
        for (int i = tid; i < B * H / 8; i += THREADS) reinterpret_cast<uint4*>(xs6)[i] = reinterpret_cast<const uint4*>(a.sc.n2)[i];
        __syncthreads();
    };
    if constexpr (MEGA_TMA && BT <= 2) {                               // C (gate, up) row pairs a stage, within one pair
        constexpr int RB = H * (FP8 ? 1 : 2), C = STAGE / (2 * RB);
        int* seg_s = reinterpret_cast<int*>(c.smem + 16384);
        float* gu_s = reinterpret_cast<float*>(c.smem + 20480);       // [2 * C]
        if (tid == 0) seg_s[1023] = plan_stages(lo, hi, C, I, seg_s, 1000);
        __syncthreads();
        const int nst = seg_s[1023];
        auto issue = [&](int st, unsigned char* dst, unsigned long long* bar) {
            const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / I, n0 = f % I;
            const size_t gate = ((size_t)expert(pp) * (2 * I) + n0) * RB;
            mbar_expect(bar, 2 * cnt * RB);
            bulk_load(dst, reinterpret_cast<const char*>(lw.gate_up) + gate, cnt * RB, bar);
            bulk_load(dst + C * RB, reinterpret_cast<const char*>(lw.gate_up) + gate + (size_t)I * RB, cnt * RB, bar);
        };
        if (MEGA_PRIME) ring_prime(nst, c.bars, c.ring, issue);
        wait();
        load_x();
        ring_run(nst, c.bars, c.ring, c.ph, issue, [&](int st, unsigned char* src) {
            const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / I, n0 = f % I, b = pp / K, e = expert(pp);
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
        }, MEGA_PRIME != 0);
    } else {
        constexpr int NPW = FP8 ? 2 : 1;                               // n values per warp item (fp8 rows are half the bytes)
        constexpr int R6 = 2 * NPW;                                    // gate and up rows per n
        wait();
        load_x();
        for (int w = lo / NPW + warp; w < hi / NPW; w += WARPS) {
            const int p = w / (I / NPW), n = (w % (I / NPW)) * NPW, b = p / K, e = expert(p);
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
}

// DOWN: flattened rows [arg0, arg1) of (pair, n), pairs arg2 and arg2 + 1 with experts arg3, arg4 and weights arg5,
// arg6 (float bits): out2[b][n] += w[pair] * dot(act[pair], down[e][n]), fp32 atomics (decode_step.cu's P7).
template <int BT, bool FP8, typename W>
__device__ __forceinline__ void op_down(Ctx& c, const Instr& in, W& wait) {
    const Args& a = c.a;
    const Layer lw = a.layers[in.layer];
    const int lo = in.arg[0], hi = in.arg[1], p0 = in.arg[2], e0 = in.arg[3], e1 = in.arg[4];
    const float w0 = __int_as_float(in.arg[5]), w1 = __int_as_float(in.arg[6]);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    auto expert = [&](int pp) { return pp == p0 ? e0 : e1; };
    auto weight = [&](int pp) { return pp == p0 ? w0 : w1; };
    if constexpr (MEGA_TMA && BT <= 2) {                               // C rows of one pair a stage
        constexpr int RB = I * (FP8 ? 1 : 2), C = (STAGE / RB < (FP8 ? 32 : 16)) ? STAGE / RB : (FP8 ? 32 : 16);
        int* seg_s = reinterpret_cast<int*>(c.smem + 16384);
        if (tid == 0) seg_s[1023] = plan_stages(lo, hi, C, H, seg_s, 1000);
        __syncthreads();
        const int nst = seg_s[1023];
        auto issue = [&](int st, unsigned char* dst, unsigned long long* bar) {
            const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / H, n0 = f % H;
            mbar_expect(bar, cnt * RB);
            bulk_load(dst, reinterpret_cast<const char*>(lw.down) + ((size_t)expert(pp) * H + n0) * RB, cnt * RB, bar);
        };
        if (MEGA_PRIME) ring_prime(nst, c.bars, c.ring, issue);
        wait();
        ring_run(nst, c.bars, c.ring, c.ph, issue, [&](int st, unsigned char* src) {
            const int f = seg_s[st], cnt = seg_s[st + 1] - f, pp = f / H, n0 = f % H, e = expert(pp);
            const bf16* x = a.sc.act + (size_t)pp * I;
            for (int j = warp; j < cnt; j += WARPS) {
                float v;
                if constexpr (FP8) {
                    v = smem_dot_fp8_8<I>(src + j * RB, x) * lw.down_scale[(size_t)e * H + n0 + j];
                } else {
                    smem_dot<I, 1>(reinterpret_cast<const bf16*>(src + j * RB), x, I, &v);
                }
                if (lane == 0) atomicAdd(a.sc.out2 + (pp / K) * H + n0 + j, v * weight(pp));
            }
        }, MEGA_PRIME != 0);
    } else {
        using DN = Down<FP8>;
        typename DN::V wv[DN::RD][DN::NV];
        auto load = [&](int w0r) {                                    // decode_step.cu's load_down, experts from the args
#pragma unroll
            for (int r = 0; r < DN::RD; ++r) {
                const int w = min(w0r + r, hi - 1), pp = w / H, nn = w % H;
                const char* row = reinterpret_cast<const char*>(lw.down) + ((size_t)expert(pp) * H + nn) * I * (FP8 ? 1 : 2);
#pragma unroll
                for (int i = 0; i < DN::NV; ++i)
                    wv[r][i] = *reinterpret_cast<const typename DN::V*>(row + ((size_t)i * DN::CH + lane * DN::EPL) * (FP8 ? 1 : 2));
            }
        };
        const int first = lo + warp * DN::RD;
        if (first < hi) load(first);
        wait();
        for (int w0r = first; w0r < hi; w0r += WARPS * DN::RD) {
            if (w0r != first) load(w0r);
            float out[DN::RD];
#pragma unroll
            for (int r = 0; r < DN::RD; ++r) {
                const int w = min(w0r + r, hi - 1), pp = w / H;
                const bf16* x = a.sc.act + (size_t)pp * I;
                float acc = 0.f;
#pragma unroll
                for (int i = 0; i < DN::NV; ++i) {
                    float xf[DN::EPL], wf[DN::EPL];
                    const uint4 xv = *reinterpret_cast<const uint4*>(x + i * DN::CH + lane * DN::EPL);
                    bf16_to_f32<8>(&xv, xf);
                    if constexpr (FP8) fp8_to_f32<DN::EPL>(&wv[r][i], wf); else bf16_to_f32<DN::EPL>(&wv[r][i], wf);
#pragma unroll
                    for (int j = 0; j < DN::EPL; ++j) acc += wf[j] * xf[j];
                }
                out[r] = warp_sum(acc);
            }
            if (lane == 0) {
#pragma unroll
                for (int r = 0; r < DN::RD; ++r) if (w0r + r < hi) {
                    const int w = w0r + r, pp = w / H, nn = w % H;
                    float v = out[r];
                    if (FP8) v *= lw.down_scale[(size_t)expert(pp) * H + nn];
                    atomicAdd(a.sc.out2 + (pp / K) * H + nn, v * weight(pp));
                }
            }
        }
    }
}

template <int BT, bool FP8, typename W>
__device__ __forceinline__ void run_op(Ctx& c, const Instr& in, W wait) {
    switch (in.op) {
    case OP_QKV: op_qkv<BT>(c, in, wait); break;
    case OP_ATTN: op_attn<BT>(c, in, wait); break;
    case OP_COMBINE: op_combine(c, in, wait); break;
    case OP_OPROJ: op_oproj<BT>(c, in, wait); break;
    case OP_POST: op_post<BT>(c, in, wait); break;
    case OP_TOPK: op_topk<BT>(c, in, wait); break;
    case OP_GATEUP: op_gateup<BT, FP8>(c, in, wait); break;
    case OP_DOWN: op_down<BT, FP8>(c, in, wait); break;
    case OP_LMHEAD: op_lmhead<BT>(c, in, wait); break;
    case OP_ARGMAX: op_argmax(c, in, wait); break;
    default: wait();
    }
}
