"""The Triton kernels behind models.fused (imported only on CUDA: Triton is not installed everywhere)."""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def _rms(x, w, D: tl.constexpr, eps):
    """Qwen3 RMSNorm of one fp32 row: normalise, round to bf16, then scale by the weight."""
    var = tl.sum(x * x, axis=0) / D
    xn = (x / tl.sqrt(var + eps)).to(tl.bfloat16).to(tl.float32)
    return w * xn


@triton.jit
def _rmsnorm_kernel(X, W, Y, stride_x, stride_y, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    x = tl.load(X + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, _rms(x, w, D, eps).to(tl.bfloat16), mask=m)


@triton.jit
def _add_rmsnorm_kernel(H, A, W, HO, Y, stride, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    """h' = bf16(h + a); y = rmsnorm(h'). The residual add and the next norm in one pass."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    h = tl.load(H + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    a = tl.load(A + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    s = (h + a).to(tl.bfloat16)
    tl.store(HO + row * stride + cols, s, mask=m)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, _rms(s.to(tl.float32), w, D, eps).to(tl.bfloat16), mask=m)


@triton.jit
def _rmsnorm_rope_kernel(X, W, COS, SIN, Y, stride_row, stride_head, stride_cs, eps,
                         H: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr):
    """Per (token, block of HB heads): RMSNorm over D per head, then rotary with HF's rotate_half layout. cos/sin per
    token, first HALF. X is read with its own strides (a slice of the fused q/k/v GEMM output), Y is written
    contiguous [rows, H, D]."""
    row = tl.program_id(0)
    heads = tl.program_id(1) * HB + tl.arange(0, HB)
    hm = heads < H
    i = tl.arange(0, HALF)
    xbase = row * stride_row + heads[:, None] * stride_head + i[None, :]
    m = hm[:, None] & (i < HALF)[None, :]
    x1 = tl.load(X + xbase, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X + xbase + HALF, mask=m, other=0.0).to(tl.float32)
    w1 = tl.load(W + i).to(tl.float32)[None, :]
    w2 = tl.load(W + HALF + i).to(tl.float32)[None, :]
    var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D
    r = (1.0 / tl.sqrt(var + eps))[:, None]
    n1 = w1 * (x1 * r).to(tl.bfloat16).to(tl.float32)
    n2 = w2 * (x2 * r).to(tl.bfloat16).to(tl.float32)
    c = tl.load(COS + row * stride_cs + i).to(tl.float32)[None, :]
    s = tl.load(SIN + row * stride_cs + i).to(tl.float32)[None, :]
    ybase = (row * H + heads[:, None]) * D + i[None, :]
    tl.store(Y + ybase, (n1 * c - n2 * s).to(tl.bfloat16), mask=m)
    tl.store(Y + ybase + HALF, (n2 * c + n1 * s).to(tl.bfloat16), mask=m)


@triton.jit
def _silu_mul_kernel(X, Y, stride_x, stride_y, I: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = cols < I
    g = tl.load(X + row * stride_x + cols, mask=m, other=0.0).to(tl.float32)
    u = tl.load(X + row * stride_x + I + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (g * tl.sigmoid(g) * u).to(tl.bfloat16), mask=m)


@triton.jit
def _topk_softmax_kernel(X, Wt, Ids, stride_x, E: tl.constexpr, BLOCK_E: tl.constexpr, K: tl.constexpr,
                         RENORM: tl.constexpr):
    """Softmax over E experts in fp32, the top K by iterated argmax (lowest index on ties), optional renormalise."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    m = cols < E
    x = tl.load(X + row * stride_x + cols, mask=m, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, axis=0))
    p = p / tl.sum(p, axis=0)
    p = tl.where(m, p, -1.0)
    q = p
    total = 0.0
    for _ in tl.static_range(K):
        v = tl.max(q, axis=0)
        idx = tl.min(tl.where(q == v, cols, BLOCK_E), axis=0)
        total += v
        q = tl.where(cols == idx, -1.0, q)
    scale = 1.0 / total if RENORM else 1.0
    for j in tl.static_range(K):
        v = tl.max(p, axis=0)
        # A NaN row (a graph's dummy rows can carry them) matches nothing: clamp so the id stays a real expert.
        idx = tl.minimum(tl.min(tl.where(p == v, cols, BLOCK_E), axis=0), E - 1)
        tl.store(Wt + row * K + j, v * scale)
        tl.store(Ids + row * K + j, idx.to(tl.int64))
        p = tl.where(cols == idx, -1.0, p)


@triton.jit
def _add_cast_kernel(H, Y, OUT, stride, D: tl.constexpr, BLOCK: tl.constexpr):
    """out = bf16(h + y) for a bf16 residual and an fp32 row."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < D
    h = tl.load(H + row * stride + cols, mask=m, other=0.0).to(tl.float32)
    y = tl.load(Y + row * stride + cols, mask=m, other=0.0)
    tl.store(OUT + row * stride + cols, (h + y).to(tl.bfloat16), mask=m)


@triton.jit
def _kv_store_kernel(KN, VN, PK, PV, BT, LB, SLOT, stride_kr, stride_kh, stride_vr, stride_vh, stride_bt,
                     stride_pb, stride_ph, stride_ps, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """Row r's new K and V ([H, D] each, their own strides: V is a slice of the fused q/k/v GEMM) into
    pool[bt[r, lb[r]], :, slot[r], :], both pools in one launch."""
    r = tl.program_id(0)
    b = tl.load(BT + r * stride_bt + tl.load(LB + r)).to(tl.int64)
    s = tl.load(SLOT + r).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < H * D
    h, d = offs // D, offs % D
    dst = b * stride_pb + h * stride_ph + s * stride_ps + d
    tl.store(PK + dst, tl.load(KN + r * stride_kr + h * stride_kh + d, mask=m, other=0.0).to(PK.dtype.element_ty), mask=m)
    tl.store(PV + dst, tl.load(VN + r * stride_vr + h * stride_vh + d, mask=m, other=0.0).to(PV.dtype.element_ty), mask=m)



@triton.jit
def _embed_rope_kernel(IDS, POS, WEMB, INV, HOUT, COS, SIN, H: tl.constexpr, BLOCK_H: tl.constexpr,
                       HALF: tl.constexpr):
    """Per token: its embedding row, and its rotary cos/sin (fp32 angle = position * inv_freq, accurate cos/sin, bf16;
    HF's layout, the halves repeat)."""
    row = tl.program_id(0)
    tok = tl.load(IDS + row).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    for c in range(0, H, BLOCK_H):
        m = (c + cols) < H
        tl.store(HOUT + row.to(tl.int64) * H + c + cols, tl.load(WEMB + tok * H + c + cols, mask=m), mask=m)
    i = tl.arange(0, HALF)
    f = tl.load(POS + row).to(tl.float32) * tl.load(INV + i)
    c16, s16 = libdevice.cos(f).to(tl.bfloat16), libdevice.sin(f).to(tl.bfloat16)
    base = row * (2 * HALF)
    tl.store(COS + base + i, c16)
    tl.store(COS + base + HALF + i, c16)
    tl.store(SIN + base + i, s16)
    tl.store(SIN + base + HALF + i, s16)


@triton.jit
def _step_inputs_kernel(SEQ, KV_BLOCK, KV_SLOT, ATTN_SL, N, BS: tl.constexpr, BLOCK: tl.constexpr):
    """A decode step's per-row K/V block and slot and the attention lengths, from the rows' lengths."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < N
    s = tl.load(SEQ + r, mask=m, other=0)
    tl.store(KV_BLOCK + r, s // BS, mask=m)
    tl.store(KV_SLOT + r, s % BS, mask=m)
    tl.store(ATTN_SL + r, (s + 1).to(tl.int32), mask=m)


@triton.jit
def _stage_decode_kernel(CUR, POS, SEQ, BT, NEW_BLK, NEW_LB, G_TOK, G_POS, G_SEQ, G_BT, SEQ_NEXT, n, bt_cols,
                         stride_cur, stride_pos, stride_bt, g_cols, scratch, HAS_NEW: tl.constexpr,
                         ADVANCE: tl.constexpr, BLOCK_C: tl.constexpr):
    """A captured decode graph's inputs for bucket row r (live when r < n): its token, position and length (dummies
    0, 0, 1), its int32 block table (clamped at 0; dummies and stale columns the scratch block), a block allocated this
    step written into both tables, and the rows' lengths after the step."""
    r = tl.program_id(0)
    live = r < n
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < g_cols
    v = tl.load(BT + r * stride_bt + c, mask=cm & (c < bt_cols) & live, other=scratch)
    v = tl.maximum(v, 0)
    if HAS_NEW:
        nb = tl.load(NEW_BLK + r, mask=live, other=-1)
        hit = (c == tl.load(NEW_LB + r, mask=live, other=-1)) & (nb >= 0) & cm & (c < bt_cols)
        v = tl.where(hit, nb, v)
        tl.store(BT + r * stride_bt + c, nb + tl.zeros_like(v), mask=hit)
    tl.store(G_BT + r * g_cols + c, v.to(tl.int32), mask=cm)
    if tl.program_id(1) == 0:
        s = tl.load(SEQ + r, mask=live, other=1)
        tl.store(G_TOK + r, tl.load(CUR + r * stride_cur, mask=live, other=0))
        tl.store(G_POS + r, tl.load(POS + r * stride_pos, mask=live, other=0))
        tl.store(G_SEQ + r, s)
        if ADVANCE:
            tl.store(SEQ_NEXT + r, s + 1, mask=live)


@triton.jit
def _argmax_partial_kernel(X, PV, PI, V, stride_x, n_chunks, CHUNK: tl.constexpr):
    """Row r, chunk ch of the vocab: its largest logit and that logit's index (the first, on ties)."""
    r = tl.program_id(0)
    ch = tl.program_id(1)
    cols = ch * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(X + r.to(tl.int64) * stride_x + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
    i = tl.argmax(x, 0, tie_break_left=True)
    tl.store(PV + r * n_chunks + ch, tl.max(x, 0))
    tl.store(PI + r * n_chunks + ch, ch * CHUNK + i)


@triton.jit
def _argmax_final_kernel(PV, PI, OUT, n_chunks, BLOCK: tl.constexpr):
    """Row r: the index of the largest chunk maximum (the first chunk, on ties)."""
    r = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    v = tl.load(PV + r * n_chunks + i, mask=i < n_chunks, other=float("-inf"))
    best = tl.argmax(v, 0, tie_break_left=True)
    tl.store(OUT + r, tl.load(PI + r * n_chunks + best).to(tl.int64))


@triton.jit
def _kv_store_rows_kernel(KN, VN, PK, PV, BIDS, SLOTS, stride_kt, stride_kh, stride_vt, stride_vh, stride_pb,
                          stride_ph, stride_ps, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """Token t's new K and V ([H, D] each, their own strides) into pool[bids[t], :, slots[t], :], cast to the pools'
    dtype, both pools in one launch: a packed prefill's K/V write."""
    t = tl.program_id(0)
    b = tl.load(BIDS + t).to(tl.int64)
    s = tl.load(SLOTS + t).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < H * D
    h, d = offs // D, offs % D
    dst = b * stride_pb + h * stride_ph + s * stride_ps + d
    tl.store(PK + dst, tl.load(KN + t * stride_kt + h * stride_kh + d, mask=m, other=0.0).to(PK.dtype.element_ty), mask=m)
    tl.store(PV + dst, tl.load(VN + t * stride_vt + h * stride_vh + d, mask=m, other=0.0).to(PV.dtype.element_ty), mask=m)


@triton.jit
def _packed_slots_kernel(PREFIX, SUFFIX, START, BT, BIDS, SLOTS, stride_bt, BS: tl.constexpr, BLOCK: tl.constexpr):
    """Row r's chunk tokens of a packed wave: each token's K/V block (from the row's table) and slot."""
    r = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < tl.load(SUFFIX + r)
    p = tl.load(PREFIX + r) + i
    t = tl.load(START + r) + i
    tl.store(BIDS + t, tl.load(BT + r * stride_bt + p // BS, mask=m, other=0).to(tl.int64), mask=m)
    tl.store(SLOTS + t, (p % BS).to(tl.int64), mask=m)
