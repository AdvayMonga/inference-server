"""The Triton kernels behind models.fused (imported only on CUDA: Triton is not installed everywhere)."""

from __future__ import annotations

import triton
import triton.language as tl

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

