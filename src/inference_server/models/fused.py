"""Fused Triton kernels for the Qwen3-MoE layer on CUDA (bf16 activations).

A decode step launched ~4,500 kernels across 48 layers (RMSNorm as 8 ops, RoPE as ~6, SiLU*up as 3, routing as
~6, ...), and at 1-3 us each those tiny launches, not the GEMMs, set the ~20 ms step. Each function here is one
launch with fp32 arithmetic inside and one bf16 rounding at the end; the torch code it replaces stays as the
CPU path and the reference the tests compare against. Off with CUSTOM_BACKEND_FUSED=0."""

from __future__ import annotations

import os

import importlib.util

import torch

ENABLED = os.environ.get("CUSTOM_BACKEND_FUSED", "1") == "1" and importlib.util.find_spec("triton") is not None


class _Lazy:
    """The kernels module, imported on first use."""
    def __getattr__(self, name):
        from inference_server.models import fused_kernels
        return getattr(fused_kernels, name)


K = _Lazy()


def _on(*ts: torch.Tensor) -> bool:
    return ENABLED and all(t.is_cuda and t.dtype == torch.bfloat16 for t in ts)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    K._rmsnorm_kernel[(x2.shape[0],)](x2, weight, y, x2.stride(0), y.stride(0), eps, D=D,
                                     BLOCK=K.triton.next_power_of_2(D))
    return y.view(x.shape)


def add_rmsnorm(h: torch.Tensor, a: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(h + a, rmsnorm(h + a)), both bf16, one launch."""
    D = h.shape[-1]
    h2, a2 = h.reshape(-1, D).contiguous(), a.reshape(-1, D).contiguous()
    ho, y = torch.empty_like(h2), torch.empty_like(h2)
    K._add_rmsnorm_kernel[(h2.shape[0],)](h2, a2, weight, ho, y, h2.stride(0), eps, D=D, BLOCK=K.triton.next_power_of_2(D))
    return ho.view(h.shape), y.view(h.shape)


def rmsnorm_rope(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float) -> torch.Tensor:
    """x [B, S, H, D] -> head-wise RMSNorm then rotary (cos/sin [B, S, D], HF layout: halves repeat)."""
    B, S, H, D = x.shape
    if x.stride(3) != 1 or x.stride(2) != D or (B > 1 and x.stride(0) != S * x.stride(1)):
        x = x.contiguous()                      # any layout but "rows of H*D-contiguous heads" (never the qkv slice)
    y = torch.empty(B, S, H, D, dtype=x.dtype, device=x.device)
    cos2, sin2 = cos.reshape(B * S, -1), sin.reshape(B * S, -1)
    if not cos2.is_contiguous():
        cos2, sin2 = cos2.contiguous(), sin2.contiguous()
    K._rmsnorm_rope_kernel[(B * S * H,)](x, weight, cos2, sin2, y, x.stride(1), x.stride(2), cos2.stride(0), eps,
                                        H=H, D=D, HALF=D // 2)
    return y


def silu_mul(x: torch.Tensor) -> torch.Tensor:
    """x [M, 2I] -> silu(x[:, :I]) * x[:, I:]."""
    M, two_i = x.shape
    I = two_i // 2
    x = x.contiguous()
    y = torch.empty(M, I, dtype=torch.bfloat16, device=x.device)      # x may be the fp32 split-K accumulator
    BLOCK = 1024
    K._silu_mul_kernel[(M, K.triton.cdiv(I, BLOCK))](x, y, x.stride(0), y.stride(0), I=I, BLOCK=BLOCK)
    return y


def kv_store(pool_k: torch.Tensor, pool_v: torch.Tensor, bt: torch.Tensor, lb: torch.Tensor, slot: torch.Tensor,
             k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    """pool[bt[r, lb[r]], :, slot[r], :] = new[r, :, 0, :] for K and V in one launch (k/v_new [R, H, 1, D], `bt`
    the layer's [R, cols] block table, the pools [blocks, H, bs, D]; the gather and two index_put scatters it
    replaces were ~9 launches per layer)."""
    R, H, _, D = k_new.shape
    assert k_new.stride(3) == 1 and v_new.stride(3) == 1 and bt.stride(1) == 1
    K._kv_store_kernel[(R,)](k_new, v_new, pool_k, pool_v, bt, lb, slot, k_new.stride(0), k_new.stride(1),
                             v_new.stride(0), v_new.stride(1), bt.stride(0), pool_k.stride(0), pool_k.stride(1),
                             pool_k.stride(2), H=H, D=D, BLOCK=K.triton.next_power_of_2(H * D))


def topk_softmax(logits: torch.Tensor, k: int, renorm: bool, fp32: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Like models.qwen3_moe.route: (weights in logits' dtype, or fp32 as computed, expert ids int64), both [M, k]."""
    M, E = logits.shape
    logits = logits.contiguous()
    w = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, k, dtype=torch.int64, device=logits.device)
    K._topk_softmax_kernel[(M,)](logits, w, ids, logits.stride(0), E=E, BLOCK_E=K.triton.next_power_of_2(E), K=k,
                               RENORM=renorm)
    return (w if fp32 else w.to(logits.dtype)), ids


def add_cast(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """bf16(h + y) for a bf16 residual `h` and an fp32 `y` of the same shape, one launch."""
    D = h.shape[-1]
    h2, y2 = h.reshape(-1, D).contiguous(), y.reshape(-1, D).contiguous()
    out = torch.empty_like(h2)
    K._add_cast_kernel[(h2.shape[0],)](h2, y2, out, D, D=D, BLOCK=K.triton.next_power_of_2(D))
    return out.view(h.shape)
