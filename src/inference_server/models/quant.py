"""Weight-only int8 quantization for the custom Gemma 4 forward (decode bandwidth lever).

Decode is memory-bandwidth-bound on weight reads (see HANDOFF 2026-06-13), so we store the
Linear weights as int8 and read 2× fewer bytes/step. Activations stay bf16 (we're not
compute-bound, so int8 activations would only cost accuracy).

The win requires the int8→bf16 dequant to be FUSED into the matmul so weights stay int8 in
HBM — a naive dequant-then-bf16-matmul materializes the bf16 weight first and moves the same
bytes (no win). On CUDA that fusion comes from `torch.compile` (Inductor fuses the `.to(dtype)`
into the GEMV prologue) — the gpt-fast/torchao path, and why our forward being compile-clean
matters. The forward here is the plain formulation (`F.linear(x, w_int8.to(x))*scale`); the
backend compiles these modules on CUDA. (`torch._weight_int8pack_mm` is the CPU/MPS-only ATen
op — no CUDA kernel exists — so we don't use it.)

Quantization is per-output-channel symmetric round-to-nearest (one bf16 scale per output
feature): calibration-free and near-lossless for weight-only, since we only quantize weights
(activation outliers, the usual int8 failure mode, never enter the picture).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _int8_linear(x: torch.Tensor, weight_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return F.linear(x, weight_int8.to(x.dtype)) * scale


# One shared compiled artifact for ALL quantized layers. `dynamic=True` makes the weight/seq
# dims symbolic, so the ~10 distinct layer shapes reuse a single graph instead of thrashing the
# recompile limit (per-module .compile() shares one code-object cache → shape thrash → OOM).
_int8_linear_compiled = torch.compile(_int8_linear, dynamic=True)


class QuantizedLinear(nn.Module):
    """Bias-free int8 weight-only linear. Stores int8 weight [out, in] + bf16 scale [out].
    `compiled=True` routes through the shared dynamic-compiled dequant-matmul (CUDA: Inductor
    fuses the dequant into the GEMV → int8 stays in HBM). Eager otherwise (CPU/correctness)."""

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, compiled: bool = False):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)  # [out, in] int8
        self.register_buffer("scale", scale)              # [out] bf16
        self.out_features = weight_int8.shape[0]
        self._fn = _int8_linear_compiled if compiled else _int8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fn(x, self.weight_int8, self.scale)

    @classmethod
    def from_linear(cls, lin: nn.Linear, compiled: bool = False) -> "QuantizedLinear":
        w = lin.weight.data
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12).to(w.dtype)  # per-output-channel
        w_int8 = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return cls(w_int8, scale, compiled=compiled)


def quantize_model_int8(model: nn.Module, skip: tuple[str, ...] = ("lm_head",),
                        compiled: bool = False) -> int:
    """Replace every nn.Linear (except `skip` names) with a QuantizedLinear, in place.

    Skips lm_head — its weight is tied to the embedding table (used for a gather, not a matmul),
    so quantizing it would corrupt the embeddings. Embeddings/norms/scalars aren't nn.Linear,
    so they're untouched. `compiled` routes layers through the shared compiled GEMV. Returns count."""
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name not in skip:
                setattr(parent, name, QuantizedLinear.from_linear(child, compiled=compiled))
                n += 1
    return n
