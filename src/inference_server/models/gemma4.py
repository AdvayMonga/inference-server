"""Custom Gemma 4 forward (M1).

Bottom-up: each piece is parity-tested against tests/fixtures/gemma4_e2b_parity.pt
before the next one is built. No PagedAttention yet — that's M2.

Architecture quirks to remember:
  - Tied embeddings; output scaled by sqrt(hidden_size) computed in bf16.
  - 5 layernorms per decoder layer; QK-norm before attention.
  - GeGLU MLP (gelu_pytorch_tanh), not SwiGLU.
  - GQA 8:1, head_dim 256, hidden_size 1536 (Q maps 1536→2048, K/V map 1536→256).
  - 35 layers, pattern of (4 sliding window=512, 1 full) × 7.
  - Dual RoPE: sliding base=10000 full-rotation; full base=1e6, partial_rotary_factor=0.25.
  - Per-layer embedding gating: embed_tokens_per_layer + per_layer_input_gate + per_layer_projection + post_per_layer_input_norm injected each layer.
  - Final-logit softcapping at 30: tanh(x/30) * 30.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GemmaRMSNorm(nn.Module):
    """RMSNorm: x / sqrt(mean(x²) + eps) * weight. Compute in fp32, cast back."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        # eps inside the sqrt for stability; pow(-.5) to mirror HF's JAX-portability choice
        normed = x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.to(x.dtype)


class GemmaEmbedding(nn.Module):
    """Token embedding + sqrt(hidden_size) scaling (computed in bf16, ~39.25 for E2B)."""

    def __init__(self, vocab_size: int, hidden_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        # Cast scale to bf16 to match HF's exact arithmetic (sqrt(1536) → 39.25 in bf16)
        self.register_buffer(
            "normalizer",
            torch.tensor(hidden_size, dtype=torch.float32).sqrt().to(dtype),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def load_hf_weights(self, hf_state_dict: dict) -> None:
        """Load weights from a Gemma4 HF state_dict.

        Expects keys under `model.language_model.embed_tokens.weight`.
        """
        self.embed_tokens.weight.data.copy_(
            hf_state_dict["model.language_model.embed_tokens.weight"]
        )
