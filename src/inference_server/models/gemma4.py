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
import torch.nn.functional as F


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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF-style: [-x[..., d/2:], x[..., :d/2]]. Differs from original RoPE's interleaved pairs."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """Apply rotary embeddings: (x * cos) + (rotate_half(x) * sin). cos/sin broadcast over heads."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class GemmaRotaryEmbedding(nn.Module):
    """Produces (cos, sin) tables for one layer type.

    Gemma 4 uses two configs: sliding layers (base=10000, full rotation) and
    full-attention layers (base=1e6, partial_rotary_factor=0.25). Partial rotation
    is encoded by zero-padding the upper portion of inv_freq → those dims pass through
    identity (cos=1, sin=0).
    """

    def __init__(self, head_dim: int, base: float, partial_rotary_factor: float = 1.0):
        super().__init__()
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        # Real frequencies for the rotated portion
        freqs = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim))
        # Pad to head_dim/2 with zeros (those dims become identity rotation)
        pad = head_dim // 2 - rope_angles
        if pad > 0:
            freqs = torch.cat([freqs, torch.zeros(pad, dtype=torch.float32)], dim=0)
        self.register_buffer("inv_freq", freqs, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, S] → cos/sin: [B, S, head_dim]
        freqs = position_ids[..., None].float() * self.inv_freq[None, None, :]  # [B, S, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, head_dim] — matches rotate_half layout
        return emb.cos(), emb.sin()


class GemmaAttention(nn.Module):
    """GQA attention with Q-norm, K-norm, V-norm-no-scale, RoPE on Q/K, scale=1.0."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False, dtype=dtype)

        self.q_norm = GemmaRMSNorm(head_dim, eps=eps, dtype=dtype)
        self.k_norm = GemmaRMSNorm(head_dim, eps=eps, dtype=dtype)
        self.v_norm = GemmaRMSNorm(head_dim, eps=eps, with_scale=False, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rotary(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)  # [B, Hq, S, D]

        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = apply_rotary(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)  # [B, Hkv, S, D]

        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_norm(v).transpose(1, 2)                              # [B, Hkv, S, D]

        # GQA expansion (free via enable_gqa, but explicit is clearer for now)
        if self.num_q_heads != self.num_kv_heads:
            rep = self.num_q_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # scale=1.0 — magnitude already absorbed by Q/K RMSNorm
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        attn = attn.transpose(1, 2).reshape(B, S, -1).contiguous()
        return self.o_proj(attn)

    def load_hf_weights(self, sd: dict, layer_idx: int) -> None:
        p = f"model.language_model.layers.{layer_idx}.self_attn"
        self.q_proj.weight.data.copy_(sd[f"{p}.q_proj.weight"])
        self.k_proj.weight.data.copy_(sd[f"{p}.k_proj.weight"])
        self.v_proj.weight.data.copy_(sd[f"{p}.v_proj.weight"])
        self.o_proj.weight.data.copy_(sd[f"{p}.o_proj.weight"])
        self.q_norm.weight.data.copy_(sd[f"{p}.q_norm.weight"])
        self.k_norm.weight.data.copy_(sd[f"{p}.k_norm.weight"])


class GemmaMLP(nn.Module):
    """GeGLU MLP: down( gelu_tanh(gate(x)) * up(x) ). No bias. Hidden→intermediate→hidden."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

    def load_hf_weights(self, sd: dict, layer_idx: int) -> None:
        p = f"model.language_model.layers.{layer_idx}.mlp"
        self.gate_proj.weight.data.copy_(sd[f"{p}.gate_proj.weight"])
        self.up_proj.weight.data.copy_(sd[f"{p}.up_proj.weight"])
        self.down_proj.weight.data.copy_(sd[f"{p}.down_proj.weight"])


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
