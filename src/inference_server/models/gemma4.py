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


class KVCache:
    """Per-layer (K, V) tensors stored as contiguous [B, H, S, D]. Append concatenates along S."""

    def __init__(self, num_layers: int):
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.k[layer_idx] is None:
            return None
        return self.k[layer_idx], self.v[layer_idx]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)

    @property
    def seq_len(self) -> int:
        """Length of cached sequence. Any populated layer suffices."""
        for k in self.k:
            if k is not None:
                return k.shape[2]
        return 0


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
    """GQA attention. KV-shared layers skip k_proj/v_proj and reuse another layer's K/V."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        sliding_window: int | None = None,
        is_kv_shared: bool = False,
        store_kv: bool = False,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.is_kv_shared = is_kv_shared
        self.store_kv = store_kv  # last non-shared layer of its type → cache K/V for sharers

        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False, dtype=dtype)
        self.q_norm = GemmaRMSNorm(head_dim, eps=eps, dtype=dtype)

        if not is_kv_shared:
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
            self.k_norm = GemmaRMSNorm(head_dim, eps=eps, dtype=dtype)
            self.v_norm = GemmaRMSNorm(head_dim, eps=eps, with_scale=False, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
        paged_ctx: object | None = None,
        kv_layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, S, _ = x.shape
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        q = self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rotary(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)  # [B, Hq, S, D]

        # CUDA kernel decode path: write new K/V into blocks, attend via block tables.
        # No gather, no padding. Non-shared layers append; shared layers read the source
        # layer's pool (already populated earlier this step). Returns (out, None).
        if paged_ctx is not None:
            from inference_server.models.paged_attention_kernel import (
                paged_decode_attention, paged_prefill_attention,
            )
            is_prefill = getattr(paged_ctx, "is_prefill", False)
            if not self.is_kv_shared:
                k_new = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
                k_new = self.k_norm(k_new)
                k_new = apply_rotary(k_new, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                v_new = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
                v_new = self.v_norm(v_new).transpose(1, 2)
                paged_ctx.append(kv_layer_idx, k_new, v_new)
            pool = paged_ctx.pools[kv_layer_idx]
            window = self.sliding_window if self.sliding_window is not None else (1 << 30)
            if is_prefill:
                # Multi-query paged prefill: suffix queries attend (causally) to paged prefix+suffix.
                bt = paged_ctx.block_table_tensor(kv_layer_idx)
                out = paged_prefill_attention(
                    q.transpose(1, 2), pool.k, pool.v, bt,
                    paged_ctx.prefix_lens, paged_ctx.suffix_lens, scale=1.0, window=window,
                )
                return self.o_proj(out.reshape(B, S, -1)), None
            bt, sl = paged_ctx.block_table_tensor(kv_layer_idx)
            out = paged_decode_attention(q.squeeze(2), pool.k, pool.v, bt, sl, scale=1.0, window=window)
            return self.o_proj(out.reshape(B, S, -1)), None

        if self.is_kv_shared:
            assert shared_kv is not None, "KV-shared layer needs shared_kv"
            k, v = shared_kv
        else:
            k_new = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
            k_new = self.k_norm(k_new)
            k_new = apply_rotary(k_new, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            v_new = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
            v_new = self.v_norm(v_new).transpose(1, 2)
            if past_kv is not None:
                k = torch.cat([past_kv[0], k_new], dim=2)
                v = torch.cat([past_kv[1], v_new], dim=2)
            else:
                k, v = k_new, v_new

        # GQA expansion
        if self.num_q_heads != self.num_kv_heads:
            rep = self.num_q_heads // self.num_kv_heads
            k_exp = k.repeat_interleave(rep, dim=1)
            v_exp = v.repeat_interleave(rep, dim=1)
        else:
            k_exp, v_exp = k, v

        # Causal + sliding-window masking. Sliding layers attend only to the last
        # `sliding_window` keys; full layers attend to all causal keys. RoPE is baked into the
        # stored K, so masking purely by position is correct.
        S_q, S_k = q.shape[2], k_exp.shape[2]
        W = self.sliding_window
        if attn_mask is not None:
            # Batched decode: external mask encodes per-row validity (real tokens right-aligned).
            # Sliding window = keep only the rightmost W columns (the W most-recent keys), same
            # cutoff for every row.
            if W is not None and S_k > W:
                keep = torch.arange(S_k, device=q.device) >= (S_k - W)
                attn_mask = attn_mask & keep[None, None, None, :]
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask, scale=1.0)
        elif (W is not None and S_k > W) or (S_k > S_q and S_q > 1):
            # Window active, OR a cached multi-token suffix (is_causal can't express the cache
            # offset). Build an explicit [S_q, S_k] mask by absolute position.
            qpos = torch.arange(S_q, device=q.device) + (S_k - S_q)
            kpos = torch.arange(S_k, device=q.device)
            allowed = kpos[None, :] <= qpos[:, None]                 # causal
            if W is not None:
                allowed = allowed & ((qpos[:, None] - kpos[None, :]) < W)
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=allowed, scale=1.0)
        else:
            # Short/cold square prefill or single-row decode within the window — exact prior
            # path (byte-identical to the parity fixture).
            is_causal = S_q == S_k
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal, scale=1.0)
        attn = attn.transpose(1, 2).reshape(B, S, -1).contiguous()
        out = self.o_proj(attn)
        # Return ONLY the newly-computed K/V for this layer. The model owns the cache and
        # decides whether to append to a contiguous tensor or paged blocks.
        # Shared layers produced nothing → None.
        if self.is_kv_shared:
            kv_out = None
        elif past_kv is None:
            kv_out = (k, v)        # no past — "new" is all of it
        else:
            # Slice off the new portion. k/v are [B, H, S_total, D]; new is the last S tokens.
            kv_out = (k[:, :, -S:, :], v[:, :, -S:, :])
        return out, kv_out

    def load_hf_weights(self, sd: dict, layer_idx: int) -> None:
        p = f"model.language_model.layers.{layer_idx}.self_attn"
        self.q_proj.weight.data.copy_(sd[f"{p}.q_proj.weight"])
        self.o_proj.weight.data.copy_(sd[f"{p}.o_proj.weight"])
        self.q_norm.weight.data.copy_(sd[f"{p}.q_norm.weight"])
        if not self.is_kv_shared:
            self.k_proj.weight.data.copy_(sd[f"{p}.k_proj.weight"])
            self.v_proj.weight.data.copy_(sd[f"{p}.v_proj.weight"])
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


class GemmaDecoderLayer(nn.Module):
    """One transformer block: input-norm → attn → post-attn-norm + residual
    → pre-ffn-norm → MLP → post-ffn-norm + residual
    → per-layer-input gating block (gate, gelu, mul, project, norm) + residual
    → multiply by learned layer_scalar.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_per_layer: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        sliding_window: int | None = None,
        is_kv_shared: bool = False,
        store_kv: bool = False,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size=hidden_size, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, eps=eps, dtype=dtype, sliding_window=sliding_window,
            is_kv_shared=is_kv_shared, store_kv=store_kv,
        )
        self.mlp = GemmaMLP(hidden_size, intermediate_size, dtype=dtype)

        self.input_layernorm = GemmaRMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps=eps, dtype=dtype)

        # Per-layer-input gating block
        self.per_layer_input_gate = nn.Linear(hidden_size, hidden_per_layer, bias=False, dtype=dtype)
        self.per_layer_projection = nn.Linear(hidden_per_layer, hidden_size, bias=False, dtype=dtype)
        self.post_per_layer_input_norm = GemmaRMSNorm(hidden_size, eps=eps, dtype=dtype)

        # Trained scalar, registered as buffer in HF.
        self.register_buffer("layer_scalar", torch.ones(1, dtype=dtype))

    def forward(
        self,
        h: torch.Tensor,
        per_layer_input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
        paged_ctx: object | None = None,
        kv_layer_idx: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Attention block
        residual = h
        a = self.input_layernorm(h)
        a, kv_out = self.self_attn(a, cos, sin, shared_kv=shared_kv, past_kv=past_kv,
                                   attn_mask=attn_mask, paged_ctx=paged_ctx, kv_layer_idx=kv_layer_idx)
        a = self.post_attention_layernorm(a)
        h = residual + a

        # MLP block
        residual = h
        m = self.pre_feedforward_layernorm(h)
        m = self.mlp(m)
        m = self.post_feedforward_layernorm(m)
        h = residual + m

        # Per-layer-input gating block
        residual = h
        g = self.per_layer_input_gate(h)
        g = F.gelu(g, approximate="tanh")
        g = g * per_layer_input
        g = self.per_layer_projection(g)
        g = self.post_per_layer_input_norm(g)
        h = residual + g

        h = h * self.layer_scalar
        return h, kv_out

    def load_hf_weights(self, sd: dict, layer_idx: int) -> None:
        p = f"model.language_model.layers.{layer_idx}"
        self.self_attn.load_hf_weights(sd, layer_idx)
        self.mlp.load_hf_weights(sd, layer_idx)
        self.input_layernorm.weight.data.copy_(sd[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm.weight.data.copy_(sd[f"{p}.post_attention_layernorm.weight"])
        self.pre_feedforward_layernorm.weight.data.copy_(sd[f"{p}.pre_feedforward_layernorm.weight"])
        self.post_feedforward_layernorm.weight.data.copy_(sd[f"{p}.post_feedforward_layernorm.weight"])
        self.per_layer_input_gate.weight.data.copy_(sd[f"{p}.per_layer_input_gate.weight"])
        self.per_layer_projection.weight.data.copy_(sd[f"{p}.per_layer_projection.weight"])
        self.post_per_layer_input_norm.weight.data.copy_(sd[f"{p}.post_per_layer_input_norm.weight"])
        self.layer_scalar.data.copy_(sd[f"{p}.layer_scalar"])


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


class GemmaModel(nn.Module):
    """Full stack: embedding (×√hidden), per-layer-input prep, 35 decoder layers, final norm."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        intermediate_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        global_head_dim: int,
        hidden_per_layer: int,
        layer_types: list[str],
        sliding_window: int,
        num_kv_shared_layers: int,
        use_double_wide_mlp: bool = True,   # E2B True; E4B False (config-driven)
        rope_base_sliding: float = 10000.0,
        rope_base_full: float = 1_000_000.0,
        rope_partial_full: float = 0.25,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_per_layer = hidden_per_layer
        self.layer_types = layer_types
        self.hidden_size = hidden_size

        # First non-shared / KV-shared split + lookup of which earlier layer to share from.
        first_shared = num_layers - num_kv_shared_layers
        prev = layer_types[:first_shared]
        self.first_shared = first_shared
        self.share_source: dict[int, int] = {}
        self.store_kv_per_idx: set[int] = set()
        # The "last non-shared layer of each type" stores K/V; every shared layer of that type reuses it.
        for t in set(prev):
            last_idx_of_type = first_shared - 1 - prev[::-1].index(t)
            self.store_kv_per_idx.add(last_idx_of_type)
            for i in range(first_shared, num_layers):
                if layer_types[i] == t:
                    self.share_source[i] = last_idx_of_type

        # Main embedding (scaled by √hidden in bf16 → 39.25 for E2B)
        self.embed_tokens = GemmaEmbedding(vocab_size, hidden_size, dtype=dtype)

        # Per-layer-input pipeline:
        #   1. embed_tokens_per_layer(input_ids) × √hidden_per_layer  → [B, S, L, D_per]
        #   2. per_layer_model_projection(inputs_embeds) × 1/√hidden  → reshape  → per_layer_projection_norm
        #   combined: (1 + 2) × 1/√2
        self.embed_tokens_per_layer = nn.Embedding(
            vocab_size, num_layers * hidden_per_layer, dtype=dtype,
        )
        self.register_buffer(
            "per_layer_embed_scale",
            torch.tensor(hidden_per_layer, dtype=torch.float32).sqrt().to(dtype),
            persistent=False,
        )
        self.per_layer_model_projection = nn.Linear(
            hidden_size, num_layers * hidden_per_layer, bias=False, dtype=dtype,
        )
        # Keep as Python floats (HF does) — avoids bf16 truncation of the scale value
        self.per_layer_proj_scale: float = hidden_size ** -0.5
        self.per_layer_projection_norm = GemmaRMSNorm(hidden_per_layer, eps=eps, dtype=dtype)
        self.per_layer_input_combine_scale: float = 2.0 ** -0.5

        # Two rotary tables (sliding head_dim, full global_head_dim).
        self.rope_sliding = GemmaRotaryEmbedding(head_dim, rope_base_sliding, partial_rotary_factor=1.0)
        self.rope_full = GemmaRotaryEmbedding(global_head_dim, rope_base_full, partial_rotary_factor=rope_partial_full)

        # Build layers.
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_sliding = layer_types[i] == "sliding_attention"
            layer_head_dim = head_dim if is_sliding else global_head_dim
            self.layers.append(GemmaDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size * (2 if (use_double_wide_mlp and i >= first_shared) else 1),
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=layer_head_dim,
                hidden_per_layer=hidden_per_layer,
                eps=eps,
                dtype=dtype,
                sliding_window=sliding_window if is_sliding else None,
                is_kv_shared=(i >= first_shared),
                store_kv=(i in self.store_kv_per_idx),
            ))

        self.norm = GemmaRMSNorm(hidden_size, eps=eps, dtype=dtype)

    def _per_layer_inputs(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        # Source 1: dedicated per-layer embedding table
        src1 = self.embed_tokens_per_layer(input_ids) * self.per_layer_embed_scale
        src1 = src1.view(B, S, self.num_layers, self.hidden_per_layer)
        # Source 2: project the main embeddings
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_proj_scale
        proj = proj.view(B, S, self.num_layers, self.hidden_per_layer)
        proj = self.per_layer_projection_norm(proj)
        return (proj + src1) * self.per_layer_input_combine_scale

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        paged_ctx: object | None = None,
    ):
        h = self.embed_tokens(input_ids)
        cached_len = kv_cache.seq_len if kv_cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                cached_len, cached_len + input_ids.shape[1], device=input_ids.device,
            ).unsqueeze(0)

        per_layer = self._per_layer_inputs(input_ids, h)  # [B, S, L, D_per]

        cos_sin = {
            "sliding_attention": self.rope_sliding(position_ids),
            "full_attention": self.rope_full(position_ids),
        }

        all_hidden = [] if return_hidden_states else None
        shared_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for i, layer in enumerate(self.layers):
            if return_hidden_states:
                all_hidden.append(h)  # HF records input-to-layer-i, then post-norm at the end
            cos, sin = cos_sin[self.layer_types[i]]

            if paged_ctx is not None:
                # Kernel decode path: shared layers read the source layer's pool directly.
                kv_idx = self.share_source[i] if i in self.share_source else i
                h, _ = layer(h, per_layer[:, :, i, :], cos, sin,
                             paged_ctx=paged_ctx, kv_layer_idx=kv_idx)
                continue

            sk = shared_kv.get(self.share_source[i]) if i in self.share_source else None
            past = kv_cache.get(i) if kv_cache is not None else None
            h, kv_new = layer(h, per_layer[:, :, i, :], cos, sin, shared_kv=sk, past_kv=past, attn_mask=attn_mask)
            if kv_new is not None:
                if kv_cache is not None:
                    kv_cache.append(i, kv_new[0], kv_new[1])
                if i in self.store_kv_per_idx:
                    # Downstream sharers need the FULL (cached + new) K/V at this layer.
                    if kv_cache is not None:
                        shared_kv[i] = kv_cache.get(i)
                    elif past is None:
                        shared_kv[i] = kv_new
                    else:
                        shared_kv[i] = (
                            torch.cat([past[0], kv_new[0]], dim=2),
                            torch.cat([past[1], kv_new[1]], dim=2),
                        )

        h = self.norm(h)
        if return_hidden_states:
            all_hidden.append(h)
            return h, all_hidden
        return h

    def load_hf_weights(self, sd: dict) -> None:
        self.embed_tokens.load_hf_weights(sd)
        self.embed_tokens_per_layer.weight.data.copy_(sd["model.language_model.embed_tokens_per_layer.weight"])
        self.per_layer_model_projection.weight.data.copy_(sd["model.language_model.per_layer_model_projection.weight"])
        self.per_layer_projection_norm.weight.data.copy_(sd["model.language_model.per_layer_projection_norm.weight"])
        self.norm.weight.data.copy_(sd["model.language_model.norm.weight"])
        for i, layer in enumerate(self.layers):
            layer.load_hf_weights(sd, layer_idx=i)


class GemmaForCausalLM(nn.Module):
    """Wraps GemmaModel with tied LM head and final-logit softcapping."""

    def __init__(self, model: GemmaModel, final_logit_softcapping: float = 30.0):
        super().__init__()
        self.model = model
        self.final_logit_softcapping = final_logit_softcapping
        # lm_head shares weight with embed_tokens (tied).
        self.lm_head = nn.Linear(model.hidden_size, model.embed_tokens.embed_tokens.num_embeddings, bias=False, dtype=model.embed_tokens.embed_tokens.weight.dtype)
        # Wire the tie.
        self.lm_head.weight = self.model.embed_tokens.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        kv_cache: KVCache | None = None,
        attn_mask: torch.Tensor | None = None,
        paged_ctx: object | None = None,
    ):
        if return_hidden_states:
            h, all_h = self.model(input_ids, position_ids, return_hidden_states=True, kv_cache=kv_cache)
        else:
            h = self.model(input_ids, position_ids, kv_cache=kv_cache, attn_mask=attn_mask, paged_ctx=paged_ctx)
        logits = self.lm_head(h)
        # softcap: tanh(x / s) * s — bounds logits to ±s, helps long-tail tokens
        if self.final_logit_softcapping:
            s = self.final_logit_softcapping
            logits = torch.tanh(logits / s) * s
        if return_hidden_states:
            return logits, all_h
        return logits

    @classmethod
    def from_hf(cls, model_name: str = "google/gemma-4-E2B-it", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_name).text_config
        model = GemmaModel(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_hidden_layers,
            intermediate_size=cfg.intermediate_size,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            global_head_dim=cfg.global_head_dim,
            hidden_per_layer=cfg.hidden_size_per_layer_input,
            layer_types=list(cfg.layer_types),
            sliding_window=cfg.sliding_window,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            use_double_wide_mlp=getattr(cfg, "use_double_wide_mlp", True),
            eps=cfg.rms_norm_eps,
            dtype=dtype,
        )
        sd = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).state_dict()
        model.load_hf_weights(sd)
        return cls(model, final_logit_softcapping=cfg.final_logit_softcapping)
