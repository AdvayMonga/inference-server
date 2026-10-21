"""Custom Qwen3-MoE forward (Qwen3-30B-A3B), parity-tested against HF Qwen3MoeForCausalLM.

Call surface: kv_cache / attn_mask / paged_ctx / logits_index; the backend, scheduler, prefix
cache and paged-attention kernels drive it through these.

Architecture: every layer is full attention (GQA, per-head QK RMSNorm, RoPE, scale 1/sqrt(d))
followed by a sparse MoE block (softmax router → top-k → renormalise → SwiGLU experts). Untied
lm_head, no softcap. Module names mirror HF's state_dict keys so loading is one load_state_dict.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference_server.models import fused
from inference_server.models.common import RotaryEmbedding, _skip_param_init, apply_rotary


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class Qwen3RMSNorm(nn.Module):
    """HF Qwen3 RMSNorm: normalise in fp32, cast back, THEN scale by the weight."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and fused._on(x):
            return fused.rmsnorm(x, self.weight, self.eps)
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


class Qwen3Attention(nn.Module):
    """GQA attention, full causal every layer. Exposes the attrs make_pools reads."""

    is_kv_shared = False

    def __init__(self, hidden_size: int, num_q_heads: int, num_kv_heads: int, head_dim: int,
                 eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=eps, dtype=dtype)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=eps, dtype=dtype)

    @torch.no_grad()
    def fuse_qkv(self) -> None:
        """One GEMM for q, k and v (their weights concatenated): three launches per layer become one."""
        w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        del self.q_proj, self.k_proj, self.v_proj
        self.register_buffer("qkv_weight", w.contiguous())

    def _qkv(self, x):
        qd, kd = self.num_q_heads * self.head_dim, self.num_kv_heads * self.head_dim
        if hasattr(self, "qkv_weight"):
            return F.linear(x, self.qkv_weight).split([qd, kd, kd], dim=-1)
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def forward(self, x, cos, sin, past_kv=None, attn_mask=None, paged_ctx=None, kv_layer_idx=None):
        B, S, _ = x.shape
        q, k, v = self._qkv(x)
        v = v.view(B, S, self.num_kv_heads, self.head_dim)
        if x.is_cuda and fused._on(x):          # head norm + rotary in one launch each
            q = fused.rmsnorm_rope(q.view(B, S, self.num_q_heads, self.head_dim), self.q_norm.weight,
                                   cos, sin, self.q_norm.eps).transpose(1, 2)
            k = fused.rmsnorm_rope(k.view(B, S, self.num_kv_heads, self.head_dim), self.k_norm.weight,
                                   cos, sin, self.k_norm.eps).transpose(1, 2)
        else:
            q = self.q_norm(q.view(B, S, self.num_q_heads, self.head_dim))
            k = self.k_norm(k.view(B, S, self.num_kv_heads, self.head_dim))
            q = apply_rotary(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [B, Hq, S, D]
            k = apply_rotary(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [B, Hkv, S, D]
        v = v.transpose(1, 2)

        if paged_ctx is not None:   # paged path (the Triton kernels are imported only where used: CPU runs it too)
            from inference_server.models.prefill_attention import sdpa_prefill, sdpa_prefill_packed
            paged_ctx.append(kv_layer_idx, k, v)
            pool = paged_ctx.pools[kv_layer_idx]
            if getattr(paged_ctx, "is_prefill", False):
                eager = hasattr(paged_ctx, "caches")                           # the graph ctx has no caches
                if eager and PREFILL_ATTN == "flashinfer" and q.is_cuda:       # padded or packed alike
                    from inference_server.models.prefill_flashinfer import flashinfer_prefill
                    out = flashinfer_prefill(q, paged_ctx, kv_layer_idx, self.scale)
                elif eager and getattr(paged_ctx, "packed", False):
                    out = sdpa_prefill_packed(q, k, v, paged_ctx, kv_layer_idx, self.scale)
                elif eager and PREFILL_ATTN == "sdpa":
                    out = sdpa_prefill(q, k, v, paged_ctx, kv_layer_idx, self.scale)
                else:
                    from inference_server.models.paged_attention_kernel import paged_prefill_attention
                    bt = paged_ctx.block_table_tensor(kv_layer_idx)
                    out = paged_prefill_attention(q.transpose(1, 2), pool.k, pool.v, bt,
                                                  paged_ctx.prefix_lens, paged_ctx.suffix_lens,
                                                  scale=self.scale)
            else:
                from inference_server.models.paged_attention_kernel import paged_decode_attention
                bt, sl = paged_ctx.block_table_tensor(kv_layer_idx)
                out = paged_decode_attention(q.squeeze(2), pool.k, pool.v, bt, sl, scale=self.scale)
            return self.o_proj(out.reshape(B, S, -1)), None

        if past_kv is not None:
            k_all = torch.cat([past_kv[0], k], dim=2)
            v_all = torch.cat([past_kv[1], v], dim=2)
        else:
            k_all, v_all = k, v
        rep = self.num_q_heads // self.num_kv_heads
        k_exp = k_all.repeat_interleave(rep, dim=1)
        v_exp = v_all.repeat_interleave(rep, dim=1)

        S_q, S_k = q.shape[2], k_exp.shape[2]
        if attn_mask is None and S_k > S_q > 1:
            # Cached multi-token suffix: is_causal cannot express the offset, so mask by position.
            qpos = torch.arange(S_q, device=q.device) + (S_k - S_q)
            attn_mask = torch.arange(S_k, device=q.device)[None, :] <= qpos[:, None]
        if attn_mask is not None:
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask, scale=self.scale)
        else:
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=S_q == S_k, scale=self.scale)
        out = self.o_proj(attn.transpose(1, 2).reshape(B, S, -1))
        return out, (k, v)   # only the new K/V; the model owns the cache


@functools.lru_cache(maxsize=None)
def _experts_arange(num_experts: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_experts, device=device)


def route(router_logits: torch.Tensor, top_k: int, norm_topk_prob: bool):
    """Softmax over experts in fp32 → top-k → optional renormalise. Returns (weights, expert ids)."""
    probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    weights, experts = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(router_logits.dtype), experts


# Prefill attention on CUDA: "flashinfer" (default) is FlashInfer's paged batch prefill (prefill_flashinfer.py;
# H200 2026-10-17: long_prompt 1.77 -> 8.43 req/s and saturated TTFT p99 at 128 clients 1.3 -> 0.67 s against
# sdpa under the mixed scheduler), falling back to "sdpa" (torch SDPA over gathered prefixes) when the package is
# missing; "triton" is the paged kernel (~100x slower at 4k+).
PREFILL_ATTN = os.environ.get("CUSTOM_BACKEND_PREFILL_ATTN", "flashinfer")
if PREFILL_ATTN == "flashinfer":
    from inference_server.models import prefill_flashinfer as _fi
    if not _fi.available():
        PREFILL_ATTN = "sdpa"

# Up to this many (token, slot) pairs the MoE GEMMs run in the Triton kernels (moe_kernel.py); 0 = never. Measured per
# layer call on the H200 (2026-10-20, ms: torch grouped GEMMs / Triton bf16 activations / Triton W8A8): 256 rows
# 0.38 / 0.34 / 0.29, 512 rows 0.39 / 0.44 / 0.30, 1,024 rows 0.47 / 0.70 / 0.50, 2,048 rows 0.61 / 1.17 / 0.83, 8,192
# rows 2.95 / 4.00 / 2.86: the Triton paths win up to ~1,024 rows (8,192 pairs), W8A8 from 256 rows; beyond, torch's.
# All three sit at 10-13% of fp8 peak at prefill sizes.
MOE_KERNEL_MAX_ROWS = int(os.environ.get("CUSTOM_BACKEND_MOE_KERNEL_ROWS", "8192"))


# From this many pairs the Triton GEMMs take fp8 activations too (W8A8 on the tensor cores, per-token scales: the same
# quantization as the torch prefill path); below it the activations stay bf16 (decode at small batch is bound by the
# weight stream, and bf16 rows are the cheaper tile there: see the table above).
MOE_A8_MIN_PAIRS = int(os.environ.get("CUSTOM_BACKEND_MOE_A8_PAIRS", "2048"))


def moe_block_m(pairs: int) -> int:
    """Rows per GEMM tile for `pairs` (token, slot) pairs: 16 at decode sizes (a tile is one expert's few rows); taller
    tiles once each expert has tens of rows, so its weights are read once per tile of 64 rows, not of 16."""
    return 16 if pairs <= 2048 else 32 if pairs <= 4096 else 64
# Up to this many tokens the whole router (softmax, top-k, grouping, padding) is one launch (moe_kernel.route_fused);
# above it, the same in three (route_multi, a counting sort). 0 = never the one-launch router.
ROUTE_FUSED_MAX_TOKENS = int(os.environ.get("CUSTOM_BACKEND_ROUTE_FUSED_TOKENS", "32"))

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric fp8 e4m3 along the last dim: (q, scale), scale fp32 of shape x.shape[:-1], so x ~ q * scale[..., None]."""
    amax = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12)
    scale = amax / FP8_MAX
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale.squeeze(-1)


def _fake_quantize(x: torch.Tensor) -> torch.Tensor:
    q, s = quantize_rowwise(x)
    return (q.to(torch.float32) * s[..., None]).to(x.dtype)


class Qwen3Experts(nn.Module):
    """All experts' SwiGLU weights as 3D tensors (HF's layout): gate_up [E, 2I, H], down [E, H, I]. `quantize_fp8`
    swaps them for fp8 e4m3 with one fp32 scale per expert and output channel; activations are then quantized per
    token on the fly and the GEMMs run in fp8 (half the weight bytes per decode step, the MoE's limit)."""

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate: int,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.fp8 = False
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate, hidden_size, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate, dtype=dtype))

    @torch.no_grad()
    def quantize_fp8(self) -> None:
        """Weights to fp8 (per output channel), the bf16 copies freed. Purely mathematical: no data touches it."""
        for name in ("gate_up_proj", "down_proj"):
            w = self._parameters.pop(name).data
            q, s = quantize_rowwise(w)
            del w
            self.register_buffer(f"{name}_fp8", q)
            self.register_buffer(f"{name}_scale", s.contiguous())
        self.fp8 = True

    def gate_up(self, a: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        return self._mm(a, "gate_up_proj", offs)

    def down(self, a: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        return self._mm(a, "down_proj", offs)

    def weights(self, name: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(weight stack, per-channel scale or None) for the Triton kernels."""
        if self.fp8:
            return getattr(self, f"{name}_fp8"), getattr(self, f"{name}_scale")
        return getattr(self, name), None

    def _mm(self, a: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM of the expert-sorted rows `a` against weight stack `name`, groups ending at `offs`."""
        if not self.fp8:
            return torch._grouped_mm(a, getattr(self, name).transpose(-2, -1), offs=offs)
        w, s = getattr(self, f"{name}_fp8"), getattr(self, f"{name}_scale")
        if a.is_cuda:
            if fused._on(a):
                from inference_server.models import moe_kernel
                a8, sa = moe_kernel.quant_rows(a.contiguous())
            else:
                a8, sa = quantize_rowwise(a)
            return torch._scaled_grouped_mm(a8, w.transpose(-2, -1), sa, s, offs=offs, out_dtype=a.dtype)
        # Reference path (no fp8 GEMM here): the same roundings on both operands, then a bf16 grouped GEMM.
        deq = (w.to(torch.float32) * s[..., None]).to(a.dtype)
        return torch._grouped_mm(_fake_quantize(a), deq.transpose(-2, -1), offs=offs)


class Qwen3SparseMoE(nn.Module):
    """Router + experts as two grouped GEMMs over expert-sorted tokens; no host sync."""

    def __init__(self, hidden_size: int, moe_intermediate: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        self.experts = Qwen3Experts(num_experts, hidden_size, moe_intermediate, dtype=dtype)

    def forward(self, x: torch.Tensor, tokens: torch.Tensor | None = None) -> torch.Tensor:
        """`tokens`: flat indices of the real tokens of a right-padded wave; only those are routed and run, the
        padding positions come back as zeros (their outputs are never read)."""
        B, S, H = x.shape
        flat = x.reshape(-1, H)
        if tokens is None:
            return self._run(flat).view(B, S, H)
        y = self._run(flat[tokens])
        return torch.zeros(flat.shape, dtype=y.dtype, device=y.device).index_copy_(0, tokens, y).view(B, S, H)

    def _run(self, flat: torch.Tensor) -> torch.Tensor:
        """The experts' output per row: bf16, or fp32 from the Triton path (its caller adds it to the residual)."""
        logits = self.gate(flat)
        on = flat.is_cuda and fused._on(flat)
        E = self.experts.num_experts
        T = flat.shape[0]
        M = T * self.top_k
        if on and M <= MOE_KERNEL_MAX_ROWS:
            # Decode-sized M: the Triton grouped GEMMs read only the active experts' weights (moe_kernel.py), bf16 or
            # fp8 with a per-channel scale; the routing is one launch (or four above ROUTE_FUSED_MAX_TOKENS) and the
            # down GEMM scatters weighted rows into an fp32 output.
            from inference_server.models import moe_kernel
            gu, gu_s = self.experts.weights("gate_up_proj")
            dn, dn_s = self.experts.weights("down_proj")
            router = moe_kernel.route_fused if T <= ROUTE_FUSED_MAX_TOKENS else moe_kernel.route_multi
            bm = moe_block_m(M)
            sorted_ids, expert_ids, tok, pair_w = router(logits, E, self.top_k, self.norm_topk_prob, bm)
            a8 = gu_s is not None and M >= MOE_A8_MIN_PAIRS
            a, sa = moe_kernel.quant_rows(flat) if a8 else (flat, None)
            h = moe_kernel.moe_gemm(a, tok, gu, sorted_ids, expert_ids, M, block_m=bm, w_scale=gu_s, a_scale=sa)
            act = fused.silu_mul(h)
            if a8:
                act, sact = moe_kernel.quant_rows(act)
            else:
                sact = None
            out = torch.zeros(flat.shape, dtype=torch.float32, device=flat.device)
            moe_kernel.moe_gemm_scatter(act, dn, sorted_ids, expert_ids, M, tok, pair_w, out, block_m=bm, w_scale=dn_s,
                                        a_scale=sact)
            return out
        weights, experts = (fused.topk_softmax(logits, self.top_k, self.norm_topk_prob, fp32=True) if on
                            else route(logits, self.top_k, self.norm_topk_prob))
        sorted_e, order = torch.sort(experts.view(-1), stable=True)   # (token, slot) pairs grouped by expert
        tok = order // self.top_k
        # Group ends, on device (bincount would sync to size its output).
        offs = torch.searchsorted(sorted_e, _experts_arange(E, flat.device), right=True).to(torch.int32)
        h = self.experts.gate_up(flat[tok], offs)
        if on:
            act = fused.silu_mul(h)
        else:
            gate, up = h.chunk(2, dim=-1)
            act = F.silu(gate) * up
        y = self.experts.down(act, offs)
        if on:                                      # one pass per token: its K rows gathered, weighted, summed (bf16)
            from inference_server.models import moe_kernel
            pos = torch.empty_like(order).scatter_(0, order, _experts_arange(order.numel(), flat.device))
            return moe_kernel.gather_weighted(y, pos.view(T, self.top_k), weights.contiguous())
        y = y * weights.view(-1)[order, None]
        return torch.zeros_like(flat).index_add_(0, tok, y.to(flat.dtype))


class Qwen3DecoderLayer(nn.Module):
    """input-norm → attn → residual → post-attn-norm → MoE → residual."""

    def __init__(self, *, hidden_size, num_q_heads, num_kv_heads, head_dim, moe_intermediate,
                 num_experts, top_k, norm_topk_prob, eps, dtype):
        super().__init__()
        self.self_attn = Qwen3Attention(hidden_size, num_q_heads, num_kv_heads, head_dim, eps, dtype)
        self.mlp = Qwen3SparseMoE(hidden_size, moe_intermediate, num_experts, top_k, norm_topk_prob, dtype)
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=eps, dtype=dtype)

    def forward(self, h, cos, sin, past_kv=None, attn_mask=None, paged_ctx=None, kv_layer_idx=None):
        a, kv_new = self.self_attn(self.input_layernorm(h), cos, sin, past_kv=past_kv,
                                   attn_mask=attn_mask, paged_ctx=paged_ctx, kv_layer_idx=kv_layer_idx)
        tokens = getattr(paged_ctx, "tokens", None)      # a padded prefill wave: the experts see real tokens only
        if h.is_cuda and fused._on(h):          # residual add + post-attention norm in one launch
            h, n = fused.add_rmsnorm(h, a, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
            y = self.mlp(n, tokens)
            return (fused.add_cast(h, y) if y.dtype == torch.float32 else h + y), kv_new
        h = h + a
        return h + self.mlp(self.post_attention_layernorm(h), tokens), kv_new


class Qwen3MoeModel(nn.Module):
    """Embedding → N decoder layers → final norm."""

    def __init__(self, *, vocab_size, hidden_size, num_layers, num_q_heads, num_kv_heads, head_dim,
                 moe_intermediate, num_experts, top_k, norm_topk_prob, rope_theta, eps,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.rope = RotaryEmbedding(head_dim, rope_theta)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(
            hidden_size=hidden_size, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, moe_intermediate=moe_intermediate, num_experts=num_experts,
            top_k=top_k, norm_topk_prob=norm_topk_prob, eps=eps, dtype=dtype,
        ) for _ in range(num_layers))
        self.norm = Qwen3RMSNorm(hidden_size, eps=eps, dtype=dtype)

    def forward(self, input_ids, position_ids=None, kv_cache=None, attn_mask=None, paged_ctx=None):
        h = self.embed_tokens(input_ids)
        if position_ids is None:
            start = kv_cache.seq_len if kv_cache is not None else 0
            position_ids = torch.arange(start, start + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = (t.to(h.dtype) for t in self.rope(position_ids))
        for i, layer in enumerate(self.layers):
            if paged_ctx is not None:
                h, _ = layer(h, cos, sin, paged_ctx=paged_ctx, kv_layer_idx=i)
                continue
            past = kv_cache.get(i) if kv_cache is not None else None
            h, kv_new = layer(h, cos, sin, past_kv=past, attn_mask=attn_mask)
            if kv_cache is not None:
                kv_cache.append(i, kv_new[0], kv_new[1])
        return self.norm(h)


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3MoeModel + untied lm_head."""

    def __init__(self, model: Qwen3MoeModel, vocab_size: int):
        super().__init__()
        self.model = model
        self.lm_head = nn.Linear(model.hidden_size, vocab_size, bias=False,
                                 dtype=model.embed_tokens.weight.dtype)

    def forward(self, input_ids, position_ids=None, kv_cache=None, attn_mask=None,
                paged_ctx=None, logits_index=None):
        """`logits_index` [B]: lm_head on that one position per row → [B, 1, V]."""
        h = self.model(input_ids, position_ids, kv_cache=kv_cache, attn_mask=attn_mask, paged_ctx=paged_ctx)
        if logits_index is not None:
            if h.shape[0] == 1 and logits_index.numel() > 1:       # a packed wave: several rows' last tokens in one row
                h = h[:, logits_index]
            else:
                h = h.gather(1, logits_index.view(-1, 1, 1).expand(-1, 1, h.shape[-1]))
        return self.lm_head(h)

    @classmethod
    def from_config(cls, cfg, dtype: torch.dtype = torch.bfloat16) -> "Qwen3MoeForCausalLM":
        """Build (parameters allocated, not initialised) from an HF Qwen3MoeConfig."""
        rope = cfg.rope_parameters or {}
        unsupported = {
            "dense MLP layers": list(cfg.mlp_only_layers) or cfg.decoder_sparse_step != 1,
            "sliding window": cfg.sliding_window is not None,
            "attention bias": cfg.attention_bias,
            "rope scaling": rope.get("rope_type", "default") != "default",
            "hidden_act": cfg.hidden_act != "silu",
            "tied embeddings": cfg.tie_word_embeddings,
        }
        bad = [k for k, v in unsupported.items() if v]
        if bad:
            raise ValueError(f"Qwen3-MoE forward does not support: {', '.join(bad)}")
        with _skip_param_init():
            model = Qwen3MoeModel(
                vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size,
                num_layers=cfg.num_hidden_layers, num_q_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                moe_intermediate=cfg.moe_intermediate_size, num_experts=cfg.num_experts,
                top_k=cfg.num_experts_per_tok, norm_topk_prob=cfg.norm_topk_prob,
                rope_theta=rope["rope_theta"], eps=cfg.rms_norm_eps, dtype=dtype,
            )
            return cls(model, cfg.vocab_size)

    @classmethod
    def from_state_dict(cls, cfg, sd: dict, dtype: torch.dtype = torch.bfloat16) -> "Qwen3MoeForCausalLM":
        """Adopt an HF Qwen3MoeForCausalLM state_dict's tensors in place (strict keys, no copy)."""
        model = cls.from_config(cfg, dtype=dtype)
        model.load_state_dict(sd, strict=True, assign=True)
        return model

    @classmethod
    def from_safetensors(cls, model_name: str, device: str | torch.device = "cpu",
                         dtype: torch.dtype = torch.bfloat16, workers: int = 16) -> "Qwen3MoeForCausalLM":
        """Allocate on `device`, then stream every checkpoint shard straight into the parameters, in parallel."""
        import collections
        import glob
        import json
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        path = model_name if os.path.isdir(model_name) else snapshot_download(model_name)
        cfg = AutoConfig.from_pretrained(path)
        with torch.device(device):
            model = cls.from_config(cfg, dtype=dtype)
        params = dict(model.named_parameters())
        inter = cfg.moe_intermediate_size

        def target(key: str) -> torch.Tensor:
            """Where checkpoint tensor `key` lands: a whole parameter, or one expert's slice of the 3D stack."""
            if key in params:
                return params[key]
            head, _, rest = key.partition(".mlp.experts.")
            e, proj = rest.split(".")[:2]
            if proj == "down_proj":
                return params[f"{head}.mlp.experts.down_proj"][int(e)]
            half = slice(0, inter) if proj == "gate_proj" else slice(inter, 2 * inter)
            return params[f"{head}.mlp.experts.gate_up_proj"][int(e), half]

        local = threading.local()

        def load(shard: str) -> list[str]:
            """Per tensor: pread into this thread's reused host buffer, copy to the device. No fresh host memory per shard."""
            fd = os.open(shard, os.O_RDONLY)
            try:
                n = int.from_bytes(os.pread(fd, 8, 0), "little")
                header = json.loads(os.pread(fd, n, 8))
                header.pop("__metadata__", None)
                with torch.no_grad():   # no_grad is per-thread
                    for key, h in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                        start, end = h["data_offsets"]
                        if getattr(local, "buf", None) is None or local.buf.numel() < end - start:
                            local.buf = torch.empty(end - start, dtype=torch.uint8)
                        chunk = local.buf[:end - start]
                        view, done = memoryview(chunk.numpy()), 0
                        while done < end - start:   # preadv may return short
                            got = os.preadv(fd, [view[done:]], 8 + n + start + done)
                            if got <= 0:
                                raise ValueError(f"{shard}: truncated at {key}")
                            done += got
                        target(key).copy_(chunk.view(_ST_DTYPES[h["dtype"]]).view(h["shape"]))
            finally:
                os.close(fd)
            return list(header)

        shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        with ThreadPoolExecutor(workers) as pool:
            keys = [k for ks in pool.map(load, shards) for k in ks]
        # Every parameter filled exactly once: whole tensors by name, expert stacks by (expert, proj).
        filled = collections.Counter(k if k in params else k.rsplit(".", 1)[0] for k in keys)
        per_expert = [f"{n.rsplit('.', 1)[0]}.{e}.{p}" for n in params if n.endswith("experts.down_proj")
                      for e in range(cfg.num_experts) for p in ("gate_proj", "up_proj", "down_proj")]
        fused_ok = all(filled[n] == 1 for n in params)
        split_ok = all(filled[n] == 1 for n in params if ".mlp.experts." not in n) and \
            all(filled[k] == 1 for k in per_expert)
        if not (fused_ok or split_ok):
            raise ValueError(f"{path}: checkpoint keys do not cover the model exactly once")
        return model.eval()

    @classmethod
    def from_hf(cls, model_name: str = "Qwen/Qwen3-30B-A3B", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_name)
        sd = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).state_dict()
        return cls.from_state_dict(cfg, sd, dtype=dtype)
