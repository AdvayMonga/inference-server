"""Custom Qwen3-MoE forward (Qwen3-30B-A3B), parity-tested against HF Qwen3MoeForCausalLM.

Same call surface as GemmaForCausalLM (kv_cache / attn_mask / paged_ctx / logits_index), so the
backend, scheduler, prefix cache and paged-attention kernels drive it unchanged.

Architecture: every layer is full attention (GQA, per-head QK RMSNorm, RoPE, scale 1/sqrt(d))
followed by a sparse MoE block (softmax router → top-k → renormalise → SwiGLU experts). Untied
lm_head, no softcap. Module names mirror HF's state_dict keys so loading is one load_state_dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from inference_server.models.gemma4 import GemmaRotaryEmbedding, _skip_param_init, apply_rotary


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class Qwen3RMSNorm(nn.Module):
    """HF Qwen3 RMSNorm: normalise in fp32, cast back, THEN scale by the weight."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


class Qwen3Attention(nn.Module):
    """GQA attention, full causal every layer. Exposes the attrs make_pools_for_gemma reads."""

    is_kv_shared = False
    sliding_window = None

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

    def forward(self, x, cos, sin, past_kv=None, attn_mask=None, paged_ctx=None, kv_layer_idx=None):
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, S, self.num_q_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        q = apply_rotary(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [B, Hq, S, D]
        k = apply_rotary(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [B, Hkv, S, D]
        v = v.transpose(1, 2)

        if paged_ctx is not None:   # CUDA kernel path, identical contract to GemmaAttention
            from inference_server.models.paged_attention_kernel import (
                paged_decode_attention, paged_prefill_attention,
            )
            paged_ctx.append(kv_layer_idx, k, v)
            pool = paged_ctx.pools[kv_layer_idx]
            if getattr(paged_ctx, "is_prefill", False):
                bt = paged_ctx.block_table_tensor(kv_layer_idx)
                out = paged_prefill_attention(q.transpose(1, 2), pool.k, pool.v, bt,
                                              paged_ctx.prefix_lens, paged_ctx.suffix_lens,
                                              scale=self.scale)
            else:
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


def route(router_logits: torch.Tensor, top_k: int, norm_topk_prob: bool):
    """Softmax over experts in fp32 → top-k → optional renormalise. Returns (weights, expert ids)."""
    probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
    weights, experts = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(router_logits.dtype), experts


class Qwen3Experts(nn.Module):
    """All experts' SwiGLU weights as 3D tensors (HF's layout): gate_up [E, 2I, H], down [E, H, I]."""

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate: int,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_intermediate, hidden_size, dtype=dtype))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, moe_intermediate, dtype=dtype))


class Qwen3SparseMoE(nn.Module):
    """Router + experts as two grouped GEMMs over expert-sorted tokens; no host sync."""

    def __init__(self, hidden_size: int, moe_intermediate: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        self.experts = Qwen3Experts(num_experts, hidden_size, moe_intermediate, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        flat = x.reshape(-1, H)
        weights, experts = route(self.gate(flat), self.top_k, self.norm_topk_prob)
        E = self.experts.gate_up_proj.shape[0]
        sorted_e, order = torch.sort(experts.view(-1), stable=True)   # (token, slot) pairs grouped by expert
        tok = order // self.top_k
        # Group ends, on device (bincount would sync to size its output).
        offs = torch.searchsorted(sorted_e, torch.arange(E, device=x.device), right=True).to(torch.int32)
        gate, up = torch._grouped_mm(flat[tok], self.experts.gate_up_proj.transpose(-2, -1),
                                     offs=offs).chunk(2, dim=-1)
        y = torch._grouped_mm(F.silu(gate) * up, self.experts.down_proj.transpose(-2, -1), offs=offs)
        y = y * weights.view(-1)[order, None]
        return torch.zeros_like(flat).index_add_(0, tok, y.to(flat.dtype)).view(B, S, H)


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
        h = h + a
        return h + self.mlp(self.post_attention_layernorm(h)), kv_new


class Qwen3MoeModel(nn.Module):
    """Embedding → N decoder layers → final norm."""

    def __init__(self, *, vocab_size, hidden_size, num_layers, num_q_heads, num_kv_heads, head_dim,
                 moe_intermediate, num_experts, top_k, norm_topk_prob, rope_theta, eps,
                 dtype=torch.bfloat16):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=dtype)
        self.rope = GemmaRotaryEmbedding(head_dim, rope_theta)
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
        """`logits_index` [B]: lm_head on that one position per row → [B, 1, V] (see Gemma's)."""
        h = self.model(input_ids, position_ids, kv_cache=kv_cache, attn_mask=attn_mask, paged_ctx=paged_ctx)
        if logits_index is not None:
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
                         dtype: torch.dtype = torch.bfloat16, workers: int = 8) -> "Qwen3MoeForCausalLM":
        """Allocate on `device`, then stream every checkpoint shard straight into the parameters, in parallel."""
        import collections
        import glob
        import json
        import os
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

        def load(shard: str) -> list[str]:
            """One large read per shard (parallel across threads, unlike mmap faults), then zero-copy views."""
            buf = bytearray(os.path.getsize(shard))
            with open(shard, "rb", buffering=0) as f:
                f.readinto(memoryview(buf))
            n = int.from_bytes(buf[:8], "little")
            header = json.loads(buf[8:8 + n])
            header.pop("__metadata__", None)
            data = torch.frombuffer(buf, dtype=torch.uint8)[8 + n:]
            with torch.no_grad():   # no_grad is per-thread
                for key, h in header.items():
                    start, end = h["data_offsets"]
                    t = data[start:end].view(_ST_DTYPES[h["dtype"]]).view(h["shape"])
                    target(key).copy_(t)
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
        if not (fused_ok or split_ok) or sum(filled.values()) != len(keys):
            raise ValueError(f"{path}: checkpoint keys do not cover the model exactly once")
        return model.eval()

    @classmethod
    def from_hf(cls, model_name: str = "Qwen/Qwen3-30B-A3B", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(model_name)
        sd = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).state_dict()
        return cls.from_state_dict(cfg, sd, dtype=dtype)
