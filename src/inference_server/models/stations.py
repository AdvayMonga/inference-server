"""One layer's decode step as named stations: the code the CUDA decode forward runs (fused kernels on, Triton MoE
path), each station declaring the context entries it reads and writes, so a stretch of them can be captured, searched
and swapped as one region (`models/regions/`). Prefill, the CPU path and the torch MoE path do not run through here.

Context entries: activations ("h", "x", "q", ...), step inputs (`STEP`: rotary tables and the per-step K/V indices,
`paged_kv_cache.step_inputs`), the layer's K/V pool ("k_cache", "v_cache", written in place), the layer's weights
(`layer_weights`, "w.*") and its shapes ("cfg", `layer_config`)."""

from __future__ import annotations

import functools
import importlib
import pkgutil
import types
from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn.functional as F

from inference_server.models import fused, qwen3_moe, region_capture
from inference_server.models.paged_kv_cache import write_token


@dataclass(frozen=True)
class Station:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    fn: Callable[[dict], dict]          # fn(ctx) -> {name: tensor} for its writes


@dataclass(frozen=True)
class LayerConfig:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    eps: float                          # every norm of the layer (one rms_norm_eps in the HF config)
    num_experts: int
    top_k: int
    norm_topk_prob: bool


STEP = ("cos", "sin", "block_tables", "kv_rows", "kv_block", "kv_slot", "attn_block_tables", "attn_seq_lens")
LIVE = ("h", "k_cache", "v_cache")      # what outlives a layer's stations: the residual stream and the K/V pool
IN_PLACE = ("k_cache", "v_cache")       # written in place; an override must return these same tensors


def _input_norm(c):
    return {"x": fused.rmsnorm(c["h"], c["w.input_norm"], c["cfg"].eps)}


def _qkv(c):
    cfg = c["cfg"]
    qd, kd = cfg.num_q_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim
    q, k, v = F.linear(c["x"], c["w.qkv"]).split([qd, kd, kd], dim=-1)
    return {"q": q, "k": k, "v": v}


def _qk_norm_rope(c):
    """q, k [B, 1, H*D] -> head-wise RMSNorm then rotary, as [B, H, 1, D]."""
    cfg, q, k = c["cfg"], c["q"], c["k"]
    B, S = q.shape[:2]
    q = fused.rmsnorm_rope(q.view(B, S, cfg.num_q_heads, cfg.head_dim), c["w.q_norm"], c["cos"], c["sin"], cfg.eps)
    k = fused.rmsnorm_rope(k.view(B, S, cfg.num_kv_heads, cfg.head_dim), c["w.k_norm"], c["cos"], c["sin"], cfg.eps)
    return {"q": q.transpose(1, 2), "k": k.transpose(1, 2)}


def _kv_append(c):
    cfg, v = c["cfg"], c["v"]
    B, S = v.shape[:2]
    v = v.view(B, S, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
    pool = types.SimpleNamespace(k=c["k_cache"], v=c["v_cache"])
    write_token(pool, c["block_tables"], c["kv_rows"], c["kv_block"], c["kv_slot"], c["k"], v)
    return {"k_cache": c["k_cache"], "v_cache": c["v_cache"]}


def _attention(c):
    from inference_server.models.paged_attention_kernel import paged_decode_attention
    return {"attn": paged_decode_attention(c["q"].squeeze(2), c["k_cache"], c["v_cache"], c["attn_block_tables"],
                                           c["attn_seq_lens"], scale=c["cfg"].scale)}


def _o_proj(c):
    attn = c["attn"]
    return {"a": F.linear(attn.reshape(attn.shape[0], 1, -1), c["w.o_proj"])}


def _add_norm(c):
    h, n = fused.add_rmsnorm(c["h"], c["a"], c["w.post_norm"], c["cfg"].eps)
    return {"h": h, "n": n}


def _router(c):
    n = c["n"]
    return {"logits": F.linear(n.reshape(-1, n.shape[-1]), c["w.router"])}


def _route(c):
    from inference_server.models import moe_kernel
    cfg, logits = c["cfg"], c["logits"]
    T = logits.shape[0]
    router = moe_kernel.route_fused if T <= qwen3_moe.ROUTE_FUSED_MAX_TOKENS else moe_kernel.route_multi
    sorted_ids, expert_ids, tok, pair_w = router(logits, cfg.num_experts, cfg.top_k, cfg.norm_topk_prob,
                                                 qwen3_moe.moe_block_m(T * cfg.top_k))
    return {"sorted_ids": sorted_ids, "expert_ids": expert_ids, "tok": tok, "pair_w": pair_w}


def _a8(w_scale, pairs: int) -> bool:
    """fp8 activations too (W8A8) from MOE_A8_MIN_PAIRS pairs, with fp8 weights."""
    return w_scale is not None and pairs >= qwen3_moe.MOE_A8_MIN_PAIRS


def _gate_up(c):
    from inference_server.models import moe_kernel
    n = c["n"]
    flat, M = n.reshape(-1, n.shape[-1]), c["tok"].numel()
    a, sa = moe_kernel.quant_rows(flat) if _a8(c["w.gate_up_scale"], M) else (flat, None)
    return {"hgu": moe_kernel.moe_gemm(a, c["tok"], c["w.gate_up"], c["sorted_ids"], c["expert_ids"], M,
                                       block_m=qwen3_moe.moe_block_m(M), w_scale=c["w.gate_up_scale"], a_scale=sa)}


def _silu_mul(c):
    return {"act": fused.silu_mul(c["hgu"])}


def _down(c):
    """The down GEMM scattering each pair's weighted row into its token's fp32 output row."""
    from inference_server.models import moe_kernel
    act, M = c["act"], c["tok"].numel()
    act, sact = moe_kernel.quant_rows(act) if _a8(c["w.down_scale"], M) else (act, None)
    out = torch.zeros(M // c["cfg"].top_k, c["w.down"].shape[1], dtype=torch.float32, device=act.device)
    moe_kernel.moe_gemm_scatter(act, c["w.down"], c["sorted_ids"], c["expert_ids"], M, c["tok"], c["pair_w"], out,
                                block_m=qwen3_moe.moe_block_m(M), w_scale=c["w.down_scale"], a_scale=sact)
    return {"y": out}


def _residual(c):
    return {"h": fused.add_cast(c["h"], c["y"])}


DECODE: tuple[Station, ...] = (
    Station("input_norm", ("h", "w.input_norm", "cfg"), ("x",), _input_norm),
    Station("qkv", ("x", "w.qkv", "cfg"), ("q", "k", "v"), _qkv),
    Station("qk_norm_rope", ("q", "k", "cos", "sin", "w.q_norm", "w.k_norm", "cfg"), ("q", "k"), _qk_norm_rope),
    Station("kv_append", ("k", "v", "k_cache", "v_cache", "block_tables", "kv_rows", "kv_block", "kv_slot", "cfg"),
            ("k_cache", "v_cache"), _kv_append),
    Station("attention", ("q", "k_cache", "v_cache", "attn_block_tables", "attn_seq_lens", "cfg"), ("attn",),
            _attention),
    Station("o_proj", ("attn", "w.o_proj"), ("a",), _o_proj),
    Station("add_norm", ("h", "a", "w.post_norm", "cfg"), ("h", "n"), _add_norm),
    Station("router", ("n", "w.router"), ("logits",), _router),
    Station("route", ("logits", "cfg"), ("sorted_ids", "expert_ids", "tok", "pair_w"), _route),
    Station("gate_up", ("n", "tok", "sorted_ids", "expert_ids", "w.gate_up", "w.gate_up_scale", "cfg"), ("hgu",),
            _gate_up),
    Station("silu_mul", ("hgu",), ("act",), _silu_mul),
    Station("down", ("act", "sorted_ids", "expert_ids", "tok", "pair_w", "w.down", "w.down_scale", "cfg"), ("y",),
            _down),
    Station("residual", ("h", "y"), ("h",), _residual),
)


def _weights(layer) -> dict:
    attn, mlp = layer.self_attn, layer.mlp
    gu, gu_s = mlp.experts.weights("gate_up_proj")
    dn, dn_s = mlp.experts.weights("down_proj")
    return {"w.input_norm": layer.input_layernorm.weight, "w.qkv": attn.qkv_weight, "w.q_norm": attn.q_norm.weight,
            "w.k_norm": attn.k_norm.weight, "w.o_proj": attn.o_proj.weight,
            "w.post_norm": layer.post_attention_layernorm.weight, "w.router": mlp.gate.weight,
            "w.gate_up": gu, "w.gate_up_scale": gu_s, "w.down": dn, "w.down_scale": dn_s}


def layer_weights(model, layer_index: int) -> dict:
    """The "w.*" entries of one loaded layer (QKV fused, experts as loaded: fp8 with scales or bf16 with None)."""
    return _weights(_layers(model)[layer_index])


def layer_config(model) -> LayerConfig:
    layer = _layers(model)[0]
    attn, mlp = layer.self_attn, layer.mlp
    return LayerConfig(attn.num_q_heads, attn.num_kv_heads, attn.head_dim, attn.scale, layer.input_layernorm.eps,
                       mlp.experts.num_experts, mlp.top_k, mlp.norm_topk_prob)


def _layers(model):
    return getattr(model, "model", model).layers     # Qwen3MoeForCausalLM or Qwen3MoeModel


def applies(model, h: torch.Tensor, paged_ctx) -> bool:
    """Whether this forward is a decode step the stations run: a decode context, one token per row, the fused kernels
    on, QKV fused and the Triton MoE path."""
    layer = model.layers[0]
    return (hasattr(paged_ctx, "step_inputs") and not getattr(paged_ctx, "is_prefill", False) and h.shape[1] == 1
            and fused._on(h) and hasattr(layer.self_attn, "qkv_weight")
            and h.shape[0] * layer.mlp.top_k <= qwen3_moe.MOE_KERNEL_MAX_ROWS)


def forward(model, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, paged_ctx) -> torch.Tensor:
    """Every layer of `model` (a Qwen3MoeModel) through the resolved plan; the residual stream out."""
    step = {"cos": cos, "sin": sin, "cfg": layer_config(model), **paged_ctx.step_inputs()}
    steps = plan()
    cap = region_capture.active()
    for i, layer in enumerate(model.layers):
        pool = paged_ctx.pools[i]
        c = {**step, **_weights(layer), "h": h, "k_cache": pool.k, "v_cache": pool.v}
        for _, _, run in (cap.steps(steps, i) if cap is not None and cap.wants(i) else steps):
            c.update(run(c))
        h = c["h"]
    return h


def interface(stations: tuple[Station, ...], first: str, last: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(inputs, writes, outputs) of a stretch: what it reads before writing, all it writes, and what of that a later
    station reads or outlives the layer (`LIVE`)."""
    names = [s.name for s in stations]
    if first not in names or last not in names or names.index(first) > names.index(last):
        raise ValueError(f"no stretch {first}..{last} in stations {names}")
    i, j = names.index(first), names.index(last) + 1
    inputs, written = [], []
    for s in stations[i:j]:
        inputs += [r for r in s.reads if r not in written and r not in inputs]
        written += [w for w in s.writes if w not in written]
    needed = {r for s in stations[j:] for r in s.reads} | set(LIVE)
    return tuple(inputs), tuple(written), tuple(w for w in written if w in needed)


def _guarded(key: tuple[str, str], fn: Callable, inputs, writes, outputs) -> Callable[[dict], dict]:
    """An override called with only its stretch's inputs; what it returns checked against the stretch's writes."""
    allowed, required = set(writes), set(outputs)

    def run(c):
        out = fn({k: c[k] for k in inputs})
        if not required <= out.keys() <= allowed:
            raise ValueError(f"override {key} returned {sorted(out)}; it must write {sorted(required)} and may write "
                             f"only {sorted(allowed)}")
        for k in IN_PLACE:
            if k in out and out[k] is not c[k]:
                raise ValueError(f"override {key}: {k} must be written in place")
        return out
    return run


def resolve(stations: tuple[Station, ...], overrides: dict) -> tuple[tuple[str, str, Callable[[dict], dict]], ...]:
    """What one layer runs, in order, as (first station, last station, function): each station's own, or an
    override's in place of its stretch."""
    names = [s.name for s in stations]
    spans = []
    for key, fn in overrides.items():
        first, last = key
        inputs, writes, outputs = interface(stations, first, last)
        spans.append((names.index(first), names.index(last), _guarded(key, fn, inputs, writes, outputs), key))
    spans.sort(key=lambda s: s[0])
    for a, b in zip(spans, spans[1:]):
        if b[0] <= a[1]:
            raise ValueError(f"overrides {a[3]} and {b[3]} overlap")
    own = [(s.name, s.name, s.fn) for s in stations]
    out, i = [], 0
    for start, end, run, key in spans:
        out += own[i:start] + [(*key, run)]
        i = end + 1
    return tuple(out + own[i:])


def overrides(package: str = "inference_server.models.regions") -> dict:
    """Every module's OVERRIDES in `package`, merged; one stretch may be overridden once."""
    found: dict = {}
    pkg = importlib.import_module(package)
    for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        mod = importlib.import_module(f"{package}.{m.name}")
        for key, fn in getattr(mod, "OVERRIDES", {}).items():
            if tuple(key) in found:
                raise ValueError(f"stretch {tuple(key)} overridden twice (in {m.name})")
            found[tuple(key)] = fn
    return found


@functools.lru_cache(maxsize=None)
def plan() -> tuple[tuple[str, str, Callable[[dict], dict]], ...]:
    """DECODE with the overrides of models/regions/, resolved at the first decode forward (the warmup before CUDA
    graphs are captured, so graphs bake it in) and never again."""
    return resolve(DECODE, overrides())


def functional(stations: tuple[Station, ...] = DECODE) -> tuple[Station, ...]:
    """The stations as pure functions, for the lab: one that writes an entry in place gets a copy of it to write."""
    def pure(s: Station) -> Station:
        inplace = [n for n in s.writes if n in IN_PLACE]
        if not inplace:
            return s
        return replace(s, fn=lambda c: s.fn({**c, **{n: c[n].clone() for n in inplace}}))
    return tuple(pure(s) for s in stations)
