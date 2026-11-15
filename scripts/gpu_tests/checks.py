"""The CUDA correctness checks, as plain functions any CUDA box can run.

`tune_triton_launch.py` and the environment's GPU VM (from BlameGraph: `python -m lab.vm run -- python scripts/gpu_tests/checks.py`) call these, so
the gate is one set of checks with two launchers rather than two copies that drift. Each check
returns `(passed, detail)` and imports torch lazily: this module must import on a laptop with
no CUDA and no triton, so the checks can be listed and dry-run from a laptop.
"""

from __future__ import annotations

import time
from typing import Callable

TOL = 5e-2


def _reference(q, k_pool, v_pool, block_tables, seq_lens, scale):
    """Pure-torch paged attention: gather each seq's real K/V, softmax over all its keys."""
    import torch

    N, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    bs = k_pool.shape[2]
    group = Hq // num_kv_heads
    out = torch.empty(N, Hq, D, dtype=torch.float32, device=q.device)
    for i in range(N):
        L = int(seq_lens[i].item())
        nb = (L + bs - 1) // bs
        ks, vs = [], []
        for b in range(nb):
            blk = int(block_tables[i, b].item())
            ks.append(k_pool[blk])  # [num_kv_heads, bs, D]
            vs.append(v_pool[blk])
        K = torch.cat(ks, dim=1)[:, :L, :].float()  # [num_kv_heads, L, D]
        V = torch.cat(vs, dim=1)[:, :L, :].float()
        K = K.repeat_interleave(group, dim=0)        # [Hq, L, D]
        V = V.repeat_interleave(group, dim=0)
        qi = q[i].float()                            # [Hq, D]
        scores = (qi[:, None, :] * K).sum(-1) * scale  # [Hq, L]
        attn = torch.softmax(scores, dim=-1)
        out[i] = (attn[:, :, None] * V).sum(dim=1)   # [Hq, D]
    return out


def paged_decode_parity() -> tuple[bool, str]:
    """Decode kernel vs the torch reference: Qwen3-MoE's head_dim 128 at its GQA 8:1 and at one KV head."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_decode_attention

    torch.manual_seed(0)
    results = []
    for D, num_kv_heads, Hq in [(128, 4, 32), (128, 1, 8)]:
        N, num_blocks, bs = 5, 64, 16
        max_blocks = 8
        seq_lens = torch.tensor([1, 16, 17, 40, 100][:N], dtype=torch.int32, device="cuda")
        seq_lens = torch.clamp(seq_lens, max=max_blocks * bs)
        k_pool = torch.randn(num_blocks, num_kv_heads, bs, D, dtype=torch.bfloat16, device="cuda")
        v_pool = torch.randn_like(k_pool)
        # distinct physical blocks per (seq, logical block)
        bt = torch.arange(N * max_blocks, dtype=torch.int32, device="cuda").reshape(N, max_blocks) % num_blocks
        q = torch.randn(N, Hq, D, dtype=torch.bfloat16, device="cuda")

        got = paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        ref = _reference(q, k_pool, v_pool, bt, seq_lens, scale=1.0).to(got.dtype)
        diff = (got.float() - ref.float()).abs().max().item()
        results.append((D, num_kv_heads, Hq, diff))
    ok = all(diff < TOL for *_, diff in results)
    return ok, " ".join(f"D={D} kvh={kvh} Hq={Hq} max|diff|={d:.2e}" for D, kvh, Hq, d in results)


def paged_decode_no_recompile() -> tuple[bool, str]:
    """Vary block-count widely: with the runtime loop bound only the first call per head_dim
    compiles. A constexpr bound would recompile per distinct count → ~1s spikes → the
    mixed-sweep stall."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_decode_attention

    torch.manual_seed(0)
    D, num_kv_heads, Hq, bs = 128, 4, 32, 16
    num_blocks = 4096
    k_pool = torch.randn(num_blocks, num_kv_heads, bs, D, dtype=torch.bfloat16, device="cuda")
    v_pool = torch.randn_like(k_pool)
    timings, slow = [], []
    for nb in [2, 8, 20, 40, 64, 96, 128]:  # distinct block-counts (would each recompile if constexpr)
        max_blocks = nb
        seq_lens = torch.tensor([nb * bs], dtype=torch.int32, device="cuda")
        bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").reshape(1, max_blocks) % num_blocks
        q = torch.randn(1, Hq, D, dtype=torch.bfloat16, device="cuda")
        paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        timings.append(f"{nb}:{dt:.1f}ms")
        if dt > 100:
            slow.append((nb, dt))
    return not slow, "block_count:call " + " ".join(timings)


def paged_prefill_parity() -> tuple[bool, str]:
    """Prefill kernel vs reference at head_dim 64 and 128 (Qwen3-MoE's untiled path)."""
    results = [_prefill_parity_at(D) for D in (64, 128)]
    return all(ok for ok, _ in results), " | ".join(f"D={D}: {msg}" for D, (_, msg) in zip((64, 128), results))


def _prefill_parity_at(D: int) -> tuple[bool, str]:
    """Prefill kernel (multi-query, no gather) vs a gather + per-query causal-softmax reference:
    varied prefix/suffix lengths incl. a cold row and a 1-token cache-hit row, GQA 8:1."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_prefill_attention

    torch.manual_seed(0)
    dev = "cuda"
    Hq, num_kv, bs = 8, 1, 16               # GQA 8:1 (kernel handles any D)
    prefix_lens = [20, 5, 33, 0]            # incl. a cold row (no prefix)
    suffix_lens = [4, 1, 7, 6]              # incl. a 1-token (cache-hit) row
    N = len(prefix_lens)
    Sq = max(suffix_lens)
    totals = [p + s for p, s in zip(prefix_lens, suffix_lens)]
    max_blocks = (max(totals) + bs - 1) // bs

    num_blocks = N * max_blocks + 1
    k_pool = torch.randn(num_blocks, num_kv, bs, D, device=dev)
    v_pool = torch.randn(num_blocks, num_kv, bs, D, device=dev)
    bt = torch.zeros(N, max_blocks, dtype=torch.int32, device=dev)

    # Per-seq full KV (prefix+suffix), scattered into blocks i*max_blocks + b.
    full_k = [torch.randn(totals[i], num_kv, D, device=dev) for i in range(N)]
    full_v = [torch.randn(totals[i], num_kv, D, device=dev) for i in range(N)]
    for i in range(N):
        for b in range((totals[i] + bs - 1) // bs):
            blk = i * max_blocks + b
            bt[i, b] = blk
            for s in range(bs):
                pos = b * bs + s
                if pos < totals[i]:
                    k_pool[blk, :, s, :] = full_k[i][pos]
                    v_pool[blk, :, s, :] = full_v[i][pos]

    q = torch.randn(N, Sq, Hq, D, device=dev)
    pl = torch.tensor(prefix_lens, dtype=torch.int32, device=dev)
    sl = torch.tensor(suffix_lens, dtype=torch.int32, device=dev)
    out = paged_prefill_attention(q, k_pool, v_pool, bt, pl, sl, scale=1.0)

    maxdiff = 0.0
    for i in range(N):
        fk, fv = full_k[i][:, 0, :], full_v[i][:, 0, :]   # [total, D] (num_kv=1)
        for j in range(suffix_lens[i]):
            qpos = prefix_lens[i] + j
            scores = q[i, j] @ fk[:qpos + 1].T            # [Hq, qpos+1]
            ref = torch.softmax(scores.float(), dim=-1) @ fv[:qpos + 1].float()
            maxdiff = max(maxdiff, (out[i, j].float() - ref).abs().max().item())
    return maxdiff < 1e-2, (f"max|kernel - ref|={maxdiff:.2e} "
                            f"(N={N}, prefix={prefix_lens}, suffix={suffix_lens})")


def _stations_model(fp8: bool):
    """Two layers at Qwen3-30B-A3B's shapes, random weights, QKV fused, experts fp8 or bf16 (as the backend loads)."""
    import torch
    from transformers import Qwen3MoeConfig

    from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM

    cfg = Qwen3MoeConfig(vocab_size=1024, hidden_size=2048, num_hidden_layers=2, num_attention_heads=32,
                         num_key_value_heads=4, head_dim=128, num_experts=128, num_experts_per_tok=8,
                         moe_intermediate_size=768, norm_topk_prob=True, tie_word_embeddings=False,
                         rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Qwen3MoeForCausalLM.from_config(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.1 * torch.randn_like(p) if name.endswith("norm.weight") else 0.02 * torch.randn_like(p))
    for layer in model.model.layers:
        layer.self_attn.fuse_qkv()
        if fp8:
            layer.mlp.experts.quantize_fp8()
    return model.eval()


def _noop_override(first: str, last: str):
    """An override that runs its stretch's own stations in order."""
    from inference_server.models import stations
    names = [s.name for s in stations.DECODE]
    stretch = stations.DECODE[names.index(first):names.index(last) + 1]

    def fn(c):
        c, out = dict(c), {}
        for s in stretch:
            w = s.fn(c)
            c.update(w)
            out.update(w)
        return out
    return fn


def _deterministic_scatter(a, w, sorted_ids, expert_ids, M, tok, pair_w, out, block_m=16, w_scale=None, a_scale=None,
                           tile=None):
    """moe_gemm_scatter without atomics: each pair's row from moe_gemm, each token's top_k rows summed in a fixed order
    (pairs are numbered token * top_k + slot)."""
    from inference_server.models import moe_kernel
    y = moe_kernel.moe_gemm(a, None, w, sorted_ids, expert_ids, M, block_m=block_m, split_k=1, w_scale=w_scale,
                            a_scale=a_scale)
    out.add_((y.float() * pair_w[:, None]).view(out.shape[0], M // out.shape[0], -1).sum(1))


def decode_stations_identical() -> tuple[bool, str]:
    """The decode stations (models/stations.py) against the layer forward they replaced, and a no-op override of two
    stretches, at batch 1-1,024, short and long contexts, fp8 and bf16 experts, eager and as a captured CUDA graph (as
    the backend captures it). Identity: every variant on a deterministic MoE down scatter (the engine's adds each
    token's expert rows with fp32 atomics, in no fixed order, so the layer forward differs from itself), where logits
    and K/V pools must be bit-identical, the layer forward's own repeat included. Time: each variant's graph replay on
    the engine's own kernels."""
    import types

    import torch

    from inference_server.backends.custom_torch_backend import _GraphCtx
    from inference_server.models import moe_kernel, stations

    bs, failures, rows = 16, [], []
    old = (lambda *a: False, stations.plan)
    variants = {
        "layers": old, "layers#2": old,
        "stations": (stations.applies, stations.plan),
        "override": (stations.applies, lambda: stations.resolve(stations.DECODE, {
            ("input_norm", "attention"): _noop_override("input_norm", "attention"),
            ("router", "down"): _noop_override("router", "down")})),
    }
    scatter = moe_kernel.moe_gemm_scatter

    def same(a, b) -> bool:
        return all(torch.equal(x, y) for x, y in [(a[0], b[0]), *zip(a[1], b[1]), (a[2], b[2]), *zip(a[3], b[3])])
    for fp8 in (True, False):
        model = _stations_model(fp8)
        attn = model.model.layers[0].self_attn
        for B, ctx_len in [(1, 32), (1, 8192), (2, 32), (4, 4096), (8, 32), (16, 2048), (64, 32), (64, 2048),
                           (256, 32), (256, 1024), (512, 32), (1024, 32)]:
            g = torch.Generator(device="cuda").manual_seed(B * 7919 + ctx_len)
            cols = (ctx_len + 1 + bs - 1) // bs
            seq_lens = torch.randint(1, ctx_len + 1, (B,), device="cuda", generator=g)
            bt = torch.randperm(B * cols, device="cuda", generator=g).view(B, cols)
            tokens = torch.randint(0, 1024, (B, 1), device="cuda", generator=g)
            kv0 = [torch.randn(B * cols, attn.num_kv_heads, bs, attn.head_dim, device="cuda", dtype=torch.bfloat16,
                               generator=g) for _ in range(4)]
            pools = [types.SimpleNamespace(k=kv0[0].clone(), v=kv0[1].clone()),
                     types.SimpleNamespace(k=kv0[2].clone(), v=kv0[3].clone())]
            ctx = _GraphCtx(pools, seq_lens, bt, bs)
            pos = seq_lens.view(B, 1)

            def reset():
                for p, k, v in zip(pools, kv0[0::2], kv0[1::2]):
                    p.k.copy_(k)
                    p.v.copy_(v)

            got, ms = {}, {}
            for deterministic in (True, False):
                for name, (applies, plan) in variants.items():
                    if not deterministic and name == "layers#2":
                        continue
                    stations.applies, stations.plan = applies, plan
                    moe_kernel.moe_gemm_scatter = _deterministic_scatter if deterministic else scatter
                    try:
                        with torch.no_grad():
                            reset()
                            eager = model(tokens, position_ids=pos, paged_ctx=ctx)
                            eager_kv = [t.clone() for p in pools for t in (p.k, p.v)]
                            s = torch.cuda.Stream()
                            s.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s):
                                for _ in range(2):
                                    model(tokens, position_ids=pos, paged_ctx=ctx)
                            torch.cuda.current_stream().wait_stream(s)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                out = model(tokens, position_ids=pos, paged_ctx=ctx)
                            reset()
                            graph.replay()
                            graphed, graphed_kv = out.clone(), [t.clone() for p in pools for t in (p.k, p.v)]
                            times = []
                            for _ in range(30):
                                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                a.record()
                                graph.replay()
                                b.record()
                                b.synchronize()
                                times.append(a.elapsed_time(b))
                            del graph
                    finally:
                        stations.applies, stations.plan = variants["stations"]
                        moe_kernel.moe_gemm_scatter = scatter
                    if deterministic:
                        got[name] = (eager, eager_kv, graphed, graphed_kv)
                    else:
                        ms[name] = sorted(times)[len(times) // 2]
            verdict = {n: same(got[n], got["layers"]) for n in ("layers#2", "stations", "override")}
            failures += [f"{n} fp8={fp8} B={B} ctx={ctx_len}" for n, ok in verdict.items() if not ok]
            rows.append(f"fp8={fp8} B={B} ctx={ctx_len}: identical " +
                        ", ".join(f"{n} {'yes' if ok else 'NO'}" for n, ok in verdict.items()) + "; ms per replay " +
                        " / ".join(f"{n} {ms[n]:.3f}" for n in ("layers", "stations", "override")))
        del model
        torch.cuda.empty_cache()
    print("\n".join(rows))
    return not failures, "bit-identical everywhere" if not failures else "differ: " + "; ".join(failures)


def stations_reference_close() -> tuple[bool, str]:
    """stations.reference() on fp32 copies against the stations on the real kernels (bf16), one layer at
    Qwen3-30B-A3B's shapes, fp8 and bf16 experts, batch 1-1,024 (both routers, bf16 and fp8 activations): the relative
    error of the residual out and the K/V pool, under 2%."""
    import types

    import torch

    from inference_server.backends.custom_torch_backend import _GraphCtx
    from inference_server.models import stations

    bs, worst, rows = 16, 0.0, []
    for fp8 in (True, False):
        model = _stations_model(fp8)
        attn = model.model.layers[0].self_attn
        for B in (1, 8, 64, 256, 1024):
            g = torch.Generator(device="cuda").manual_seed(B)
            cols = (256 + 1 + bs - 1) // bs
            seq_lens = torch.randint(1, 257, (B,), device="cuda", generator=g)
            bt = torch.randperm(B * cols, device="cuda", generator=g).view(B, cols)
            k, v = (torch.randn(B * cols, attn.num_kv_heads, bs, attn.head_dim, device="cuda", dtype=torch.bfloat16,
                                generator=g) for _ in range(2))
            ctx = _GraphCtx([types.SimpleNamespace(k=k, v=v)], seq_lens, bt, bs)
            cos, sin = (t.to(torch.bfloat16) for t in model.model.rope(seq_lens.view(B, 1)))
            h = torch.randn(B, 1, 2048, device="cuda", dtype=torch.bfloat16, generator=g)
            c = {**ctx.step_inputs(), "cos": cos, "sin": sin, "cfg": stations.layer_config(model),
                 **stations.layer_weights(model, 0), "h": h, "k_cache": k, "v_cache": v}
            with torch.no_grad():
                real, ref = dict(c), {n: x.float() if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                                      for n, x in c.items()}
                for s in stations.functional():
                    real.update(s.fn(real))
                for s in stations.reference():
                    ref.update(s.fn(ref))
            errs = {n: ((ref[n] - real[n].float()).norm() / ref[n].norm()).item() for n in stations.LIVE}
            worst = max(worst, *errs.values())
            rows.append(f"fp8={fp8} B={B}: " + ", ".join(f"{n} {e:.2e}" for n, e in errs.items()))
        del model
        torch.cuda.empty_cache()
    print("\n".join(rows))
    return worst < 2e-2, f"worst relative error {worst:.2e}"


def region_capture_replays() -> tuple[bool, str]:
    """CAPTURE_REGION under the real kernels: each sample, with the layer's weights, replays through its stretch's
    stations to its recorded writes (relative error < 1e-3: the MoE atomics add in no fixed order) and through
    stations.reference() in fp32 to within 2%, on the kernels' routing; on its own fp32 routing (near-tied experts
    chosen differently for a few tokens) its error is reported beside."""
    import json
    import os
    import tempfile
    import types
    from pathlib import Path

    import torch

    from inference_server.backends.custom_torch_backend import _GraphCtx
    from inference_server.models import region_capture, stations

    model, bs, B, rows, worst_real, worst_ref = _stations_model(True), 16, 64, [], 0.0, 0.0
    attn, names = model.model.layers[0].self_attn, [s.name for s in stations.DECODE]

    def rel(a, b) -> float:
        if not b.is_floating_point():
            return float(not torch.equal(a, b))
        a, b = a.float(), b.float()
        return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    for spec in ("kv_append:attention", "router:down", "input_norm:residual"):
        out = Path(tempfile.mkdtemp())
        g = torch.Generator(device="cuda").manual_seed(0)
        cols = (512 + 1 + bs - 1) // bs
        seq_lens = torch.randint(1, 513, (B,), device="cuda", generator=g)
        bt = torch.randperm(B * cols, device="cuda", generator=g).view(B, cols)
        pools = [types.SimpleNamespace(k=torch.randn(B * cols, attn.num_kv_heads, bs, attn.head_dim, device="cuda",
                                                     dtype=torch.bfloat16, generator=g)) for _ in range(2)]
        for p in pools:
            p.v = torch.randn_like(p.k)
        os.environ.update(CAPTURE_REGION=spec, CAPTURE_REGION_DIR=str(out))
        region_capture.active.cache_clear()
        try:
            with torch.no_grad():
                model(torch.randint(0, 1024, (B, 1), device="cuda", generator=g), position_ids=seq_lens.view(B, 1),
                      paged_ctx=_GraphCtx(pools, seq_lens, bt, bs))
            region_capture.active()._log.flush()
        finally:
            del os.environ["CAPTURE_REGION"], os.environ["CAPTURE_REGION_DIR"]
            region_capture.active.cache_clear()
        first, _, last = spec.partition(":")
        i, j = names.index(first), names.index(last) + 1
        calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
        for call in calls:
            sample = torch.load(out / "samples" / call["sample"], weights_only=True)
            c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in sample["reads"].items()}
            c.update(stations.layer_weights(model, call["layer"]), cfg=stations.LayerConfig(**c["cfg"]))
            up = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in c.items()}
            routing = ("sorted_ids", "expert_ids", "tok", "pair_w")
            with torch.no_grad():
                real, ref, routed = dict(c), dict(up), dict(up)
                for s in stations.functional()[i:j]:
                    real.update(s.fn(real))
                for s in stations.reference()[i:j]:
                    ref.update(s.fn(ref))
                    routed.update({k: v for k, v in s.fn(routed).items() if k not in routing}
                                  if s.name == "route" else s.fn(routed))
                    if s.name == "route":
                        routed.update({k: real[k] for k in routing})
            w = sample["writes"]
            e_real = max(rel(real[n], w[n].cuda()) for n in w)
            e_ref = max(rel(routed[n], w[n].cuda()) for n in w)
            e_own = max(rel(ref[n], w[n].cuda()) for n in w)
            worst_real, worst_ref = max(worst_real, e_real), max(worst_ref, e_ref)
            rows.append(f"{spec} layer {call['layer']}: stations {e_real:.2e}, reference {e_ref:.2e} "
                        f"(on its own fp32 routing {e_own:.2e})")
    print("\n".join(rows))
    ok = len(rows) == 6 and worst_real < 1e-3 and worst_ref < 2e-2
    return ok, f"{len(rows)} samples; worst relative error: stations {worst_real:.2e}, reference {worst_ref:.2e}"


# Name → check. The gate runs all of them; each modal script runs its own subset.
CHECKS: dict[str, Callable[[], tuple[bool, str]]] = {
    "paged_decode_parity": paged_decode_parity,
    "paged_decode_no_recompile": paged_decode_no_recompile,
    "paged_prefill_parity": paged_prefill_parity,
    "decode_stations_identical": decode_stations_identical,
    "stations_reference_close": stations_reference_close,
    "region_capture_replays": region_capture_replays,
}


def main() -> int:
    """Run every check; exit 1 if any fails. `python scripts/gpu_tests/checks.py [name ...]`."""
    import sys
    names = sys.argv[1:] or list(CHECKS)
    failed = 0
    for name in names:
        ok, detail = CHECKS[name]()
        failed += not ok
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    print(f"{len(names) - failed}/{len(names)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
