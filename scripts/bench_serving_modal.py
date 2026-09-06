"""Open-loop serving sweep of OUR engine, CO-LOCATED in Modal — the P3/P4 real run.

Runs the load generator INSIDE the container, calling the scheduler directly (like
bench_load_sweep_modal.py), so measured TTFT/TPOT is the engine's, NOT laptop<->Modal
internet RTT (which would swamp the p95 TTFT<200ms SLO). Difference from that script:
open-loop Poisson arrivals (load is an input rate, not fixed concurrency) so the queue
can grow and we find the saturation point + max throughput within SLO.

    venv/bin/modal run --detach scripts/bench_serving_modal.py            # full sweep
    BENCH_RATES=8,32 BENCH_DURATION=8 venv/bin/modal run scripts/bench_serving_modal.py  # smoke

SLO (PLAN.md P1): p95 TTFT < 200ms, p95 TPOT < 50ms/token.
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
# Rates are req/s of FULL ShareGPT-sized requests (~150 output tokens each), so each req is
# heavy — the knee sits at low single-digit req/s, not the dozens. Sweep low to bracket it.
RATES = [float(x) for x in os.environ.get("BENCH_RATES", "1,2,3,4,6,8").split(",")]
DURATION = float(os.environ.get("BENCH_DURATION", "30"))
WARMUP = float(os.environ.get("BENCH_WARMUP", "4"))
COMPILE = os.environ.get("CUSTOM_BACKEND_COMPILE", "1")
# Interleave two arms inside ONE container. Arms in separate containers are not
# comparable even with a shared run_group label: a tiled-prefill A/B showed TPOT p50
# 95.3 vs 46.4ms between arms differing only in a PREFILL flag — machine variation.
AB_TILED = os.environ.get("BENCH_AB_TILED", "0") == "1"

SLO_TTFT_MS, SLO_TPOT_MS = 200.0, 50.0
# Distinct prompts drawn from. NOTE: this is a prefix-cache-HIT benchmark at small values —
# with 64 prompts and a warm cache, after warmup almost every request hits with a tiny suffix,
# which flatters prefill work and hides cache-miss behaviour. Raise it (BENCH_POOL_SIZE) for a
# miss-heavy curve; scripts/bench_stress_modal.py covers the miss + overload case directly.
POOL_SIZE = int(os.environ.get("BENCH_POOL_SIZE", "64"))
PROMPT_MU, PROMPT_SIGMA = 5.48, 0.75  # ShareGPT-ish lognormal: ~240 prompt tokens
OUTPUT_MU, OUTPUT_SIGMA = 5.01, 0.65  # ~150 output tokens

# Env baked into the image (Modal re-imports this module in-container, without the shell env).
# BENCH_* MUST be here too or the container falls back to defaults (the compile-off smoke bug).
_env = {
    # Panels are emitted inside an ephemeral container with no git repo; carry the
    # launching side's sha in so the validity block can attribute the run.
    "RESEARCH_ENGINE_SHA": os.environ.get("RESEARCH_ENGINE_SHA", ""),
    "RESEARCH_ENGINE_DIRTY": os.environ.get("RESEARCH_ENGINE_DIRTY", ""),
    "RESEARCH_RUN_GROUP": os.environ.get("RESEARCH_RUN_GROUP", ""),
    "BACKEND": "custom-cuda", "MODEL_NAME": MODEL, "BENCH_GPU": GPU, "BENCH_MODEL": MODEL,
    "BENCH_RATES": os.environ.get("BENCH_RATES", "1,2,3,4,6,8"),
    "BENCH_DURATION": os.environ.get("BENCH_DURATION", "30"),
    "BENCH_WARMUP": os.environ.get("BENCH_WARMUP", "4"),
    "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "256"),
    "PREFILL_MODE": "batched",
    "BENCH_POOL_SIZE": os.environ.get("BENCH_POOL_SIZE", "64"),
    "WAVE_WINDOW_MULT": os.environ.get("WAVE_WINDOW_MULT", "4"),
    "CUSTOM_BACKEND_COMPILE": COMPILE,
    "BENCH_AB_TILED": os.environ.get("BENCH_AB_TILED", "0"),
    "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/inductor",
    "CUSTOM_BACKEND_PREFILL_GRAPH": os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "0"),
    "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
    "KV_CACHE_NUM_BLOCKS": "16384", "MAX_ACTIVE_KV_TOKENS": "200000",
}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env(_env)
    .add_local_python_source("inference_server")
)
app = modal.App("serving-sweep", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Inductor artifacts survive across runs. torch.compile is the whole reason cold start
# costs ~21 min (each decode graph bucket is a fresh static shape = a fresh compile);
# without this every run pays it again.
inductor_cache = modal.Volume.from_name("inductor-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/inductor": inductor_cache},
              secrets=[hf_secret], timeout=1800)
def sweep():
    import asyncio
    import random
    import time

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.research import harness as H
    from inference_server.kv_cache.cache_manager import CacheManager
    from inference_server.scheduler import ContinuousBatchScheduler, QueueFullError, ScheduledRequest

    print(f"[serving] MODEL={MODEL} GPU={GPU} batch={settings.max_batch_size} "
          f"compile={COMPILE} rates={RATES}", flush=True)
    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    layer_shapes = backend.kv_shape_per_layer() if hasattr(backend, "kv_shape_per_layer") else None
    cache = CacheManager(
        num_blocks=settings.kv_cache_num_blocks, block_size=settings.kv_cache_block_size,
        eviction_policy=settings.eviction_policy, layer_shapes=layer_shapes,
        device=str(getattr(backend, "device", "cpu")), dtype=getattr(backend, "kv_dtype", None),
    )
    backend.set_cache_adapter(cache)

    # Bounded prompt pool with ShareGPT-ish sampled lengths (token-id prompts, no tokenizer needed).
    r0 = random.Random(1234)
    pool = []
    for i in range(POOL_SIZE):
        p = int(min(2048, max(8, r0.lognormvariate(PROMPT_MU, PROMPT_SIGMA))))
        o = int(min(1024, max(8, r0.lognormvariate(OUTPUT_MU, OUTPUT_SIGMA))))
        ids = [40000 + i] + [100 + (j % 200) for j in range(p - 1)]  # unique lead → distinct cache entry
        pool.append((ids, o))

    async def one(sched, ids, max_tokens):
        q: asyncio.Queue = asyncio.Queue()
        req = ScheduledRequest(token_ids=ids, max_tokens=max_tokens, session_id="u",
                               future=asyncio.get_running_loop().create_future(), token_queue=q)
        t0 = time.perf_counter()
        try:
            sched.enqueue(req)
        except QueueFullError:
            return None
        stamps = []
        while True:
            tok = await q.get()
            if tok is None:
                break
            stamps.append(time.perf_counter())
        # req.admit_ts splits TTFT: queue-wait (admit - enqueue) vs prefill (first token - admit).
        return (t0, stamps, req.admit_ts) if stamps else None

    async def run_rate(sched, rate, duration, rng, collect):
        """Open-loop: fire on a Poisson schedule for `duration`s regardless of completions."""
        tasks, k = [], [0]

        async def fire():
            ids, o = pool[k[0] % POOL_SIZE]
            k[0] += 1
            res = await one(sched, ids, o)
            if res and collect is not None:
                collect.append(res)

        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            tasks.append(asyncio.create_task(fire()))
            await asyncio.sleep(rng.expovariate(rate))
        if tasks:
            await asyncio.wait(tasks, timeout=300.0)

    panels: list = []

    async def main():
        rng = random.Random(0)
        sched = ContinuousBatchScheduler(
            backend, max_batch_size=settings.max_batch_size,
            max_active_kv_tokens=settings.max_active_kv_tokens,
            max_queue_size=4096,   # large: past the knee, overload shows as LATENCY not rejection
            prefill_mode="batched", prefill_chunk_size=256,
            wave_window_mult=settings.wave_window_mult,
        )
        sched.start()
        rows = []
        try:
            # Warmup: single requests ONE AT A TIME, awaited to completion, so the multi-minute
            # lazy torch.compile (decode + K=1 prefill buckets) happens with an EMPTY queue. An
            # open-loop burst here piles hundreds of reqs behind the compile → the scheduler then
            # batch-prefills the whole backlog in one forward → 143GiB OOM (what killed the last run).
            for i in range(4):
                ids, o = pool[i % POOL_SIZE]
                await one(sched, ids, min(o, 16))
            # ABBA, not ABAB. Three identical back-to-back runs measured 1494 / 401 / 229 ms
            # — strictly decreasing — so whichever arm runs second is systematically favoured.
            # Alternating the pair order balances that out across replicates.
            if AB_TILED:
                plan = []
                for i, r in enumerate(RATES):
                    pair = [("baseline", False), ("treatment", True)]
                    if i % 2:
                        pair.reverse()
                    plan += [(r, n, t) for n, t in pair]
            else:
                plan = [(r, "", None) for r in RATES]
            for rate, arm_name, arm_tiled in plan:
                if arm_tiled is not None:
                    from inference_server.models import paged_attention_kernel as _K
                    _K._TILED_PREFILL = arm_tiled
                    # Start every arm from a COLD prefix cache. Running arms in sequence lets
                    # the cache warm between them, so the second arm measures a different
                    # workload — the validity gate caught exactly this (cache_miss_heavy vs
                    # mixed) and refused the comparison.
                    pc = getattr(backend, "prefix_cache", None)
                    if pc is not None:
                        backend.prefix_cache = type(pc)(
                            pools=backend.pools,
                            max_entries=settings.prefix_cache_max_entries,
                            max_block_fraction=settings.prefix_cache_block_fraction)
                collected: list = []
                t_start = time.perf_counter()
                await run_rate(sched, rate, DURATION, rng, collected)
                window = time.perf_counter() - t_start
                ttfts, tpots, tok = [], [], 0
                queue_ms, prefill_ms = [], []
                for t0, stamps, admit_ts in collected:
                    ttfts.append((stamps[0] - t0) * 1000)
                    if admit_ts:
                        queue_ms.append((admit_ts - t0) * 1000)
                        prefill_ms.append((stamps[0] - admit_ts) * 1000)
                    tok += len(stamps)
                    if len(stamps) > 1:
                        tpots.append((stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000)
                p95t, p95p = _pct(ttfts, .95), _pct(tpots, .95)
                row = {
                    "rate": rate, "reqs": len(collected), "tok_s": round(tok / window, 1),
                    "ttft_p50": round(_pct(ttfts, .50)), "ttft_p95": round(p95t),
                    "tpot_p50": round(_pct(tpots, .50), 1), "tpot_p95": round(p95p, 1),
                    "within_slo": bool(collected) and p95t < SLO_TTFT_MS and p95p < SLO_TPOT_MS,
                }
                rows.append(row)
                print(f"[serving] {arm_name:<9} rate={rate:>5} reqs={row['reqs']:>4} tok/s={row['tok_s']:>7} "
                      f"TTFT={row['ttft_p50']}/{row['ttft_p95']}ms TPOT={row['tpot_p50']}/"
                      f"{row['tpot_p95']}ms SLO={'ok' if row['within_slo'] else 'X'}", flush=True)
                # Padding waste only exists when prefill waves are WIDE. If the queue never
                # backs up, every arrival is its own K=1 wave and there is nothing to group.
                print(f"           wave sizes (K->count): {sched.stats().get('wave_sizes')}",
                      flush=True)
                print(f"           TTFT split — queue p50/p95={_pct(queue_ms, .50):.0f}/"
                      f"{_pct(queue_ms, .95):.0f}ms  prefill p50/p95={_pct(prefill_ms, .50):.0f}/"
                      f"{_pct(prefill_ms, .95):.0f}ms", flush=True)
                st = sched.stats()
                print(f"           active={st['active_size']} pending={st['pending_depth']} "
                      f"queue_hw={st['pending_high_water']} kv_blocked={st['kv_admit_blocked']} "
                      f"rejected={st['total_rejected']} expired={st['total_expired']} "
                      f"preempted={st['total_preempted']} iter_errors={st['total_iteration_errors']}",
                      flush=True)
                # Machine record for the research loop; the tables above stay for humans.
                cache_stats = (backend.prefix_cache.stats()
                               if getattr(backend, "prefix_cache", None) else {})
                cfg = {
                    "model": MODEL, "gpu": GPU, "max_batch_size": settings.max_batch_size,
                    "prefill_mode": "batched", "compile": COMPILE,
                    "prefill_graph": os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "0"),
                    "blocks": os.environ.get("CUSTOM_BACKEND_BLOCKS"),
                    "sliding_blocks": os.environ.get("CUSTOM_BACKEND_SLIDING_BLOCKS"),
                    "context_window": os.environ.get("CONTEXT_WINDOW"),
                    "rates": str(rate), "duration": DURATION, "pool_size": POOL_SIZE,
                    "max_queue_wait_s": settings.max_queue_wait_s,
                    "prefix_cache_impl": settings.prefix_cache_impl,
                    "wave_window_mult": settings.wave_window_mult,
                }
                panels.append(H.panel_from_stats(
                    H.build_validity(
                        "bench_serving", cfg,
                        n_samples=len(collected),
                        workload_regime=H.infer_regime(cache_stats.get("hit_rate"), POOL_SIZE),
                        stderr_value=H.stderr(ttfts),
                        concurrency_observed=st.get("pending_high_water"),
                        notes=f"open-loop Poisson, rate={rate}"
                              + (f", arm={arm_name}" if arm_name else ""),
                    ),
                    scheduler_stats=st, cache_stats=cache_stats,
                    tok_s_within_slo=row["tok_s"] if row["within_slo"] else None,
                    slo_ttft_ms=SLO_TTFT_MS, slo_tpot_ms=SLO_TPOT_MS,
                    ttft_p50=row["ttft_p50"], ttft_p95=row["ttft_p95"],
                    ttft_queue_p50=_pct(queue_ms, .50), ttft_queue_p95=_pct(queue_ms, .95),
                    ttft_prefill_p50=_pct(prefill_ms, .50),
                    ttft_prefill_p95=_pct(prefill_ms, .95),
                    tpot_p50=row["tpot_p50"], tpot_p95=row["tpot_p95"],
                    wall_s=window,
                ))
                panels[-1].validate()   # fail here, in the run, not on the way home

                pc = getattr(backend, "prefix_cache", None)
                if pc is not None:
                    c = pc.stats()
                    # Quote this with any prefill number: a small POOL_SIZE makes this a
                    # cache-HIT benchmark and the two regimes are not comparable without it.
                    print(f"           cache: hit_rate={c['hit_rate']} entries={c['entries']} "
                          f"evictions={c['evictions']} "
                          f"blocks={c['blocks_held']}/{c['max_blocks']} "
                          f"pool_util={c['pool_utilization']}", flush=True)
                await asyncio.sleep(2)
        finally:
            await sched.stop()
            inductor_cache.commit()   # persist compiled artifacts for the next run
        return rows

    rows = asyncio.run(main())
    return {"rows": rows, "panels": [p.to_dict() for p in panels]}


@app.local_entrypoint()
def main():
    import csv
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    result = sweep.remote()
    rows, panels = result["rows"], result["panels"]
    within = [r for r in rows if r["within_slo"]]
    best = max(within, key=lambda r: r["tok_s"], default=None)
    knee = next((r for r in rows if not r["within_slo"]), None)
    print("\n=== open-loop serving sweep (A100/E4B, co-located) ===")
    print(f"{'rate':>6} {'reqs':>5} {'tok/s':>8} {'TTFT p50/p95':>14} {'TPOT p50/p95':>14}  SLO")
    for r in rows:
        print(f"{r['rate']:>6} {r['reqs']:>5} {r['tok_s']:>8} "
              f"{r['ttft_p50']:>6}/{r['ttft_p95']:<7} {r['tpot_p50']:>6}/{r['tpot_p95']:<7}  "
              f"{'ok' if r['within_slo'] else 'X'}")
    if best:
        print(f"\nMAX THROUGHPUT WITHIN SLO: {best['tok_s']:.0f} tok/s @ rate {best['rate']}")
    if knee:
        print(f"SATURATION KNEE: SLO broken at rate {knee['rate']} "
              f"(TTFT p95 {knee['ttft_p95']}ms, TPOT p95 {knee['tpot_p95']}ms)")

    out = Path("benchmarks") / "serving_a100_e4b.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")

    # Persist the machine records locally — the container's filesystem is gone by now.
    from inference_server.research.schemas import Vitals
    runs = Path("runs")
    for d in panels:
        v = Vitals.from_dict(d)
        v.to_json(runs / f"{v.validity.run_id}.json")
    if panels:
        grp = panels[0]["validity"]["run_group"]
        print(f"wrote {len(panels)} panel(s) to runs/ (run_group={grp})")
        print("  compare arms with: python -m inference_server.research.loop attribute "
              f"runs/{panels[0]['validity']['run_id']}.json")
