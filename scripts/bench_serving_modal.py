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

SLO_TTFT_MS, SLO_TPOT_MS = 200.0, 50.0
POOL_SIZE = 64                       # bounded distinct prompts (PrefixCache has no eviction → no leak)
PROMPT_MU, PROMPT_SIGMA = 5.48, 0.75  # ShareGPT-ish lognormal: ~240 prompt tokens
OUTPUT_MU, OUTPUT_SIGMA = 5.01, 0.65  # ~150 output tokens

# Env baked into the image (Modal re-imports this module in-container, without the shell env).
# BENCH_* MUST be here too or the container falls back to defaults (the compile-off smoke bug).
_env = {
    "BACKEND": "custom-cuda", "MODEL_NAME": MODEL, "BENCH_GPU": GPU, "BENCH_MODEL": MODEL,
    "BENCH_RATES": os.environ.get("BENCH_RATES", "1,2,3,4,6,8"),
    "BENCH_DURATION": os.environ.get("BENCH_DURATION", "30"),
    "BENCH_WARMUP": os.environ.get("BENCH_WARMUP", "4"),
    "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "256"),
    "PREFILL_MODE": "batched",
    "CUSTOM_BACKEND_COMPILE": COMPILE,
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
hf_secret = modal.Secret.from_name("huggingface-secret")


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * (k - f)


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def sweep():
    import asyncio
    import random
    import time

    from inference_server.backends import create_backend
    from inference_server.config import settings
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
        return (t0, stamps) if stamps else None

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

    async def main():
        rng = random.Random(0)
        sched = ContinuousBatchScheduler(
            backend, max_batch_size=settings.max_batch_size,
            max_active_kv_tokens=settings.max_active_kv_tokens,
            max_queue_size=4096,   # large: past the knee, overload shows as LATENCY not rejection
            prefill_mode="batched", prefill_chunk_size=256,
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
            for rate in RATES:
                collected: list = []
                t_start = time.perf_counter()
                await run_rate(sched, rate, DURATION, rng, collected)
                window = time.perf_counter() - t_start
                ttfts, tpots, tok = [], [], 0
                for t0, stamps in collected:
                    ttfts.append((stamps[0] - t0) * 1000)
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
                print(f"[serving] rate={rate:>5} reqs={row['reqs']:>4} tok/s={row['tok_s']:>7} "
                      f"TTFT={row['ttft_p50']}/{row['ttft_p95']}ms TPOT={row['tpot_p50']}/"
                      f"{row['tpot_p95']}ms SLO={'ok' if row['within_slo'] else 'X'}", flush=True)
                await asyncio.sleep(2)
        finally:
            await sched.stop()
        return rows

    return asyncio.run(main())


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path

    rows = sweep.remote()
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
