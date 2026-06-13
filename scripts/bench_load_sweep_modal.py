"""Concurrency sweep of OUR engine (A10G) — our column of the head-to-head, plus an owned HF
baseline column. Closed-loop users hammer the scheduler at each concurrency N; we report
p50/p95/p99 TTFT/TPOT + aggregate token throughput.

Two backends, one container each (a full model each won't co-fit):
  - custom-cuda : hand-written Gemma 4 forward + paged KV + Triton kernel + CUDA graph + chunked prefill
  - torch       : HF AutoModelForCausalLM (sdpa) under the SAME scheduler — isolates what the custom
                  backend bought us (same model/hardware/scheduler).

Decode-bound workload (60-token prompt, 100 max-tokens), same regime as benchmarks/short_*. Uses
a small bounded set of distinct prompts (cycled) — NOT unique-per-request: the PrefixCache has no
eviction, so unbounded unique prompts leak blocks until the pool exhausts. Bounded → prefill is
cache-served after warmup (both backends cache, so the A/B stays fair); the sweep measures decode
throughput + TPOT under concurrency, which is what the custom backend optimizes.

    venv/bin/modal run --detach scripts/bench_load_sweep_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"CUSTOM_BACKEND_BLOCKS": "2048", "CUSTOM_BACKEND_SLIDING_BLOCKS": "1200",
          "KV_CACHE_NUM_BLOCKS": "4096", "MAX_ACTIVE_KV_TOKENS": "48000", "MAX_BATCH_SIZE": "32"})
    .add_local_python_source("inference_server")
)
app = modal.App("load-sweep", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
NS = [1, 4, 8, 16, 32]
PROMPT_TOKENS = 60
MAX_TOKENS = 100
DURATION = 15.0   # measured window per concurrency level (seconds)
WARMUP = 2.0


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def sweep(backend_name: str):
    import asyncio, time
    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.kv_cache.cache_manager import CacheManager
    from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest

    backend = create_backend(backend_name)
    backend.load_model(MODEL)
    # Replicate the server's cache wiring (no-op for custom; the cache-pool gate for torch).
    layer_shapes = backend.kv_shape_per_layer() if hasattr(backend, "kv_shape_per_layer") else None
    cache = CacheManager(
        num_blocks=settings.kv_cache_num_blocks, block_size=settings.kv_cache_block_size,
        eviction_policy=settings.eviction_policy, layer_shapes=layer_shapes,
        device=str(getattr(backend, "device", "cpu")), dtype=getattr(backend, "kv_dtype", None),
    )
    backend.set_cache_adapter(cache)
    # Bounded set of distinct prompts (cycled). Distinct lead token → distinct prefix-cache entry,
    # but only N_DISTINCT of them, so the cache (no eviction) stays bounded instead of leaking.
    N_DISTINCT = 8
    PROMPTS = [[50000 + i] + list(range(100, 100 + PROMPT_TOKENS)) for i in range(N_DISTINCT)]

    async def main():
        loop = asyncio.get_running_loop()
        counter = {"n": 0}

        async def drive(sched):
            counter["n"] += 1
            prompt = PROMPTS[counter["n"] % N_DISTINCT]
            q: asyncio.Queue = asyncio.Queue()
            req = ScheduledRequest(token_ids=prompt, max_tokens=MAX_TOKENS, session_id="u",
                                   future=loop.create_future(), token_queue=q)
            t0 = time.perf_counter()
            sched.enqueue(req)
            stamps = []
            while True:
                tok = await q.get()
                if tok is None:
                    break
                stamps.append(time.perf_counter())
            return t0, stamps

        async def user(sched, deadline, collect):
            while time.perf_counter() < deadline:
                t0, stamps = await drive(sched)
                if collect is not None and len(stamps) >= 1:  # None during warmup
                    collect.append((t0, stamps))

        results = {}
        for n in NS:
            sched = ContinuousBatchScheduler(
                backend, max_batch_size=settings.max_batch_size,
                max_active_kv_tokens=settings.max_active_kv_tokens,
            )
            sched.start()
            try:
                # warmup (untimed)
                wu = time.perf_counter() + WARMUP
                await asyncio.gather(*[user(sched, wu, None) for _ in range(n)])
                # measured window
                collected: list = []
                deadline = time.perf_counter() + DURATION
                t_start = time.perf_counter()
                await asyncio.gather(*[user(sched, deadline, collected) for _ in range(n)])
                window = time.perf_counter() - t_start
            finally:
                await sched.stop()

            ttfts, tpots, tok_total = [], [], 0
            for t0, stamps in collected:
                ttfts.append((stamps[0] - t0) * 1000)
                tok_total += len(stamps)
                if len(stamps) > 1:
                    tpots.append((stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000)
            results[n] = {
                "reqs": len(collected), "tok_s": round(tok_total / window, 1),
                "ttft_p50": round(_pct(ttfts, .50)), "ttft_p95": round(_pct(ttfts, .95)),
                "ttft_p99": round(_pct(ttfts, .99)),
                "tpot_p50": round(_pct(tpots, .50), 1), "tpot_p95": round(_pct(tpots, .95), 1),
            }
        return results

    return asyncio.run(main())


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path

    import os as _os
    # One backend per invocation by default (two sequential model loads = a long run that the
    # local Modal client tends to drop). BENCH_BACKENDS overrides. "cuda" = HF baseline.
    backends = _os.environ.get("BENCH_BACKENDS", "custom-cuda,cuda").split(",")
    all_results = {}
    for b in backends:
        try:
            all_results[b] = sweep.remote(b)
        except Exception as e:
            print(f"[{b}] FAILED: {type(e).__name__}: {str(e)[:300]}")

    out_dir = Path("benchmarks")
    for b, res in all_results.items():
        print(f"\n=== {b} ===")
        print(f"{'N':>3} {'reqs':>5} {'tok/s':>8} {'TTFT p50/p95/p99 (ms)':>24} {'TPOT p50/p95 (ms)':>18}")
        rows = []
        for n in NS:
            r = res[n]
            print(f"{n:>3} {r['reqs']:>5} {r['tok_s']:>8} "
                  f"{r['ttft_p50']:>7}/{r['ttft_p95']:>6}/{r['ttft_p99']:>6}   "
                  f"{r['tpot_p50']:>8}/{r['tpot_p95']:>7}")
            rows.append({"N": n, **r})
        fname = out_dir / f"sweep_{b.replace('-', '_')}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N", "reqs", "tok_s", "ttft_p50", "ttft_p95",
                                              "ttft_p99", "tpot_p50", "tpot_p95"])
            w.writeheader()
            w.writerows(rows)
        print(f"  saved {fname}")
