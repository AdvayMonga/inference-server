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

import os

import modal

# Hardware + model are env-driven so the same script runs the A10G/E2B and A100/E4B sweeps.
# Read at module load (the @app.function decorator + image .env() bake these in at import).
GPU = os.environ.get("BENCH_GPU", "A10G")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E2B-it")


def _hw_tag() -> str:
    """Filename suffix keying a CSV to its GPU+model so different sweeps don't clobber.

    A10G/E2B (the original sweep) keeps the bare filename for continuity; others get a tag.
    """
    g = GPU.split("-")[0].lower()                       # a10g, a100, h100
    parts = MODEL.split("/")[-1].split("-")
    m = (parts[2] if len(parts) > 2 else parts[-1]).lower()  # e2b, e4b
    return "" if (g == "a10g" and m == "e2b") else f"_{g}_{m}"

# KV pools scale by GPU class: A100/H100 (80GB) has the headroom for far bigger pools, and
# E4B's KV is bigger (42 layers, head_dim 512 on full layers). Each var is env-overridable.
_BIG = GPU.startswith(("A100", "H100"))
_g = lambda k, big, small: os.environ.get(k, big if _BIG else small)
# BENCH_GPU/MODEL must be baked into the image env: the remote function reads MODEL as a
# module global, and Modal re-imports this module IN the container (where the local shell's
# env is absent) — so without this they'd silently fall back to the E2B/A10G defaults.
_kv_env = {
    "BENCH_GPU": GPU,
    "BENCH_MODEL": MODEL,
    # Forward backend knobs into the container too (same module-reimport reason as BENCH_*).
    "CUSTOM_BACKEND_COMPILE": os.environ.get("CUSTOM_BACKEND_COMPILE", "0"),
    "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/inductor",
    "CUSTOM_BACKEND_COMPILE_MODE": os.environ.get("CUSTOM_BACKEND_COMPILE_MODE", ""),
    "CUSTOM_BACKEND_EXPLAIN": os.environ.get("CUSTOM_BACKEND_EXPLAIN", "0"),
    "CUSTOM_BACKEND_QUANT": os.environ.get("CUSTOM_BACKEND_QUANT", ""),
    "CUSTOM_BACKEND_BLOCKS": _g("CUSTOM_BACKEND_BLOCKS", "8192", "2048"),
    "CUSTOM_BACKEND_SLIDING_BLOCKS": _g("CUSTOM_BACKEND_SLIDING_BLOCKS", "4096", "1200"),
    "KV_CACHE_NUM_BLOCKS": _g("KV_CACHE_NUM_BLOCKS", "16384", "4096"),
    "MAX_ACTIVE_KV_TOKENS": _g("MAX_ACTIVE_KV_TOKENS", "200000", "48000"),
    "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "32"),
    "WAVE_WINDOW_MULT": os.environ.get("WAVE_WINDOW_MULT", "4"),
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env(_kv_env)
    .add_local_python_source("inference_server")
)
app = modal.App("load-sweep", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Inductor artifacts survive across runs. torch.compile is the whole reason cold start
# costs ~21 min (each decode graph bucket is a fresh static shape = a fresh compile);
# without this every run pays it again.
inductor_cache = modal.Volume.from_name("inductor-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

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


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/inductor": inductor_cache},
              secrets=[hf_secret], timeout=5400)   # graph capture with compile on can take ~20min
def sweep(backend_name: str, prefill_mode: str = "monolithic"):
    import asyncio, time
    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.kv_cache.cache_manager import CacheManager
    from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest

    print(f"[sweep] backend={backend_name} MODEL={MODEL} GPU={GPU}", flush=True)
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
                prefill_mode=(None if prefill_mode == "monolithic" else prefill_mode),
                wave_window_mult=settings.wave_window_mult,
                prefill_chunk_size=256,
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
            r = results[n]  # print here too → survives a local client disconnect
            print(f"[{backend_name}] N={n:>2} reqs={r['reqs']:>4} tok/s={r['tok_s']:>7} "
                  f"TTFT={r['ttft_p50']}/{r['ttft_p95']}/{r['ttft_p99']}ms "
                  f"TPOT={r['tpot_p50']}/{r['tpot_p95']}ms", flush=True)
        return results

    inductor_cache.commit()   # persist compiled artifacts

    return asyncio.run(main())


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path

    # One backend per invocation by default (two sequential model loads = a long run that the
    # local Modal client tends to drop). BENCH_BACKENDS overrides. "cuda" = HF baseline.
    backends = os.environ.get("BENCH_BACKENDS", "custom-cuda,cuda").split(",")
    mode = os.environ.get("SWEEP_PREFILL_MODE", "monolithic")  # monolithic|batched|chunked
    all_results = {}
    for b in backends:
        try:
            all_results[b] = sweep.remote(b, mode)
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
        suffix = "" if mode == "monolithic" else f"_{mode}"
        # Backend-variant tag so compile/quant runs don't clobber the plain baseline CSV.
        variant = ""
        if os.environ.get("CUSTOM_BACKEND_COMPILE", "0") == "1":
            variant += "_compile"
        if os.environ.get("CUSTOM_BACKEND_QUANT", "").lower() == "int8":
            variant += "_int8"
        fname = out_dir / f"sweep_{b.replace('-', '_')}{suffix}{_hw_tag()}{variant}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["N", "reqs", "tok_s", "ttft_p50", "ttft_p95",
                                              "ttft_p99", "tpot_p50", "tpot_p95"])
            w.writeheader()
            w.writerows(rows)
        print(f"  saved {fname}")
