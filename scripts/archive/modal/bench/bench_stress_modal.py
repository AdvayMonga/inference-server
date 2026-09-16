"""Overload + KV-pressure stress: does the engine degrade or does it break?

Everything else in benchmarks/ runs the engine BELOW saturation against 64 reused prompts and a
never-evicting prefix cache. That configuration cannot exercise the things most likely to take a
server down in production, so none of them were ever tested:

  * a prefix cache that grows without bound and starves the block pool
  * BlockPool.alloc() raising on the request path instead of applying backpressure
  * a scheduler iteration that throws and takes the whole worker thread with it
  * whether overload turns into REJECTIONS (graceful) or ERRORS/hangs (not)

This drives DISTINCT prompts (so the cache misses and must evict), at rates past the knee, into
a deliberately small block pool, and reports what the engine did about it.

  BENCH_HIT_RATE   fraction of requests that repeat an earlier prompt (default 0.3)
  BENCH_RATES      arrival rates to sweep (default 4,16,48)
  CUSTOM_BACKEND_BLOCKS / _SLIDING_BLOCKS  shrink to force pressure sooner

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/bench/bench_stress_modal.py
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
RATES = [float(x) for x in os.environ.get("BENCH_RATES", "4,16,48").split(",")]
DURATION = float(os.environ.get("BENCH_DURATION", "30"))
HIT_RATE = float(os.environ.get("BENCH_HIT_RATE", "0.3"))

_env = {
    # Panels are emitted inside an ephemeral container with no git repo; carry the
    # launching side's sha in so the validity block can attribute the run.
    "RESEARCH_ENGINE_SHA": os.environ.get("RESEARCH_ENGINE_SHA", ""),
    "RESEARCH_ENGINE_DIRTY": os.environ.get("RESEARCH_ENGINE_DIRTY", ""),
    "RESEARCH_RUN_GROUP": os.environ.get("RESEARCH_RUN_GROUP", ""),
    "BACKEND": "custom-cuda", "MODEL_NAME": MODEL, "BENCH_GPU": GPU, "BENCH_MODEL": MODEL,
    "BENCH_RATES": os.environ.get("BENCH_RATES", "4,16,48"),
    "BENCH_DURATION": os.environ.get("BENCH_DURATION", "30"),
    "BENCH_HIT_RATE": os.environ.get("BENCH_HIT_RATE", "0.3"),
    "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "64"),
    "PREFILL_MODE": "batched",
    "CUSTOM_BACKEND_COMPILE": "0",
    "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/inductor",
    "CUSTOM_BACKEND_PREFILL_GRAPH": os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "1"),
    # Deliberately small so KV pressure arrives in seconds rather than hours.
    "CUSTOM_BACKEND_BLOCKS": os.environ.get("CUSTOM_BACKEND_BLOCKS", "1200"),
    "CUSTOM_BACKEND_SLIDING_BLOCKS": os.environ.get("CUSTOM_BACKEND_SLIDING_BLOCKS", "800"),
    "KV_CACHE_NUM_BLOCKS": "16384",
    "MAX_ACTIVE_KV_TOKENS": os.environ.get("MAX_ACTIVE_KV_TOKENS", "120000"),
    "CONTEXT_WINDOW": "8192",
    "MAX_QUEUE_WAIT_S": os.environ.get("MAX_QUEUE_WAIT_S", "20"),
}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env(_env)
    .add_local_python_source("inference_server")
)
app = modal.App("kv-stress", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Inductor artifacts survive across runs. torch.compile is the whole reason cold start
# costs ~21 min (each decode graph bucket is a fresh static shape = a fresh compile);
# without this every run pays it again.
inductor_cache = modal.Volume.from_name("inductor-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/.cache/inductor": inductor_cache},
              secrets=[hf_secret], timeout=5400)
def run():
    import asyncio
    import random
    import time

    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.research import harness as H
    from inference_server.scheduler import (
        ContinuousBatchScheduler,
        QueueFullError,
        ScheduledRequest,
    )

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    pools = [p for p in backend.pools if p is not None]
    print(f"[stress] {settings.model_name} batch={settings.max_batch_size} "
          f"pools={len(pools)} blocks/pool={pools[0].num_blocks} "
          f"hit_rate={HIT_RATE}\n", flush=True)

    async def main():
        sched = ContinuousBatchScheduler(
            backend, max_batch_size=settings.max_batch_size,
            max_active_kv_tokens=settings.max_active_kv_tokens,
            max_queue_size=2048, prefill_mode="batched", prefill_chunk_size=256,
            max_queue_wait_s=settings.max_queue_wait_s,
        )
        sched.start()
        seen: list[list[int]] = []
        rng = random.Random(0)

        def make_prompt():
            if seen and rng.random() < HIT_RATE:
                return list(rng.choice(seen))                  # repeat -> prefix-cache HIT
            n = max(32, int(rng.lognormvariate(5.48, 0.75)))
            p = [rng.randrange(1000, 250000)] + [rng.randrange(1000, 250000) for _ in range(n - 1)]
            if len(seen) < 4000:
                seen.append(p)
            return p

        async def one(out):
            ids = make_prompt()
            req = ScheduledRequest(token_ids=ids, max_tokens=64, session_id=f"u{rng.randrange(999)}",
                                   future=asyncio.get_running_loop().create_future())
            t0 = time.perf_counter()
            try:
                toks = await sched.submit(req)
                out.append(("ok", (time.perf_counter() - t0) * 1000, len(toks)))
            except QueueFullError:
                out.append(("rejected", 0.0, 0))
            except Exception as e:                              # anything else is a REAL failure
                out.append((f"ERROR:{type(e).__name__}", 0.0, 0))

        try:
            for rate in RATES:
                out: list = []
                tasks = []
                t_end = time.perf_counter() + DURATION
                while time.perf_counter() < t_end:
                    tasks.append(asyncio.create_task(one(out)))
                    await asyncio.sleep(rng.expovariate(rate))
                await asyncio.gather(*tasks, return_exceptions=True)

                ok = [r for r in out if r[0] == "ok"]
                rej = [r for r in out if r[0] == "rejected"]
                err = [r for r in out if r[0].startswith("ERROR")]
                st = sched.stats()
                cs = backend.prefix_cache.stats()
                free = min(p.free_count for p in pools)
                lat = sorted(x[1] for x in ok)
                p95 = lat[int(.95 * (len(lat) - 1))] if lat else 0.0
                print(f"[rate {rate:>4}] ok={len(ok):<4} rejected={len(rej):<4} ERRORS={len(err):<3} "
                      f"e2e_p95={p95:>7.0f}ms", flush=True)
                print(f"            preempted={st['total_preempted']} "
                      f"expired={st['total_expired']} "
                      f"iter_errors={st['total_iteration_errors']} "
                      f"kv_admit_blocked={st['kv_admit_blocked']} queue_hw={st['pending_high_water']}",
                      flush=True)
                print(f"            cache: hit_rate={cs['hit_rate']} entries={cs['entries']} "
                      f"evictions={cs['evictions']} blocks_held={cs['blocks_held']}/{cs['max_blocks']} "
                      f"pool_free={free} util={cs['pool_utilization']}", flush=True)
                if err:
                    print(f"            !! {err[:3]}", flush=True)
                H.emit(H.panel_from_stats(
                    H.build_validity(
                        "bench_stress",
                        {"model": settings.model_name, "gpu": GPU,
                         "max_batch_size": settings.max_batch_size, "prefill_mode": "batched",
                         "compile": "0",
                         "prefill_graph": os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "1"),
                         "blocks": os.environ.get("CUSTOM_BACKEND_BLOCKS"),
                         "sliding_blocks": os.environ.get("CUSTOM_BACKEND_SLIDING_BLOCKS"),
                         "context_window": os.environ.get("CONTEXT_WINDOW"),
                         "rates": str(rate), "duration": DURATION, "pool_size": None,
                         "max_queue_wait_s": settings.max_queue_wait_s,
                         "prefix_cache_impl": settings.prefix_cache_impl,
                         "wave_window_mult": settings.wave_window_mult,
                         "hit_rate_target": HIT_RATE},
                        n_samples=len(ok), workload_regime=H.infer_regime(cs.get("hit_rate")),
                        stderr_value=H.stderr([x[1] for x in ok]),
                        concurrency_observed=st.get("pending_high_water"),
                        notes=f"overload sweep, rate={rate}, {len(err)} hard errors"),
                    scheduler_stats=st, cache_stats=cs,
                    ttft_p95=p95, wall_s=DURATION), label=f"stress rate={rate}")

                # The engine must still be alive and serving after each level.
                probe = []
                await one(probe)
                print(f"            alive-after-level: {probe[0][0]}\n", flush=True)
        finally:
            await sched.stop()
            inductor_cache.commit()   # persist compiled artifacts for the next run

    asyncio.run(main())
    print("[stress] completed without the worker dying", flush=True)
