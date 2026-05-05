"""KV-pressure backpressure validation.

Three scenarios:
  1. Active-KV soft-hold: many requests, total reservation > budget.
     Expected: kv_admit_blocked > 0, no rejects, all complete, reservation never
     exceeds budget.
  2. Active-KV hard-reject: a single request whose reservation > budget.
     Expected: that request is rejected with QueueFullError; small ones complete.
  3. Cache-pool hard-reject: a request whose blocks_needed > total cache blocks.
     Expected: rejected with QueueFullError citing cache capacity.
"""

import asyncio
import time

from inference_server.backends.mps import MPSBackend
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.scheduler import (
    ContinuousBatchScheduler,
    QueueFullError,
    ScheduledRequest,
)
from inference_server.scheduling_policy import create_scheduling_policy
from inference_server.tokenizer import Tokenizer

MODEL = "google/gemma-4-E2B-it"
TIGHT_BLOCKS = 32       # very small KV pool
BLOCK_SIZE = 16         # 32 * 16 = 512 token capacity total
ACTIVE_KV_BUDGET = 400  # tight active-KV cap (in tokens)


async def submit(sched, tok, prompt, max_tokens, sid):
    loop = asyncio.get_running_loop()
    ids = tok.encode_chat(prompt, thinking=False)
    req = ScheduledRequest(
        token_ids=ids, max_tokens=max_tokens, session_id=sid,
        future=loop.create_future(),
    )
    t0 = time.perf_counter()
    try:
        sched.enqueue(req)
        await req.future
        return ("ok", time.perf_counter() - t0, len(ids))
    except QueueFullError as e:
        return ("rejected", str(e), len(ids))
    except Exception as e:
        return ("error", repr(e), len(ids))


def _make_sched(backend):
    return ContinuousBatchScheduler(
        backend, max_batch_size=8, max_queue_size=100,
        policy=create_scheduling_policy("fcfs"),
        max_active_kv_tokens=ACTIVE_KV_BUDGET,
    )


async def scenario_active_soft_hold(backend, tok):
    print("\n=== Scenario 1: active-KV soft-hold ===")
    sched = _make_sched(backend)
    sched.start()
    # Each request: ~30-token prompt + 40 max_tokens ≈ 70 tokens reserved.
    # Budget=400 → ~5 concurrent admissions; 10 requests → must queue.
    prompt = "Briefly describe the planet Mars."
    max_observed_reserved = 0

    async def watcher():
        nonlocal max_observed_reserved
        while True:
            max_observed_reserved = max(max_observed_reserved, sched.stats()["active_kv_reserved"])
            await asyncio.sleep(0.01)

    w = asyncio.create_task(watcher())
    tasks = [asyncio.create_task(submit(sched, tok, prompt, 40, f"s{i}")) for i in range(10)]
    results = await asyncio.gather(*tasks)
    w.cancel()
    s = sched.stats()
    ok = sum(1 for r in results if r[0] == "ok")
    rej = sum(1 for r in results if r[0] == "rejected")
    print(f"  ok={ok}  rejected={rej}")
    print(f"  kv_admit_blocked={s['kv_admit_blocked']}  budget={s['active_kv_budget']}")
    print(f"  max_observed_active_kv_reserved={max_observed_reserved}")
    await sched.stop()
    return s["kv_admit_blocked"], rej, max_observed_reserved


async def scenario_active_hard_reject(backend, tok):
    print("\n=== Scenario 2: active-KV hard-reject (one over-budget request) ===")
    sched = _make_sched(backend)
    sched.start()
    # max_tokens > budget → guaranteed reservation > budget
    huge = await submit(sched, tok, "hi", ACTIVE_KV_BUDGET + 100, "huge")
    smalls = await asyncio.gather(*[
        submit(sched, tok, "What is 2+2?", 8, f"sm{i}") for i in range(3)
    ])
    print(f"  huge: status={huge[0]} detail={huge[1]}")
    print(f"  smalls: {[r[0] for r in smalls]}")
    await sched.stop()
    return huge[0]


async def scenario_cache_hard_reject(backend, tok):
    print("\n=== Scenario 3: cache-pool hard-reject ===")
    # Use a much larger active budget so the active-KV gate doesn't fire first.
    sched = ContinuousBatchScheduler(
        backend, max_batch_size=8, max_queue_size=100,
        policy=create_scheduling_policy("fcfs"),
        max_active_kv_tokens=100_000,
    )
    sched.start()
    huge_max = TIGHT_BLOCKS * BLOCK_SIZE * 2  # exceeds total cache blocks
    huge = await submit(sched, tok, "hi", huge_max, "huge")
    smalls = await asyncio.gather(*[
        submit(sched, tok, "What is 2+2?", 8, f"sm{i}") for i in range(3)
    ])
    print(f"  huge: status={huge[0]} detail={huge[1]}")
    print(f"  smalls: {[r[0] for r in smalls]}")
    await sched.stop()
    return huge[0]


async def main():
    backend = MPSBackend()
    print(f"Loading {MODEL}...")
    backend.load_model(MODEL)
    cache = CacheManager(num_blocks=TIGHT_BLOCKS, block_size=BLOCK_SIZE, eviction_policy="lru")
    backend.set_cache_adapter(cache)
    tok = Tokenizer(MODEL, 8192)
    print(f"KV pool: {TIGHT_BLOCKS} blocks × {BLOCK_SIZE} tokens = {TIGHT_BLOCKS*BLOCK_SIZE} token cap")
    print(f"Active-KV budget: {ACTIVE_KV_BUDGET} tokens")

    blocked, rej, peak = await scenario_active_soft_hold(backend, tok)
    active_huge = await scenario_active_hard_reject(backend, tok)
    cache_huge = await scenario_cache_hard_reject(backend, tok)

    print("\n=== Verdict ===")
    print(f"  active soft-hold: blocks={blocked>0} rej={rej==0} bounded={peak<=ACTIVE_KV_BUDGET}")
    print(f"  active hard-reject: {active_huge == 'rejected'}")
    print(f"  cache hard-reject:  {cache_huge == 'rejected'}")


if __name__ == "__main__":
    asyncio.run(main())
