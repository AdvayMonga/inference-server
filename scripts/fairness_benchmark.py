"""Two-session fairness benchmark: FCFS vs Fair scheduling.

Workload: session A floods many requests, session B sends one mid-stream.
Measures B's TTFT (time-to-first-token) under each policy.

Expected: under FCFS, B waits behind A's tail. Under Fair, B is admitted
within ~1 admission cycle of arrival.
"""

import asyncio
import time

from inference_server.backends.torch_backend import TorchBackend
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.scheduling_policy import create_scheduling_policy
from inference_server.tokenizer import Tokenizer

MODEL = "google/gemma-4-E2B-it"
A_FLOOD_COUNT = 8        # number of requests session A submits
A_MAX_TOKENS = 30
B_MAX_TOKENS = 16
MAX_BATCH_SIZE = 4
B_DELAY_SECONDS = 0.4    # how long after A's flood before B arrives


async def submit_streaming(sched, tok, prompt, max_tokens, session_id):
    """Submit a streaming request; return (ttft_ms, total_ms, generated_tokens)."""
    loop = asyncio.get_running_loop()
    ids = tok.encode_chat(prompt, thinking=False)
    token_queue: asyncio.Queue = asyncio.Queue()
    req = ScheduledRequest(
        token_ids=ids, max_tokens=max_tokens, session_id=session_id,
        future=loop.create_future(), token_queue=token_queue,
    )
    t0 = time.perf_counter()
    sched.enqueue(req)
    first_tok = await token_queue.get()
    ttft = time.perf_counter() - t0
    generated = [first_tok]
    while True:
        tok_id = await token_queue.get()
        if tok_id is None:
            break
        generated.append(tok_id)
    total = time.perf_counter() - t0
    await req.future  # ensure fully resolved
    return ttft, total, generated


async def submit_blocking(sched, tok, prompt, max_tokens, session_id):
    """Submit a non-streaming request; return total_ms."""
    loop = asyncio.get_running_loop()
    ids = tok.encode_chat(prompt, thinking=False)
    req = ScheduledRequest(
        token_ids=ids, max_tokens=max_tokens, session_id=session_id,
        future=loop.create_future(),
    )
    t0 = time.perf_counter()
    sched.enqueue(req)
    await req.future
    return time.perf_counter() - t0


async def run_one(backend, tok, policy_name):
    print(f"\n=== Policy: {policy_name} ===")
    sched = ContinuousBatchScheduler(
        backend, max_batch_size=MAX_BATCH_SIZE, max_queue_size=1000,
        policy=create_scheduling_policy(policy_name),
    )
    sched.start()

    a_prompt = "Write a short paragraph about the ocean."
    b_prompt = "What is 2+2?"

    # Kick off all A requests immediately.
    a_tasks = [
        asyncio.create_task(submit_blocking(sched, tok, a_prompt, A_MAX_TOKENS, "A"))
        for _ in range(A_FLOOD_COUNT)
    ]
    # Let A saturate the batch before B arrives.
    await asyncio.sleep(B_DELAY_SECONDS)
    t_b_arrival = time.perf_counter()
    b_task = asyncio.create_task(submit_streaming(sched, tok, b_prompt, B_MAX_TOKENS, "B"))

    b_ttft, b_total, _ = await b_task
    a_durations = await asyncio.gather(*a_tasks)

    print(f"  B TTFT:           {b_ttft*1000:7.0f} ms")
    print(f"  B total:          {b_total*1000:7.0f} ms")
    print(f"  A reqs completed: {len(a_durations)}")
    print(f"  A median total:   {sorted(a_durations)[len(a_durations)//2]*1000:7.0f} ms")
    print(f"  A max total:      {max(a_durations)*1000:7.0f} ms")
    print(f"  scheduler stats:  {sched.stats()}")

    await sched.stop()
    return b_ttft


async def main():
    backend = TorchBackend(device="mps")
    print(f"Loading {MODEL}...")
    backend.load_model(MODEL)
    cache = CacheManager(num_blocks=256, block_size=16, eviction_policy="lru")
    backend.set_cache_adapter(cache)
    tok = Tokenizer(MODEL, 8192)

    fcfs_ttft = await run_one(backend, tok, "fcfs")
    fair_ttft = await run_one(backend, tok, "fair")

    print("\n=== Comparison ===")
    print(f"  B TTFT under FCFS: {fcfs_ttft*1000:7.0f} ms")
    print(f"  B TTFT under Fair: {fair_ttft*1000:7.0f} ms")
    speedup = fcfs_ttft / fair_ttft if fair_ttft > 0 else float("inf")
    print(f"  Fair speedup for B: {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
