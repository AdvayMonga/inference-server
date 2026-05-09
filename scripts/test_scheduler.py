"""Smoke test for ContinuousBatchScheduler — varying max_tokens, overlap, streaming."""

import asyncio
import time

from inference_server.backends.torch_backend import TorchBackend
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.tokenizer import Tokenizer

MODEL = "google/gemma-4-E2B-it"


async def main():
    backend = TorchBackend(device="mps")
    print("Loading model...")
    backend.load_model(MODEL)
    cache = CacheManager(num_blocks=256, block_size=16, eviction_policy="lru")
    backend.set_cache_adapter(cache)
    tok = Tokenizer(MODEL, 8192)

    sched = ContinuousBatchScheduler(backend, max_batch_size=8)
    sched.start()

    prompts = [
        ("What is 2+2?", 8),                                # short generation
        ("What is the capital of France?", 12),
        ("Write a one-sentence poem about the sea.", 30),   # longer
        ("Name three colors.", 10),
    ]

    async def submit(p, mt, sid):
        loop = asyncio.get_running_loop()
        ids = tok.encode_chat(p, thinking=False)
        req = ScheduledRequest(
            token_ids=ids, max_tokens=mt, session_id=sid,
            future=loop.create_future(),
        )
        t0 = time.perf_counter()
        out = await sched.submit(req)
        dt = time.perf_counter() - t0
        return sid, dt, out

    t_all = time.perf_counter()
    results = await asyncio.gather(*[
        submit(p, mt, f"s{i}") for i, (p, mt) in enumerate(prompts)
    ])
    t_total = time.perf_counter() - t_all

    print(f"\nTotal wall time: {t_total*1000:.0f} ms")
    print("Per-request:")
    for sid, dt, out in sorted(results, key=lambda x: x[1]):
        text = tok.decode(out)[:60]
        print(f"  {sid}: {dt*1000:6.0f} ms  ({len(out):2d} toks)  {text!r}")

    print("\ncache stats:", cache.hit_rate_info)
    await sched.stop()


if __name__ == "__main__":
    asyncio.run(main())
