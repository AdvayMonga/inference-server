"""Smoke test for batched generation with per-row prefix caching."""

import time

from inference_server.backends.torch_backend import TorchBackend
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.tokenizer import Tokenizer

MODEL = "google/gemma-4-E2B-it"


def encode(tok, text):
    return tok.encode_chat(text, thinking=False)


def main():
    backend = TorchBackend(device="mps")
    print("Loading model...")
    backend.load_model(MODEL)
    cache = CacheManager(num_blocks=256, block_size=16, eviction_policy="lru")
    backend.set_cache_adapter(cache)
    tok = Tokenizer(MODEL, 8192)

    p1 = encode(tok, "What is 2+2?")
    p2 = encode(tok, "What is the capital of France?")
    p3 = encode(tok, "What is 2+2?")  # duplicate of p1 — cross-session hit

    # First batch — all cold
    print("\n--- Batch 1 (cold) ---")
    t0 = time.perf_counter()
    out1 = backend.generate_batch([p1, p2], [20, 20], session_ids=["s1", "s2"])
    dt1 = time.perf_counter() - t0
    print(f"time: {dt1*1000:.0f} ms")
    print("stats:", cache.hit_rate_info)
    for i, ids in enumerate(out1):
        print(f"  row {i}: {tok.decode(ids)[:80]!r}")

    # Second batch — p3 should hit p1's cached prefix; p2 from a new session reuses too
    print("\n--- Batch 2 (warm) ---")
    t0 = time.perf_counter()
    out2 = backend.generate_batch([p3, p2], [20, 20], session_ids=["s3", "s4"])
    dt2 = time.perf_counter() - t0
    print(f"time: {dt2*1000:.0f} ms")
    print("stats:", cache.hit_rate_info)
    for i, ids in enumerate(out2):
        print(f"  row {i}: {tok.decode(ids)[:80]!r}")

    print(f"\nspeedup: {dt1/dt2:.2f}x")


if __name__ == "__main__":
    main()
