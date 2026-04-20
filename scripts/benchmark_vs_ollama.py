"""Benchmark our server vs Ollama — same prompts, same hardware."""

import json
import time

import httpx
import numpy as np

PROMPTS = [
    "What is the capital of Japan?",
    "Explain gravity in one sentence.",
    "What is 15 * 23?",
    "Write a haiku about coding.",
    "What is Python used for?",
]

WARMUP = 2
RUNS = 5
MAX_TOKENS = 100


def benchmark_ollama(prompts, runs, max_tokens):
    """Benchmark Ollama via its API."""
    results = []
    client = httpx.Client(timeout=120.0)

    # Warmup
    for _ in range(WARMUP):
        client.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e2b", "prompt": prompts[0],
            "stream": False, "options": {"num_predict": max_tokens},
        })

    for i in range(runs):
        for prompt in prompts:
            t0 = time.perf_counter()
            resp = client.post("http://localhost:11434/api/generate", json={
                "model": "gemma4:e2b", "prompt": prompt,
                "stream": False, "options": {"num_predict": max_tokens},
            })
            total = (time.perf_counter() - t0) * 1000

            data = resp.json()
            tokens = data.get("eval_count", 0)
            ttft = data.get("prompt_eval_duration", 0) / 1e6  # ns to ms
            eval_duration = data.get("eval_duration", 0) / 1e6
            tpot = eval_duration / max(tokens - 1, 1) if tokens > 1 else 0

            results.append({
                "prompt": prompt[:30], "tokens": tokens,
                "ttft_ms": ttft, "tpot_ms": tpot, "total_ms": total,
            })

    client.close()
    return results


def benchmark_ours(prompts, runs, max_tokens):
    """Benchmark our server via its API."""
    results = []
    client = httpx.Client(timeout=120.0)

    # Warmup
    for _ in range(WARMUP):
        client.post("http://localhost:8000/generate", json={
            "text": prompts[0], "max_tokens": max_tokens,
            "stream": False, "thinking": False,
        })

    for i in range(runs):
        for prompt in prompts:
            t0 = time.perf_counter()
            resp = client.post("http://localhost:8000/generate", json={
                "text": prompt, "max_tokens": max_tokens,
                "stream": False, "thinking": False,
            })
            total = (time.perf_counter() - t0) * 1000

            data = resp.json()
            tokens = data.get("tokens_generated", 0)
            tpot = total / max(tokens, 1)

            results.append({
                "prompt": prompt[:30], "tokens": tokens,
                "ttft_ms": total,  # approximate for non-streaming
                "tpot_ms": tpot, "total_ms": total,
            })

    client.close()
    return results


def summarize(results, name):
    totals = [r["total_ms"] for r in results]
    tpots = [r["tpot_ms"] for r in results if r["tpot_ms"] > 0]
    tokens = [r["tokens"] for r in results]
    tps = [r["tokens"] / (r["total_ms"] / 1000) for r in results if r["total_ms"] > 0]

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<25} {'p50':>8} {'p95':>8} {'p99':>8}")
    print(f"  {'-' * 49}")
    print(f"  {'Total latency (ms)':<25} {np.percentile(totals, 50):>8.0f} {np.percentile(totals, 95):>8.0f} {np.percentile(totals, 99):>8.0f}")
    print(f"  {'TPOT (ms)':<25} {np.percentile(tpots, 50):>8.1f} {np.percentile(tpots, 95):>8.1f} {np.percentile(tpots, 99):>8.1f}")
    print(f"  {'Tokens/sec':<25} {np.percentile(tps, 50):>8.1f} {np.percentile(tps, 95):>8.1f} {np.percentile(tps, 99):>8.1f}")
    print(f"  {'-' * 49}")
    print(f"  {'Avg tokens generated':<25} {np.mean(tokens):>8.0f}")
    print(f"  {'Total requests':<25} {len(results):>8}")
    print(f"{'=' * 50}")


def main():
    print("Benchmarking Ollama (gemma4:e2b)...")
    ollama_results = benchmark_ollama(PROMPTS, RUNS, MAX_TOKENS)
    summarize(ollama_results, "OLLAMA (gemma4:e2b)")

    print("\nBenchmarking our server...")
    try:
        our_results = benchmark_ours(PROMPTS, RUNS, MAX_TOKENS)
        summarize(our_results, "OUR SERVER (gemma-4-E2B-it)")
    except Exception as e:
        print(f"  Our server not running: {e}")
        print("  Start with: uvicorn inference_server.server:app --port 8000")

    if ollama_results:
        ollama_tpot = np.median([r["tpot_ms"] for r in ollama_results if r["tpot_ms"] > 0])
        ollama_tps = np.median([r["tokens"] / (r["total_ms"] / 1000) for r in ollama_results if r["total_ms"] > 0])
        try:
            our_tpot = np.median([r["tpot_ms"] for r in our_results if r["tpot_ms"] > 0])
            our_tps = np.median([r["tokens"] / (r["total_ms"] / 1000) for r in our_results if r["total_ms"] > 0])
            print(f"\n--- COMPARISON ---")
            print(f"  TPOT: Ollama {ollama_tpot:.1f}ms vs Ours {our_tpot:.1f}ms ({ollama_tpot/our_tpot:.1f}x)")
            print(f"  Tok/s: Ollama {ollama_tps:.1f} vs Ours {our_tps:.1f} ({our_tps/ollama_tps:.1f}x)")
        except:
            pass


if __name__ == "__main__":
    main()
