"""Baseline benchmark — measures raw generation performance before any optimization."""

import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-4-E2B-it"
DEVICE = "mps"
DTYPE = torch.bfloat16
PROMPT = "Explain how a computer works in simple terms"
MAX_TOKENS = 100
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10


def generate_timed(model, input_ids, max_tokens, tokenizer):
    """Run one generation and return detailed timing."""
    generated = []
    kv_cache = None
    input_tensor = input_ids.clone()
    token_times = []

    with torch.no_grad():
        # Prefill — process the full prompt
        prefill_start = time.perf_counter()
        outputs = model(input_tensor, past_key_values=kv_cache, use_cache=True)
        kv_cache = outputs.past_key_values
        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
        prefill_end = time.perf_counter()

        ttft = prefill_end - prefill_start
        generated.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            return ttft, [], generated

        # Decode — generate one token at a time
        input_tensor = torch.tensor([[next_token_id]], device=input_ids.device)
        for _ in range(max_tokens - 1):
            step_start = time.perf_counter()
            outputs = model(input_tensor, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
            step_end = time.perf_counter()

            token_times.append(step_end - step_start)
            generated.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            input_tensor = torch.tensor([[next_token_id]], device=input_ids.device)

    return ttft, token_times, generated


def percentiles(values):
    """Return p50, p95, p99 for a list of values."""
    return np.percentile(values, 50), np.percentile(values, 95), np.percentile(values, 99)


def main():
    print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    prompt_tokens = input_ids.shape[1]
    print(f"Prompt: '{PROMPT}' ({prompt_tokens} tokens)")
    print(f"Max tokens: {MAX_TOKENS}")
    print()

    # Warmup
    print(f"Running {WARMUP_RUNS} warmup passes...")
    for _ in range(WARMUP_RUNS):
        generate_timed(model, input_ids, MAX_TOKENS, tokenizer)
    print("Warmup done.\n")

    # Benchmark
    print(f"Running {BENCHMARK_RUNS} benchmark passes...")
    all_ttft = []
    all_tpot = []
    all_total = []
    all_tokens_generated = []

    for i in range(BENCHMARK_RUNS):
        run_start = time.perf_counter()
        ttft, token_times, generated = generate_timed(model, input_ids, MAX_TOKENS, tokenizer)
        run_end = time.perf_counter()

        total_time = run_end - run_start
        tokens_generated = len(generated)
        avg_tpot = np.mean(token_times) if token_times else 0

        all_ttft.append(ttft)
        all_tpot.extend(token_times)
        all_total.append(total_time)
        all_tokens_generated.append(tokens_generated)

        print(f"  Run {i+1}: {tokens_generated} tokens in {total_time:.2f}s "
              f"(TTFT: {ttft*1000:.0f}ms, TPOT: {avg_tpot*1000:.1f}ms)")

    # Results
    ttft_p50, ttft_p95, ttft_p99 = percentiles(all_ttft)
    tpot_p50, tpot_p95, tpot_p99 = percentiles(all_tpot)
    total_p50, total_p95, total_p99 = percentiles(all_total)
    avg_tokens = np.mean(all_tokens_generated)
    avg_throughput = np.mean([t / l for t, l in zip(all_tokens_generated, all_total)])

    print("\n" + "=" * 60)
    print("BASELINE BENCHMARK RESULTS")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE} | Dtype: {DTYPE}")
    print(f"Prompt tokens: {prompt_tokens} | Max output tokens: {MAX_TOKENS}")
    print(f"Runs: {BENCHMARK_RUNS} (after {WARMUP_RUNS} warmup)")
    print("=" * 60)
    print(f"{'Metric':<30} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("-" * 60)
    print(f"{'TTFT (ms)':<30} {ttft_p50*1000:>10.1f} {ttft_p95*1000:>10.1f} {ttft_p99*1000:>10.1f}")
    print(f"{'TPOT (ms)':<30} {tpot_p50*1000:>10.1f} {tpot_p95*1000:>10.1f} {tpot_p99*1000:>10.1f}")
    print(f"{'Total latency (s)':<30} {total_p50:>10.2f} {total_p95:>10.2f} {total_p99:>10.2f}")
    print("-" * 60)
    print(f"{'Avg tokens generated':<30} {avg_tokens:>10.0f}")
    print(f"{'Avg throughput (tokens/s)':<30} {avg_throughput:>10.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
