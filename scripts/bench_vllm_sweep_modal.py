"""vLLM arm of the head-to-head — same closed-loop sweep as bench_load_sweep_modal.py, so the
numbers slot directly next to sweep_custom_cuda.csv (identical workload + TTFT/TPOT measurement).

Drives vLLM's V1 AsyncLLM in-process (no OpenAI shim needed for this controlled comparison).
Fairness: vLLM gets its full optimizations — CUDA graphs on (enforce_eager=False), prefix
caching + chunked prefill on (defaults), max_num_seqs=32 to match our max_batch_size.

    venv/bin/modal run scripts/bench_vllm_sweep_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    .apt_install("build-essential")  # gcc/g++ for any inductor C++ codegen during graph capture
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
app = modal.App("vllm-sweep", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
NS = [1, 4, 8, 16, 32]
PROMPT_TOKENS = 60
MAX_TOKENS = 100
DURATION = 15.0
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
async def sweep():
    import asyncio, time, uuid
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=MODEL, dtype="bfloat16", max_model_len=2048,
        gpu_memory_utilization=0.85, max_num_seqs=32, enforce_eager=False,
    ))

    N_DISTINCT = 8
    PROMPTS = [[50000 + i] + list(range(100, 100 + PROMPT_TOKENS)) for i in range(N_DISTINCT)]
    counter = {"n": 0}

    async def drive():
        counter["n"] += 1
        prompt = {"prompt_token_ids": PROMPTS[counter["n"] % N_DISTINCT]}
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
        rid = uuid.uuid4().hex
        t0 = time.perf_counter()
        stamps, prev = [], 0
        async for out in engine.generate(prompt, sp, rid):
            n = len(out.outputs[0].token_ids)
            for _ in range(n - prev):
                stamps.append(time.perf_counter())
            prev = n
            if out.finished:
                break
        return t0, stamps

    async def user(deadline, collect):
        while time.perf_counter() < deadline:
            t0, stamps = await drive()
            if collect is not None and len(stamps) >= 1:
                collect.append((t0, stamps))

    results = {}
    for n in NS:
        await asyncio.gather(*[user(time.perf_counter() + WARMUP, None) for _ in range(n)])
        collected = []
        t_start = time.perf_counter()
        await asyncio.gather(*[user(t_start + DURATION, collected) for _ in range(n)])
        window = time.perf_counter() - t_start

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
        print(f"[vllm] N={n:>2} reqs={r['reqs']:>4} tok/s={r['tok_s']:>7} "
              f"TTFT={r['ttft_p50']}/{r['ttft_p95']}/{r['ttft_p99']}ms "
              f"TPOT={r['tpot_p50']}/{r['tpot_p95']}ms", flush=True)
    return results


@app.local_entrypoint()
def main():
    import csv
    from pathlib import Path
    res = sweep.remote()
    print(f"\n=== vllm ===")
    print(f"{'N':>3} {'reqs':>5} {'tok/s':>8} {'TTFT p50/p95/p99 (ms)':>24} {'TPOT p50/p95 (ms)':>18}")
    rows = []
    for n in NS:
        r = res[n]
        print(f"{n:>3} {r['reqs']:>5} {r['tok_s']:>8} "
              f"{r['ttft_p50']:>7}/{r['ttft_p95']:>6}/{r['ttft_p99']:>6}   "
              f"{r['tpot_p50']:>8}/{r['tpot_p95']:>7}")
        rows.append({"N": n, **r})
    fname = Path("benchmarks") / "sweep_vllm.csv"
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "reqs", "tok_s", "ttft_p50", "ttft_p95",
                                          "ttft_p99", "tpot_p50", "tpot_p95"])
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {fname}")
