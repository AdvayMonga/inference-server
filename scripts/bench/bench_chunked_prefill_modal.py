"""Chunked-prefill A/B (A10G): does PREFILL_MODE=chunked protect short requests' latency
when a long prompt is admitted?

Workload: launch N short requests (they start decoding), then inject one long (2048-token)
prompt. The headline metric is the short cohort's MAX inter-token gap — the stall a decoder
sees while the long prompt prefills.
  - monolithic: the long prompt's prefill is ONE 2048-token forward that freezes the whole
    decode loop (HOL block) → a big gap in every short request's token stream.
  - chunked (V-A): prefill is sliced into PREFILL_CHUNK_SIZE chunks interleaved with decode
    steps → short requests keep flowing, small gap.
Tradeoff to watch: chunked makes the LONG request's own TTFT slightly worse (its prefill is
spread across iterations). We report both.

    venv/bin/modal run scripts/bench/bench_chunked_prefill_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("chunked-prefill-ab", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
N_SHORT = 6
SHORT_PROMPT_LEN = 16
SHORT_MAX_TOKENS = 48
LONG_PROMPT_LEN = 2048
LONG_MAX_TOKENS = 16
CHUNK = 256


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
async def run():
    import asyncio, time, statistics
    from inference_server.backends import create_backend
    from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    loop = asyncio.get_running_loop()

    short_prompt = list(range(100, 100 + SHORT_PROMPT_LEN))
    long_prompt = list(range(100, 100 + LONG_PROMPT_LEN))

    async def drive(sched, prompt, max_tokens, sid):
        """Submit one request; timestamp every streamed token. Returns (submit_t, [stamps])."""
        q: asyncio.Queue = asyncio.Queue()
        req = ScheduledRequest(token_ids=list(prompt), max_tokens=max_tokens, session_id=sid,
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

    async def ab(mode):
        sched = ContinuousBatchScheduler(
            backend, max_batch_size=16,
            prefill_chunk_size=CHUNK if mode == "chunked" else 0,
            prefill_mode=mode,
        )
        sched.start()
        try:
            shorts = [asyncio.create_task(drive(sched, short_prompt, SHORT_MAX_TOKENS, f"s{i}"))
                      for i in range(N_SHORT)]
            await asyncio.sleep(0.15)                      # let shorts get admitted + decoding
            long = asyncio.create_task(drive(sched, long_prompt, LONG_MAX_TOKENS, "long"))
            short_res = await asyncio.gather(*shorts)
            long_t0, long_stamps = await long
        finally:
            await sched.stop()

        ttfts, max_gaps, tpots = [], [], []
        for t0, stamps in short_res:
            if len(stamps) < 2:
                continue
            ttfts.append((stamps[0] - t0) * 1000)
            gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
            max_gaps.append(max(gaps))
            tpots.append(statistics.mean(gaps))
        long_ttft = (long_stamps[0] - long_t0) * 1000 if long_stamps else float("nan")
        return ttfts, max_gaps, tpots, long_ttft

    def p95(xs):
        return sorted(xs)[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else 0.0

    print(f"\nworkload: {N_SHORT} short (prompt {SHORT_PROMPT_LEN}, {SHORT_MAX_TOKENS} tok) "
          f"+ 1 long (prompt {LONG_PROMPT_LEN}) injected mid-decode; chunk={CHUNK}\n")
    print(f"{'mode':>11} | {'short TTFT p95':>14} | {'short TPOT mean':>15} | "
          f"{'short MAX gap (stall)':>21} | {'long TTFT':>10}")
    results = {}
    for mode in ("monolithic", "chunked"):
        ttfts, max_gaps, tpots, long_ttft = await ab(mode)
        results[mode] = (max(max_gaps) if max_gaps else 0.0, long_ttft)
        print(f"{mode:>11} | {p95(ttfts):>12.0f}ms | {statistics.mean(tpots):>13.0f}ms | "
              f"{max(max_gaps):>19.0f}ms | {long_ttft:>8.0f}ms")

    mono_stall, _ = results["monolithic"]
    chunk_stall, _ = results["chunked"]
    if chunk_stall > 0:
        print(f"\nHOL stall reduced {mono_stall / chunk_stall:.1f}× "
              f"({mono_stall:.0f}ms → {chunk_stall:.0f}ms)")


@app.local_entrypoint()
def main():
    run.remote()
