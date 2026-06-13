# Benchmarks

Client-measured load sweeps via `scripts/load_test.py` (per-request TTFT/TPOT/throughput
over streaming SSE, sweeping concurrency). Plot with `scripts/plot_load_test.py <csv>...`.

**Setup (all runs):** `BACKEND=custom-cuda` (hand-written Gemma 4 forward + paged KV +
prefix sharing, scheduler-driven) on Modal **A10G**, model **google/gemma-4-E2B-it**, bf16.
KV knobs: `CUSTOM_BACKEND_BLOCKS=1536` (~24.5k token-slots/layer), `MAX_ACTIVE_KV_TOKENS=22000`.
Captured 2026-06-11/12. `tok_per_s` is completed-request token throughput over the level
window; for decode-scaling read the aggregate `N / TPOT` (TPOT is per-output-token latency).

## `short_*` — decode optimization progression

Decode-bound workload (60-token prompt, 100 max-tokens). Each file is one stage of the
M2.3 work; same sweep, so they diff directly. Aggregate decode throughput (`N/TPOT`, tok/s):

| N | 1_rowbyrow | 2_batched_gather | 3_kernel | 4_kernel_depython |
|---|---|---|---|---|
| 1  | 15.6 | 18.6 | 20.2 | 21.4 |
| 8  | 15.8 | 88.6 | 112  | 141 |
| 16 | 15.4 | 114  | 160  | 236 |
| 32 | 15.7 | 83 (knee) | 185 | **323** |

- **1_rowbyrow** — one forward per row. Decode fully serialized: TPOT ≈ 64 ms × N, throughput pinned flat ~15.7 tok/s regardless of batch.
- **2_batched_gather** — one masked SDPA over a left-pad-gathered `[N,H,Lmax,D]` tensor. Scales to N=16 (~7×) but regresses at N=32 (Python per-row pad/cat loop).
- **3_kernel** — Triton paged-attention decode kernel reading K/V via block tables (no gather/pad). Knee gone, monotonic to 185 tok/s.
- **4_kernel_depython** — vectorized scatter-append + one-shot block-table build (removed per-step Python). N=32 TPOT 173→99 ms. **~20× over rowbyrow at N=32.**

## `mixed.csv` — realistic mixed load

70% short (60/100) · 20% medium (250/300) · 10% long (800/600). Single clean sweep on the
sliding-window build (2026-06-12) — 0 errors. **First correct mixed numbers:** earlier runs
predated sliding-window attention, so the 800-token-bin outputs were subtly wrong (sliding
layers attended full history). Also requires the scheduler mask-bookkeeping fix (a long row
being evicted used to crash the worker).

| N | throughput (tok/s) | TTFT p50 / p95 (ms) | TPOT p50 (ms) |
|---|---|---|---|
| 1  | 20  | 587 / 640   | 49.4 |
| 4  | 71  | 651 / 787   | 55.5 |
| 8  | 151 | 867 / 1184  | 62.1 |
| 16 | 311 | 1275 / 1528 | 74.6 |
| 32 | 397 | 2074 / 2665 | 106  |

TTFT grows with concurrency (queueing) but TPOT stays healthy and no head-of-line collapse
appears — consistent with deferring mixed/chunked prefill.

## `long.csv` — long-context load

Uniform 500-token prompt / 500 max-tokens (≈1000-token sequences, > the 512 sliding window).
N capped at 16: at 1000 tokens/request, N=32 would exceed `MAX_ACTIVE_KV_TOKENS=22000` and
admission would gate. 0 errors; correct sliding-window attention.

| N | throughput (tok/s) | TTFT p50 / p95 (ms) | TPOT p50 (ms) |
|---|---|---|---|
| 1  | 22  | 640 / 680   | 50.2 |
| 4  | 89  | 774 / 791   | 54.4 |
| 8  | 178 | 1063 / 1087 | 61.5 |
| 16 | 333 | 1577 / 1593 | 80.8 |

TPOT stays low even at 1000-token context — sliding-window attention caps 28/35 layers'
attention span at 512 keys, so decode cost doesn't blow up with context length.

## `chunked_prefill_ab.csv` — chunked prefill kills head-of-line blocking

A/B of `PREFILL_MODE` (`scripts/bench_chunked_prefill_modal.py`, A10G/E2B, in-process scheduler).
Workload: 6 short requests (16-token prompt, 48 max-tokens) decoding, then **one long 2048-token
prompt injected mid-decode**. The headline is the short cohort's **MAX inter-token gap** — the
stall a decoder sees while the long prompt prefills.

| mode | short TTFT p95 | short TPOT mean | short MAX gap (stall) | long TTFT |
|---|---|---|---|---|
| monolithic | 2607 ms | 74 ms | **2412 ms** | 2456 ms |
| chunked (256) | 392 ms | 56 ms | **201 ms** | 1801 ms |

- **HOL stall 2412 → 201 ms (12×).** Monolithic runs the long prompt's prefill as one 2048-token
  forward that freezes every active decoder for ~2.4 s. Chunked slices it into 256-token chunks
  interleaved with decode, so the worst stall is ~one chunk.
- Short **TTFT p95 6.7×** better; TPOT mean lower (the stall inflated the monolithic average).
- **No long-TTFT tradeoff at this size** — chunked's long TTFT (1801 ms) actually beat monolithic
  (2456 ms): the single 2048-token forward is itself costly enough that interleaving wins on
  wall-clock too. (A regression could appear at smaller prompts / larger chunks.)
- One workload/config — magnitude scales with prompt length and chunk size; this is the mechanism,
  not a universal "12×".

## `sweep_custom_cuda.csv` / `sweep_cuda.csv` — engine vs HF baseline

Concurrency sweep (`scripts/bench_load_sweep_modal.py`, A10G/E2B, decode-bound: 60-token prompt,
100 max-tokens, bounded distinct prompts). Closed-loop users at each N; **our engine
(`custom-cuda`) vs the HF `AutoModelForCausalLM` baseline (`cuda`)** under the *same* scheduler —
isolates what the custom forward + Triton paged kernel + CUDA graph bought us.

| N | tok/s custom / HF | speedup | TPOT p50 custom / HF (ms) | TTFT p50 custom / HF (ms) |
|---|---|---|---|---|
| 1  | 45.5 / 21.7  | 2.1× | 21.4 / 45.9 | 75 / 54 |
| 4  | 167.2 / 81.0 | 2.1× | 21.7 / 47.4 | 245 / 244 |
| 8  | 300.3 / 154.1 | 1.9× | 22.2 / 47.4 | 467 / 480 |
| 16 | 493.9 / 282.5 | 1.7× | 23.2 / 47.4 | 930 / 955 |
| 32 | 752.7 / 479.6 | 1.6× | 24.1 / 48.0 | 1841 / 1895 |

- **~2× lower TPOT (21–24 vs 46–48 ms) at every concurrency** — the decode win (custom forward +
  paged kernel + CUDA graph vs HF sdpa + DynamicCache). Both hold TPOT flat under load (both are
  real continuous batchers), so it's a clean per-token advantage.
- **Throughput speedup 2.1× → 1.6× as N grows:** at N=32 the closed loop is **TTFT-bound, not
  decode-bound** — each request's cycle is ~1.8 s TTFT (32 users contending for 32 slots) + ~2.4 s
  decode, capping throughput below the 24 ms-TPOT ceiling. TPOT shows decode has headroom; TTFT
  (queueing) is the limiter — where more KV headroom / chunked prefill for long prompts would lift it.
- **TTFT ~identical** — dominated by queueing + (cached, short) prefill, same for both backends.

## Not captured yet

- **vLLM head-to-head** — the gold-standard column. A `custom` vs `vLLM` sweep is the remaining
  comparison; `bench_load_sweep_modal.py` produces our column, vLLM needs its own driver
  (OpenAI `/v1/completions` shim + guidellm, DECISIONS [2026-05-18]) and a vLLM-supports-the-model
  check first (the compat run kept getting killed by the Modal client disconnect, not a real fail).
