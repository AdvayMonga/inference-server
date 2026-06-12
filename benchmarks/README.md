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

## Not captured yet

- vLLM head-to-head — the eventual apples-to-apples comparison (Phase 11). We're now fast
  *and* correct (sliding window), so it would finally be valid.
