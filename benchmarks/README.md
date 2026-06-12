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

70% short (60/100) · 20% medium (250/300) · 10% long (800/600). N=1,2,4 from a warm rerun;
N=8,16,32 from the full sweep. Scales cleanly (0 errors) after the scheduler mask-bookkeeping
fix (2026-06-12) that previously crashed the worker when a long row was evicted.

| N | throughput (tok/s) | TTFT p50 / p95 (ms) | TPOT p50 (ms) |
|---|---|---|---|
| 1  | 20  | 597 / 642   | 50.5 |
| 4  | 71  | 612 / 863   | 57.2 |
| 8  | 154 | 757 / 1173  | 61.5 |
| 16 | 249 | 1329 / 2133 | 77.8 |
| 32 | 397 | 1998 / 2642 | 108  |

TTFT grows with concurrency (queueing) but TPOT stays healthy and no head-of-line collapse
appears — consistent with deferring mixed/chunked prefill.

## Not captured yet

- **`long`** workload (500/500) sweep — interrupted; rerun when convenient.
- vLLM head-to-head — the eventual apples-to-apples comparison (Phase 11).
