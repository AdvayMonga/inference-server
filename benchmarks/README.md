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

## Head-to-head: `sweep_custom_cuda.csv` / `sweep_cuda.csv` / `sweep_vllm.csv`

Concurrency sweep, A10G/E2B, decode-bound (60-token prompt, 100 max-tokens, bounded distinct
prompts), closed-loop users, identical TTFT/TPOT measurement across all three:
- **custom** — our engine (`bench_load_sweep_modal.py`, `custom-cuda`).
- **HF** — `AutoModelForCausalLM` baseline (`cuda`) under our *same* scheduler.
- **vLLM** — vLLM 0.23 V1 in-process (`bench_vllm_sweep_modal.py`), full opts (CUDA graphs +
  prefix cache + chunked prefill, `max_num_seqs=32`). vLLM runs this model fine (head_dim 512 →
  its TRITON_ATTN backend, same approach as ours).

| N | tok/s custom / HF / vLLM | TPOT p50 custom / HF / vLLM (ms) | TTFT p50 custom / vLLM (ms) |
|---|---|---|---|
| 1  | 45.5 / 21.7 / 85.1     | 21.4 / 45.9 / 11.7 | 75 / 19 |
| 4  | 167.2 / 81.0 / 315.1   | 21.7 / 47.4 / 12.5 | 245 / 34 |
| 8  | 300.3 / 154.1 / 610.0  | 22.2 / 47.4 / 12.9 | 467 / 38 |
| 16 | 493.9 / 282.5 / 1155.2 | 23.2 / 47.4 / 13.4 | 930 / 55 |
| 32 | 752.7 / 479.6 / 2034.5 | 24.1 / 48.0 / 14.2 | 1841 / 93 |

**We sit between HF and vLLM.** vs HF: ~2× faster TPOT (custom forward + paged kernel + CUDA graph
vs sdpa + DynamicCache). vs vLLM: we trail.
- **TPOT ~1.7× behind vLLM** (21–24 vs 12–14 ms) — vLLM's kernels + Inductor op-fusion vs our
  Triton kernel + graph. Within 1.7× of SOTA on raw decode for a from-scratch engine.
- **TTFT is the dominant gap** — at N=32, 1841 ms (custom) vs 93 ms (vLLM), ~20×. Our closed loop is
  TTFT-bound at saturation; vLLM holds TTFT flat. This drives most of the throughput gap (2.7× @ N=32).
  Partly closed-loop **wave synchronization** (32 users finish in lockstep → resubmit together → wait
  a full batch drain) — needs an open/staggered-arrival probe to separate workload artifact from real
  scheduler-admission deficiency. Either way the fix is smoother admission under load.

## `sweep_custom_cuda_batched.csv` — batched prefill closes most of the TTFT gap

The head-to-head TTFT gap was diagnosed (`scripts/profile_prefill_modal.py`) as **eager, serial
prefill**: prefill runs eager at ~92 ms (dispatch-bound — even a 1-token prefill is 73ms; 4.4× a
21ms graphed decode step), and admission ran N of them serially per wave. `PREFILL_MODE=batched`
(`prefill_batch`) admits the whole wave in ONE padded forward instead.

| N | TTFT p50 mono → batched (vLLM) | tok/s mono → batched (vLLM) |
|---|---|---|
| 1  | 75 → 50 (19)     | 45.5 → 46.6 (85.1) |
| 8  | 467 → 104 (38)   | 300.3 → 344.5 (610.0) |
| 16 | 930 → 188 (55)   | 493.9 → 647.1 (1155.2) |
| 32 | 1841 → 349 (93)  | 752.7 → 1151.9 (2034.5) |

- **TTFT @ N=32: 1841 → 349 ms (5.3×).** Throughput 753 → 1152 tok/s (1.53×). TPOT unchanged (~24ms).
- **Gap to vLLM closed:** TTFT ~20× → ~3.7×, throughput 2.7× → ~1.8×.
- Remaining TTFT gap (349 vs 93): our batched prefill is still *eager* (one amortized dispatch);
  vLLM *graphs* prefill at bucketed sizes. The TPOT gap (24 vs 14) is the separate kernel/fusion lever.

## Not captured yet

- **Graph/compile the batched prefill** — close the residual TTFT gap (eager → bucketed graphs, vLLM-style).
- **TPOT vs vLLM** (24 vs 14 ms) — kernel/op-fusion lever; biggest remaining decode gap.
- **guidellm/stopwatch standardized run** — the in-process sweep above is the controlled comparison;
  a published-table number would use the OpenAI `/v1/completions` shim + guidellm (DECISIONS [2026-05-18]).
- **A100/H100 + E4B** — int8/fp8 throughput + the head-to-head re-run on serving hardware.
