# Benchmarks

Every CSV here is one sweep, kept so its numbers can be re-read later. Newer, replicated
serving numbers live in `runs/` panels (gitignored) and are summarised in `knowledge/`; the
files here are the closed-loop sweeps that built the engine, June 2026, plus the first
open-loop serving sweeps from September.

**Never compare a number here against one from another session** — run-to-run spread on the
same config has measured 2.3x on A100 draws. Only simultaneous arms are comparable.

## File index

| file | hardware / model | produced by | what it shows |
|---|---|---|---|
| `short_1_rowbyrow.csv` … `short_4_kernel_depython.csv` | A10G / E2B | `scripts/bench/load_test.py` | decode-path progression: row-by-row → batched gather → Triton kernel → de-Pythoned |
| `mixed.csv`, `long.csv` | A10G / E2B | `scripts/bench/load_test.py` | realistic mixed lengths; long-context with sliding-window attention |
| `chunked_prefill_ab.csv` | A10G / E2B | `scripts/bench/bench_chunked_prefill_modal.py` | chunked vs monolithic prefill: head-of-line stall 12x |
| `sweep_custom_cuda.csv`, `sweep_cuda.csv`, `sweep_vllm.csv` | A10G / E2B | `bench_load_sweep_modal.py`, `bench_vllm_sweep_modal.py` | first head-to-head: ours vs HF vs vLLM |
| `sweep_custom_cuda_batched.csv` | A10G / E2B | `bench_load_sweep_modal.py` | batched prefill: TTFT 1841 → 213 ms |
| `sweep_*_a100_e2b.csv`, `sweep_*_a100_e4b.csv` | A100-80GB / E2B, E4B | same | the real head-to-head on serving hardware and the headline model |
| `sweep_custom_cuda_batched_a100_e4b_compile.csv`, `sweep_custom_cuda_bucketed_depython_a100_e4b.csv` | A100-80GB / E4B | same | torch.compile decode; bucketed CUDA graphs |
| `serving_a100_e4b*.csv` | A100-80GB / E4B | `scripts/bench/bench_serving_modal.py` | open-loop Poisson serving sweeps (rate → throughput, TTFT, TPOT, SLO) |
| `roofline_e4b_a100.txt` | A100-80GB / E4B | `scripts/bench/roofline.py` | analytical decode ceiling and per-step breakdown |
| `tuning_log.md` | — | hand-kept | append-only ledger of every optimisation attempt, kept or dropped, pre-loop |

## Closed-loop sweeps (June 2026)

**Setup (all runs):** `BACKEND=custom-cuda` (hand-written Gemma 4 forward + paged KV +
prefix sharing, scheduler-driven) on Modal **A10G**, model **google/gemma-4-E2B-it**, bf16.
KV knobs: `CUSTOM_BACKEND_BLOCKS=1536` (~24.5k token-slots/layer), `MAX_ACTIVE_KV_TOKENS=22000`.
Captured 2026-06-11/12. `tok_per_s` is completed-request token throughput over the level
window; for decode-scaling read the aggregate `N / TPOT` (TPOT is per-output-token latency).
Plot any of these with `scripts/bench/plot_load_test.py <csv>...`.

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

A/B of `PREFILL_MODE` (`scripts/bench/bench_chunked_prefill_modal.py`, A10G/E2B, in-process scheduler).
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

## `sweep_custom_cuda_batched.csv` — batched prefill closes the TTFT gap (8.6×)

The head-to-head TTFT gap was diagnosed (`scripts/probes/profile_prefill_modal.py`) as **eager, serial
prefill**: prefill runs eager at ~92 ms (dispatch-bound — even a 1-token prefill is 73ms; 4.4× a
21ms graphed decode step), and admission ran N of them serially per wave. `PREFILL_MODE=batched`
admits the whole wave in ONE forward. Three iterations got it to vLLM's neighborhood:

**TTFT p50 @ N=32 (ms):**

| stage | TTFT | tok/s | vs vLLM TTFT | what |
|---|---|---|---|---|
| monolithic | 1841 | 753 | 20× | N serial eager prefills/wave |
| v1 batched (full) | 349 | 1152 | 3.7× | one padded forward, but re-prefills cached tokens |
| v2 gather (prefix-aware) | 304 | 1098 | 3.3× | suffix-only, but the paged-prefix **gather** ate the win |
| **v3 kernel (prefix-aware)** | **213** | **1218** | **2.3×** | paged prefill kernel — suffix-only **and** gather-free |
| vLLM | 93 | 2034 | 1× | (graphs prefill at bucketed sizes) |

- **Full arc: TTFT 1841 → 213 ms (8.6×)**, throughput 753 → 1218 tok/s (1.6×). Gap to vLLM 20× → 2.3×.
- v3 = `models/paged_attention_kernel.py::paged_prefill_attention` (multi-query, reads paged prefix+
  suffix via block tables, no gather — the prefill cousin of the decode kernel). Wired via a
  `_PrefillCtx` `paged_ctx`. CUDA only; CPU keeps the gather path (v2) as reference.
- Residual TTFT (213 vs 93): the suffix forward is still **eager** (~73ms dispatch); graphing it
  (fixed [N,1] shape on cache hits) would close most of it. Remaining throughput gap is now **TPOT**
  (24 vs 14 ms) = kernel-eff/quantization (lever 2, A100).

## A100/E4B head-to-head — `sweep_{custom_cuda_batched,vllm}_a100_e4b.csv` (2026-06-14)

The real head-to-head on **serving hardware (A100-80GB) + the headline model (E4B)**. Identical
closed-loop sweep, `PREFILL_MODE=batched`. Set via `BENCH_GPU=A100-80GB BENCH_MODEL=...E4B-it`.

| N | tok/s custom / vLLM | gap | TPOT p50 custom / vLLM (ms) | TTFT p50 custom / vLLM (ms) |
|---|---|---|---|---|
| 1  | 39.5 / 112.0   | 2.8× | 24.3 / 8.7  | 107 / 20 |
| 8  | 225.9 / 805.8  | 3.6× | 31.9 / 9.3  | 127 / 29 |
| 16 | 388.4 / 1405.8 | 3.6× | 36.8 / 10.3 | 140 / 46 |
| 32 | 649.7 / 2628.0 | **4.0×** | 43.0 / 11.3 | 189 / 48 |

**The gap widened (1.67× on A10G/E2B → 4.0× here) and the bottleneck MOVED.** A bench bug
(`BENCH_MODEL` didn't reach the container — first runs silently ran E2B) gave a free **E2B/A100**
point (`sweep_*_a100_e2b.csv`), so we can decompose:

| config | custom @N=32 | vLLM @N=32 | gap |
|---|---|---|---|
| A10G / E2B | 1217 | 2034 | 1.67× |
| A100 / E2B | 1450 | 4872 | 3.36× |
| A100 / E4B | 650  | 2628 | 4.0× |

A10G→A100 on the **same E2B**: vLLM 2034→4872 (**2.4×**, cashing in A100's bandwidth); ours
1217→1450 (**1.19×**). Our E2B TPOT 24→20 ms — **flat under 3× more bandwidth**. On A10G decode was
bandwidth-bound; on A100 it's **overhead/occupancy-bound** (per-layer Triton launches × 42 layers +
unfused ops + graph-replay floor — none fixed by more HBM bandwidth). vLLM's Inductor-fused kernels
scale with the GPU; ours don't yet. **Implication: quantization alone won't close 4× — the new #1
lever is decode kernel efficiency on A100 (fusion / occupancy / fewer launches).**

## Not captured yet

- **Decode kernel efficiency on A100** — the new #1 lever (profile the 43 ms/step; fuse per-layer
  ops; cut the 42-layer launch count). The A10G op-fusion null result assumed bandwidth-bound — false on A100.
- **Graph the suffix prefill** — close the last TTFT gap (eager 73ms → ~21ms graphed, vLLM-style).
- **Quantization** — int8/fp8; still cuts bytes but no longer expected to close the gap alone.
- **guidellm/stopwatch standardized run** — the in-process sweep above is the controlled comparison;
  a published-table number would use the OpenAI `/v1/completions` shim + guidellm (DECISIONS [2026-05-18]).
