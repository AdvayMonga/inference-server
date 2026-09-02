# Tuning Log

Append-only ledger of optimization experiments — **every** attempt recorded, kept or dropped,
so we never re-run a known-answer (expensive A100) job. One row per experiment. Newest at top.

Baseline reference: **A100-80GB / E4B / N=32**, vs vLLM 2628 tok/s, TPOT 11 ms, TTFT 48 ms.
**Trusted anchor (re-measured 2026-06-14, same session): compile-default = 1151 tok/s, TPOT 23.8/31.0 ms, TTFT 107/248 ms.** All deltas compare to this, not the stale 957.

Columns: date · experiment · change · before→after (tok/s @N=32 · TPOT p50) · parity · **keep/drop** · notes

| date | experiment | change | tok/s | TPOT | parity | verdict | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-14 | **re-anchor** compile-default | re-ran recorded "957" config (COMPILE=1, batched) same session | 957→**1151** | 29→**23.8 ms** | n/a | **anchor** | recorded 957 (last session) was ~20% low — GPU/measurement variance across sessions. Re-anchoring before experiments paid off. Future deltas vs 1151. |
| 2026-06-14 | **graph-break count** (MAJOR) | `CUSTOM_BACKEND_EXPLAIN=1` measured the decode forward | — | — | n/a | **finding** | **0 breaks, 1 graph, 2704 ops.** Overturns the ~84-break hypothesis: Dynamo traces `@triton.jit` natively + unrolls the fixed scatter. ⇒ the "wrap kernels as custom ops to kill breaks" lever is **moot — no breaks exist**. The 1.47× was already whole-graph fusion. |
| 2026-06-14 | max-autotune-no-cudagraphs | `CUSTOM_BACKEND_COMPILE_MODE=max-autotune-no-cudagraphs` (detached) | 1151→**1178.5** (+2.4%) | 23.8→23.5 ms | argmax expected-equiv (TF32 off; not separately run) | **keep flag, NOT default** | also +11% @N=16 (658→731), +4% @N=1. Marginal at saturation, non-negative everywhere. Cost: ~15–20 min autotune compile (must run `--detach`; attached heartbeat dies). Don't default until the Inductor autotune cache is persisted to the Modal volume so cold start doesn't pay it every boot. |
| 2026-06-14 | whole-model torch.compile | `CUSTOM_BACKEND_COMPILE=1` — `torch.compile(self.model, dynamic=False)` on decode forward | 650→957 (→1151 re-anchored) | 43→29 ms | argmax ✓ (job bmdivbgf5) | **kept** | 1.47× end-to-end. Also helped TTFT 189→133 (faster decode drains batch → prefills wait less). Inductor fuses the elementwise/norm/RoPE islands between graph breaks. |

## 2026-09-01 — bucketed decode graphs + de-Pythoned step (compile OFF, A100/E4B)

Both sweeps run **simultaneously, identical config** (`BENCH_BACKENDS=custom-cuda`,
`SWEEP_PREFILL_MODE=batched`, `CUSTOM_BACKEND_COMPILE=0`, `MAX_BATCH_SIZE=32`) so this is a
controlled A/B, not a comparison against an older recorded run.

| N | item1 only tok/s | +item2 tok/s | delta | TPOT p50 (1→2) | TTFT p50 (1→2) |
|---|---|---|---|---|---|
| 1 | 46.7 | 49.1 | +5% | 20.2→19.4 | 114→83 |
| 4 | 137.3 | 145.3 | +6% | 26.4→24.7 | 138→105 |
| 8 | 218.7 | 256.1 | +17% | 33.4→28.6 | 147→113 |
| 16 | 384.4 | 429.3 | +12% | 36.8→32.4 | 168→125 |
| 32 | 598.2 | **732.3** | **+22%** | 46.0→40.0 | 216→161 |

Gain grows with N — the signature of removing O(N) per-step work. TPOT growth 1→32 fell from
+128% to +106% (vLLM's is +30%, so O(N) work remains). Saved:
`sweep_custom_cuda_bucketed_depython_a100_e4b.csv`.

⚠️ **Do not compare either column to the older `sweep_custom_cuda_batched_a100_e4b.csv`
(649.7 @ N=32).** That was recorded in a different session, and this repo has already been
burned once by cross-session variance (see the 2026-06-14 re-anchor row: the same config
measured 957 then 1151). The item1-only column reading *below* it is within that band.

### Decode-graph capture cost — the bucketing tax (`probe_graph_capture_cost_modal.py`)

Bucketing makes every bucket a distinct static shape, so with `CUSTOM_BACKEND_COMPILE=1`
each pays its own Inductor compile. Measured at `MAX_BATCH_SIZE=32` (6 buckets):

| compile mode | per-bucket | total | note |
|---|---|---|---|
| `dynamic=False` (current) | ~92s each, uniform | **553s** | scales linearly with ladder; ~828s at a 9-bucket ladder |
| `dynamic=None` (auto) | 138s, 231s, then **1.8s, 1.8s, 1.9s**, 126s | 500s | one dynamic artifact covers buckets 2/4/8; bucket 32 compiles static first, 16 recompiles dynamic, **bucket 1 re-specializes** (Dynamo 0/1 specialization) |
| `dynamic=True` (forced) | — | **FAILS** | `NameError: name 's6' is not defined` — the same Inductor dynamic-shape codegen bug HANDOFF recorded for prefill (`s46`) reproduces on decode. 0 buckets captured, fell back to eager. |

**Read:** auto-dynamic wins and the win grows with the ladder (~2.2x at a 9-bucket ladder),
provided **bucket 1 is dropped** — it costs a full ~126s re-specialization and is worth only
~4% over bucket 2 at n=1 (17.73 vs 18.53 ms/step). `dynamic=True` is off the table.
The real fix is the already-deferred **persistent Inductor cache on a Modal volume** — that
trigger has now fired.

## 2026-09-02 — is torch.compile still worth it? (open-loop SLO harness, A100/E4B)

Same harness, same config, prefill graph off, ONLY `CUSTOM_BACKEND_COMPILE` differs.

| rate | ON tok/s | OFF tok/s | ON TPOT p50 | OFF TPOT p50 | ON TTFT p50 | OFF TTFT p50 |
|---|---|---|---|---|---|---|
| 1 | 121.4 | 119.4 | **15.4** | 24.5 | 121 | 103 |
| 2 | 269.1 | 229.4 | **21.2** | 36.9 | 123 | 105 |
| 3 | 366.1 | 324.2 | 33.3 | 44.2 | 129 | 110 |
| 4 | 480.0 | 416.2 | 44.7 | 52.8 | 170 | 120 |
| 6 | 606.3 | 568.0 | 62.7 | 70.2 | 174 | 134 |

**Compile is still a large decode win — 1.59x TPOT at rate 1, 1.74x at rate 2, +15-17% throughput
above rate 1.** It has NOT eroded now that bucketed graphs / de-Pythoned step / split-K have landed.

**But it buys zero SLO headroom right now: 121 vs 119 tok/s within SLO.** The metric is TTFT-bound
at rate 1 and compile only touches decode. TTFT is in fact slightly *worse* with compile on
(121 vs 103 p50 — compile does not touch prefill; likely run variance plus a longer-warmed process).

**Read:** keep compile — it is real headroom that becomes usable the moment prefill stops binding.
But it is not urgent, so **iterate benchmarks with compile OFF** (whole graph ladder captures in
5.6s vs ~21 min) and turn it on for headline numbers only, until the persistent Inductor cache lands.

**Operational:** `bench_load_sweep_modal.py` can no longer run with compile on — its Modal function
timeout is 1800s and graph capture alone took 1329s+ before being killed mid-ladder (4 buckets at
215-375s each; the automatic-dynamic reuse did NOT kick in on that run, unlike others — the compile
cost is high-variance run to run). Timeout raised.

Saved: `serving_a100_e4b.csv` (compile ON), `serving_a100_e4b_compileoff.csv` (compile OFF).

## Profiling findings (2026-06-14, compiled decode, A100/E4B, isolated N=32)

`profile_decode_kernels_modal.py` (COMPILE=1). Isolated decode = **17 ms/step, ~1870 tok/s, 43% of A100 BW** → NOT bandwidth-bound. Leaf-kernel self-CUDA breakdown (hand-derived from the raw table; auto-buckets were buggy — fixed since: skip `aten::` parents to avoid double-counting `aten::mm`/`aten::copy_` against their child kernels):

| category | ~% | kernel(s) |
|---|---|---|
| attention | ~24% | `_paged_decode_kernel` ×3 (full/sliding variants) |
| GEMM (weights) | ~23% | `ampere_bf16_*gemm` + cutlass + splitK |
| small-kernel tail | ~19% | ~400 tiny Triton kernels/step (occupancy cost) |
| **DtoD memcpy** | **~16%** | `Memcpy DtoD` (KV/activation copies) |
| **KV scatter** | **~15%** | `triton_poi_fused_index_put` (writing new K/V into blocks) |
| norm/rope/act | ~4% | `triton_red_fused_*pow*`, `gelu_mul` |

**Two robust conclusions:** (1) model-math (attn+GEMM) is only ~47%; **~31% is explicit KV-cache data movement (scatter + DtoD copy)** — bookkeeping vLLM folds into its fused paged-attention kernel. Norm-fusion is a dead lever (~4%). (2) **Isolated decode ~1870 tok/s but closed-loop sweep = 1151** → ~40% lost to prefill/scheduling/closed-loop wave-sync, *zero* of it decode-kernel. Two distinct frontiers; see HANDOFF.

## Planned / pending measurement

- **max-autotune-no-cudagraphs** — compile-mode flag; Inductor autotunes GEMMs + fuses harder. `-no-cudagraphs` because we capture our own CUDA graph. Ceiling capped by current graph breaks.
- **kill per-layer graph breaks** (BIGGER — discuss before building) — register the Triton attention kernel + the Python paged scatter as `torch.library` custom ops so Dynamo keeps them in-graph → whole-layer Inductor fusion. Raises the autotune ceiling.
