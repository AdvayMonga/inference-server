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
