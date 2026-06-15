# Tuning Log

Append-only ledger of optimization experiments — **every** attempt recorded, kept or dropped,
so we never re-run a known-answer (expensive A100) job. One row per experiment. Newest at top.

Baseline reference: **A100-80GB / E4B / N=32**, vs vLLM 2628 tok/s, TPOT 11 ms, TTFT 48 ms.
**Trusted anchor (re-measured 2026-06-14, same session): compile-default = 1151 tok/s, TPOT 23.8/31.0 ms, TTFT 107/248 ms.** All deltas compare to this, not the stale 957.

Columns: date · experiment · change · before→after (tok/s @N=32 · TPOT p50) · parity · **keep/drop** · notes

| date | experiment | change | tok/s | TPOT | parity | verdict | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-14 | **re-anchor** compile-default | re-ran recorded "957" config (COMPILE=1, batched) same session | 957→**1151** | 29→**23.8 ms** | n/a | **anchor** | recorded 957 (last session) was ~20% low — GPU/measurement variance across sessions. Re-anchoring before experiments paid off. Future deltas vs 1151. |
| 2026-06-14 | graph-break diagnostic | `CUSTOM_BACKEND_EXPLAIN=1` — `torch._dynamo.explain`, break-count-by-reason | — | — | n/a | **kept** (tooling) | ran on A100 but `logger.info` was swallowed (bench containers don't configure logging → INFO dropped). Switched to `print(flush=True)`. Count comes on run 2. |
| 2026-06-14 | whole-model torch.compile | `CUSTOM_BACKEND_COMPILE=1` — `torch.compile(self.model, dynamic=False)` on decode forward | 650→957 (→1151 re-anchored) | 43→29 ms | argmax ✓ (job bmdivbgf5) | **kept** | 1.47× end-to-end. Also helped TTFT 189→133 (faster decode drains batch → prefills wait less). Inductor fuses the elementwise/norm/RoPE islands between graph breaks. |

## Planned / pending measurement

- **max-autotune-no-cudagraphs** — compile-mode flag; Inductor autotunes GEMMs + fuses harder. `-no-cudagraphs` because we capture our own CUDA graph. Ceiling capped by current graph breaks.
- **kill per-layer graph breaks** (BIGGER — discuss before building) — register the Triton attention kernel + the Python paged scatter as `torch.library` custom ops so Dynamo keeps them in-graph → whole-layer Inductor fusion. Raises the autotune ceiling.
