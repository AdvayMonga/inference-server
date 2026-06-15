# Tuning Log

Append-only ledger of optimization experiments — **every** attempt recorded, kept or dropped,
so we never re-run a known-answer (expensive A100) job. One row per experiment. Newest at top.

Baseline reference: **A100-80GB / E4B / N=32**, vs vLLM 2628 tok/s, TPOT 11 ms, TTFT 48 ms.

Columns: date · experiment · change · before→after (tok/s @N=32 · TPOT p50) · parity · **keep/drop** · notes

| date | experiment | change | tok/s | TPOT | parity | verdict | notes |
|---|---|---|---|---|---|---|---|
| 2026-06-14 | graph-break diagnostic | `CUSTOM_BACKEND_EXPLAIN=1` — `torch._dynamo.explain` over decode forward, logs break-count-by-reason | — | — | n/a | **kept** (tooling) | measures fusion headroom; not a perf change. Pending first A100 run to get the count. |
| 2026-06-14 | whole-model torch.compile | `CUSTOM_BACKEND_COMPILE=1` — `torch.compile(self.model, dynamic=False)` on decode forward | 650→**957** | 43→**29 ms** | argmax ✓ (job bmdivbgf5) | **kept** | 1.47× end-to-end. Also helped TTFT 189→133 (faster decode drains batch → prefills wait less). Inductor fuses the elementwise/norm/RoPE islands between graph breaks. |

## Planned / pending measurement

- **max-autotune-no-cudagraphs** — compile-mode flag; Inductor autotunes GEMMs + fuses harder. `-no-cudagraphs` because we capture our own CUDA graph. Ceiling capped by current graph breaks.
- **kill per-layer graph breaks** (BIGGER — discuss before building) — register the Triton attention kernel + the Python paged scatter as `torch.library` custom ops so Dynamo keeps them in-graph → whole-layer Inductor fusion. Raises the autotune ceiling.
