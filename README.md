# Inference Server

**A multi-user LLM inference engine written from scratch in Python, and the research loop that
improves it.** Continuous batching, a block-paged KV cache with cross-session prefix sharing,
hand-written Gemma 4 forward with Triton paged-attention kernels and CUDA graphs, fair scheduling
with backpressure. Benchmarked head-to-head against vLLM on the same model and hardware. Since
September 2026 an engine change merges only with a measured, replicated experiment record behind
it.

[![Architecture overview](docs/architecture.svg)](https://advaymonga.github.io/inference-server/architecture.html)

**[→ Live, clickable architecture map](https://advaymonga.github.io/inference-server/architecture.html)** — scheduler, backend, KV cache, HTTP layer, and the research loop.

---

## Results

A100-80GB · Gemma 4 E4B · identical closed-loop sweep at 32 concurrent users
(`benchmarks/sweep_*_a100_e4b*.csv`, June 2026):

| engine | throughput | TPOT p50 | TTFT p50 |
|---|---|---|---|
| HF Transformers `generate()` under the same scheduler | 125 tok/s | 118 ms | 3459 ms |
| **this engine** (custom forward + paged kernels + CUDA graphs + `torch.compile`) | **1151 tok/s** | **24 ms** | **107 ms** |
| vLLM 0.23 (CUDA graphs + prefix cache + chunked prefill) | 2628 tok/s | 11 ms | 48 ms |

9× over the naive baseline, 2.3× behind vLLM. How the gap closed, step by step, is in
[`benchmarks/README.md`](benchmarks/README.md): batched + paged prefill took TTFT from
1841 ms to 213 ms on A10G; the Triton decode kernel and de-Pythoned decode step took
decode throughput ~20× over row-by-row; chunked prefill cut the head-of-line stall a long
prompt inflicts on running decoders from 2412 ms to 201 ms.

What the loop found afterwards, with replicated open-loop measurement, is more useful than any
single number: the TTFT tail is prefill *compute* (203 ms of 213 ms at rate 1), not queueing,
which ruled out every scheduling lever at once; A100 draws are bimodal (2.3× TPOT spread on
identical config), so only simultaneous A/B arms are evidence; and a 10.8× isolated kernel win
was worth nothing end to end. Those are in [`DECISIONS.md`](DECISIONS.md).

---

## Two halves

### 1. The engine — `src/inference_server/`

| component | what it does |
|---|---|
| **`scheduler.py`** — `ContinuousBatchScheduler` | Iteration-level scheduling: freed slots refill immediately. Monolithic, chunked, or batched prefill. Admission deadline, queue and KV backpressure (HTTP 429), preemption under KV pressure. |
| **`scheduling_policy.py`** | FCFS and per-session fair share (virtual token counter), priority hooks. Pluggable behind `SchedulerInterface`. |
| **`models/gemma4.py`** | Custom Gemma 4 forward, byte-identical to HF on the parity fixture: dual-RoPE, GQA with QK/V-norm and KV sharing, sliding-window layers, GeGLU, per-layer embedding gating, softcapping. |
| **`models/paged_kv_cache.py`** | Per-layer block pools with refcounted blocks (vLLM/PagedAttention layout), window-aware storage on sliding layers, and a radix-tree `PrefixCache` so requests sharing a system prompt skip its prefill across sessions. |
| **`models/paged_attention_kernel.py`** | Triton decode and prefill kernels that read K/V through block tables — no gather, no padding. Split-K decode variant. |
| **`backends/custom_torch_backend.py`** | Drives the above for the scheduler: batched prefill in one forward, bucketed CUDA-graph decode, optional `torch.compile` and int8 weight-only quantization. `BACKEND=custom-{cuda,mps,cpu}`. |
| **`backends/torch_backend.py`** | HF Transformers baseline backend behind the same `InferenceBackend` interface, with the `kv_cache/` block manager, radix tree and LRU / AttentionSink / H2O eviction policies. |
| **`server.py`** + `openai_shim.py` | FastAPI, SSE streaming, `session_id` threaded end to end, `/v1/completions` and `/v1/chat/completions` so standard benchmark clients and chat frontends drive the engine unmodified. No built-in UI. |
| **`scripts/bench/load_test.py`** | Out-of-process concurrency sweep; `--workload realistic` draws distinct prompts from `prompt_bank.py` so the run exercises the cache-miss path. |
| **`metrics.py`**, `prometheus_metrics.py` | Sliding-window p50/p95/p99 TTFT / TPOT / throughput on `/scheduler/stats`; aggregate Prometheus `/metrics` (Grafana dashboard in `monitoring/`). |

Everything is env-configured (`.env.example` lists every knob) and deploys as one container
per GPU (`modal_app.py`).

### 2. The research loop — `LOOP.md` + `src/inference_server/research/`

The loop exists because the expensive mistakes here were never bad code. They were valid-looking
comparisons between things that were not comparable: a closed-loop harness that measured the
wrong bottleneck for a month, a cache-hit benchmark that hid a pool leak, a "regression" that was
two runs with different batch sizes. So the loop fixes everything except the hypothesis.

```
measure ──► attribute ──► hypothesize ──► screen ──► experiment ──► judge ──► record ──► merge ──► re-measure
 (fixed       (ranked      (prediction     (cheapest    (2 arms,       (5 gates,    (win OR       (gate      (bottleneck
  panel)       gaps)        + falsifier)    tier first)  ABBA, ≥3 runs)  in order)   negative)     refuses)   moves)
```

| piece | where | what it enforces |
|---|---|---|
| Vitals panel | `research/schemas.py` | one fixed record per run, refused without its validity block (engine SHA, harness config, workload regime, n) |
| Comparison | `research/compare.py` | refuses to compare panels from different harness configs, regimes or sessions |
| Gates | `research/gates.py` | validity → sanity → significance → correctness → cost, judged in that order |
| Procedure | `research/session.py` | steps 4–6 as one call; raises before judging if arms are unbalanced, block-ordered, dirty, or stale |
| Attribution | `research/attribute.py` | panel → ranked gaps, deterministic, no ideas |
| Knowledge base | `knowledge/*.json` → `DECISIONS.md` | 68 entries; the `rejected` ones stop dead ends being re-tried |
| Experiment ledger | `experiments/*.json` | what was measured at which SHA, and what the gates said |
| Merge gate | `scripts/premerge_check.py` | an engine change with no green experiment record does not merge |

```bash
python -m inference_server.research.loop attribute runs/<id>.json --out gaps.json
python -m inference_server.research.loop kb --status rejected      # what is already disproved
python -m inference_server.research.loop screen hypotheses.json    # cheapest falsification first
python -m inference_server.research.loop judge --hyp H.json --baseline A.json --treatment B.json
python scripts/premerge_check.py <branch> --explain
```

A disproved hypothesis is a successful iteration. Of the nine experiments run through the full
procedure so far, three confirmed, three were noise, and three were invalid — the harness never
exercised the change. The last category is the reason the loop exists.

---

## Repository layout

```
src/inference_server/     the engine (subject of the loop; knows nothing about it)
  research/               the loop: schemas, compare, gates, session, attribute, kb, CLI
LOOP.md                   the method — read this before changing the engine
DECISIONS.md              generated view of knowledge/ (68 decisions, tagged, with evidence)
knowledge/                knowledge base, one JSON per finding
experiments/              experiment ledger, one JSON per A/B; what premerge_check.py reads
scripts/                  instruments and tools — bench/, probes/, gpu_tests/, tools/
benchmarks/               the sweep CSVs behind every number above, with a README
tests/                    304 tests; 270 run in ~8 s on CPU, 34 model-heavy ones opt in with -m heavy
docs/                     the GitHub Pages architecture map
monitoring/               Prometheus + Grafana stack; the only aggregate view
papers/                   reading notes on what this borrows from
modal_app.py              one-container-per-GPU deployment
```

---

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                      # set MODEL_NAME; DEVICE auto-detects CUDA → MPS → CPU

uvicorn inference_server.server:app --host 0.0.0.0 --port 8000
curl localhost:8000/                      # service index: every route the server exposes

pytest -q                                 # fast suite
pytest -m heavy                           # parity + scheduler tests that load the real model
```

### Talking to it

There is no built-in web UI, for the same reason vLLM ships none: a first-party page is a
second implementation of everything the API already does, and its metrics panel was a second
implementation of everything Prometheus already computes. Bring your own frontend.

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inference-server",
  "messages": [{"role": "user", "content": "Explain paged attention in two sentences."}],
  "stream": true
}'
```

Anything that speaks the OpenAI API works unmodified — Open WebUI for chat, guidellm or
vLLM's `benchmark_serving` for load. `/generate` remains the native surface and carries
`session_id`, priority and per-request sampling that the shim does not expose.

### Watching it

```bash
docker compose -f monitoring/docker-compose.yml up -d
open http://localhost:3000                # Grafana, admin/admin, dashboard provisioned
```

TTFT and TPOT percentiles, throughput, request rate by outcome, batch and queue depth, and KV
pressure. This is the only place aggregate numbers live; nothing else recomputes them.

Key knobs (all in `.env.example`): `BACKEND=custom-cuda` for the hand-written path,
`PREFILL_MODE=batched|chunked`, `MAX_BATCH_SIZE`, `MAX_ACTIVE_KV_TOKENS`,
`SCHEDULING_POLICY=fcfs|fair`, `CUSTOM_BACKEND_COMPILE=1`.

GPU work runs on rented hardware through Modal: `pip install -e ".[modal]"`, then
`scripts/run_instrument.sh scripts/bench/bench_serving_modal.py`.

---

## How a change lands

1. Attribute the current panel; pick a gap. Check `loop kb --status rejected` first.
2. Write the hypothesis with its predicted metric, magnitude and cheapest falsifier
   (`research/HYPOTHESIZE.md`).
3. Screen. Run the cheapest tier that could kill it — arithmetic, then CPU, then one GPU probe.
4. Experiment on a branch: two arms, same session, interleaved, ≥3 runs each.
5. `judge_group(...)` runs the five gates and writes the experiment record.
6. Open a PR. CI runs `scripts/premerge_check.py`, which refuses it unless the record is green
   for that SHA.
7. Merge, then re-measure — the bottleneck moves.

Correctness fixes take a different path: the evidence is a regression test that fails at the base
SHA and passes at the fix, and the gate re-runs it.

The mechanics — branching, the three CI lanes behind the single `ci-ok` check, the pre-push hook,
and when to dispatch the GPU lane — are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Status

Gemma 4 E2B / E4B, PyTorch, single GPU. The engine is at 2.3× behind vLLM on the headline sweep;
the loop's current attribution says the remaining gap is per-decode-step cost at low batch
(kernel and fusion work, not scheduling) and long-prompt prefill compute on the TTFT tail.
Out of scope by design: model registry, LoRA, multi-tenant auth, gateway features — the
`session_id` threading and `InferenceBackend` / `SchedulerInterface` seams are there so a
platform layer can be added without rewriting the engine.
