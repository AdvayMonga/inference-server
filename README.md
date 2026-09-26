# Inference Server

A single-GPU LLM inference engine written from scratch in Python, plus the research loop that
improves it — on its way to a regime-adaptive, multi-replica serving system whose policies are
learned offline by the loop and selected at runtime by a controller. Gemma 4 on PyTorch.

Measured against vLLM on the same model and hardware (`benchmarks/`), but that comparison is a
**guardrail**, not the objective. The objective is **GPU-seconds per session at a fixed p95 TTFT
ceiling**, starting with the cold-start regime. An engine change merges only with a measured,
replicated experiment record behind it — see [`LOOP.md`](LOOP.md).

---

## What's built

**Engine — data plane** (`src/inference_server/`)

- Continuous batching, iteration-level: freed slots refill immediately (`ContinuousBatchScheduler`)
- Prefill in three modes: monolithic, chunked (no head-of-line stall), batched (one forward)
- Block-paged KV cache with refcounted blocks, window-aware sliding layers
- Radix-tree `PrefixCache` — cross-session prefix sharing, so a shared system prompt is prefilled once
- Triton paged decode + prefill kernels reading K/V through block tables; split-K decode variant
- Bucketed CUDA-graph decode, `torch.compile`, int8 weight-only quantization
- Custom Gemma 4 forward, byte-identical to HF on the parity fixture — dual-RoPE, GQA with
  QK/V-norm and KV sharing, sliding windows, GeGLU, per-layer embedding gating, softcapping
- FCFS and per-session fair share (virtual token counter) with priority hooks; ordering is pure
  functions in `scheduling_policy.py` that the simulator shares
- Queue + KV backpressure (HTTP 429), admission deadline, preemption under KV pressure
- HF Transformers baseline backend behind the same `InferenceBackend` interface, with LRU /
  AttentionSink / H2O eviction policies
- FastAPI + SSE, `session_id` threaded end to end; OpenAI shim (`/v1/completions`,
  `/v1/chat/completions`, `/v1/models`). No built-in UI — bring your own frontend
- Sliding-window p50/p95/p99 on `/scheduler/stats`; Prometheus `/metrics` + Grafana (`monitoring/`)
- One SQLite telemetry row per request — conditions at arrival, spans, outcome — off unless
  `TELEMETRY_DIR` is set, never written on the scheduler thread

**Research loop — improvement plane** (`src/inference_server/research/`)

```
measure ──► attribute ──► hypothesize ──► screen ──► experiment ──► judge ──► record ──► merge
 (fixed       (ranked      (prediction     (cheapest   (2 arms,      (5 gates,   (win OR      (gate
  panel)       gaps)        + falsifier)    tier)       ABBA, ≥3)     in order)   negative)    refuses)
```

- Vitals panel (`schemas.py`) — refused without its validity block: engine SHA, harness config,
  workload regime, device state, n
- `compare.py` refuses panels from different harness configs, regimes, corpus versions or sessions
- Five gates in order (`gates.py`): validity → sanity → significance → correctness → cost
- `attribute.py` — panel to ranked gaps, deterministic, no ideas
- Frozen hashed workload corpus (`corpus/`), three classes, split seen / held-out
- Open-loop trace replay (`scripts/bench/replay_trace.py`); its CSV joins telemetry on `X-Trace-Id`
- Tier-1 simulator (`simulator.py`) — trace replay through the scheduler's real iteration order
  with a fitted `TimingModel` in place of attention. Rejects policy hypotheses; never confirms
- Knowledge base (`knowledge/*.json`, 69 entries, tagged by regime and validity range).
  The `rejected` entries stop dead ends being re-tried
- Merge gate (`scripts/premerge_check.py`) — an engine change with no green experiment record
  does not merge
- Venues (`research/venues.py`) — rent a GPU, run one instrument, bring the panel home, terminate

**Control plane** — router, global prefix index, replica launcher, controller, policy table.
Not built; Phases 5–7.

---

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                      # set MODEL_NAME; DEVICE auto-detects CUDA → MPS → CPU

uvicorn inference_server.server:app --host 0.0.0.0 --port 8000
curl localhost:8000/                      # service index: every route the server exposes

pytest -q                                 # fast suite (572 tests)
pytest -m heavy                           # 35 model-heavy tests that load the real model
```

Anything speaking the OpenAI API works unmodified — Open WebUI for chat, guidellm or vLLM's
`benchmark_serving` for load. `/generate` is the native surface and carries `session_id`,
priority and per-request sampling the shim does not expose.

```bash
docker compose -f monitoring/docker-compose.yml up -d   # Grafana on :3000, admin/admin
```

Key knobs (all in `.env.example`): `BACKEND=custom-cuda`, `PREFILL_MODE=batched|chunked`,
`MAX_BATCH_SIZE`, `MAX_ACTIVE_KV_TOKENS`, `SCHEDULING_POLICY=fcfs|fair`,
`CUSTOM_BACKEND_COMPILE=1`, `TELEMETRY_DIR=runs/telemetry`.

### The loop

```bash
python -m inference_server.research.loop attribute runs/<id>.json --out gaps.json
python -m inference_server.research.loop kb --status rejected      # what is already disproved
python -m inference_server.research.loop screen hypotheses.json    # cheapest falsification first
python -m inference_server.research.loop simulate --class steady_interactive --config '{"policy":"fair"}'
python scripts/bench/replay_trace.py --class steady_interactive --split seen --base-url http://HOST:8000
python -m inference_server.research.loop judge --hyp H.json --baseline A.json --treatment B.json
python scripts/premerge_check.py <branch> --explain
RUNPOD_API_KEY=... scripts/tools/run_on_runpod.py <instrument>     # rent, run, terminate
```

---

## Where to read next

| file | what it is |
|---|---|
| [`LOOP.md`](LOOP.md) | **the method** — read before changing the engine |
| [`CLAUDE.md`](CLAUDE.md) | how to work with Claude on this project |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | branch → PR → `ci-ok` → merge; the three CI lanes; the evidence paths |
| [`benchmarks/README.md`](benchmarks/README.md) | the sweep CSVs and how the vLLM gap closed, step by step |
| [`corpus/README.md`](corpus/README.md) | the frozen workload traces and their versioning |
| [`scripts/README.md`](scripts/README.md) | instrument layout and the GPU recipes |

---

## Repository layout

```
src/inference_server/     the engine (subject of the loop; knows nothing about it)
  research/               the loop: schemas, compare, gates, session, attribute, kb, corpus,
                          simulator, venues, CLI
knowledge/                knowledge base, one JSON per finding, with regime + validity range
experiments/              experiment ledger, one JSON per A/B; what premerge_check.py reads
corpus/                   frozen workload traces per class, seen / held-out, hashed
scripts/                  instruments and tools — bench/, probes/, gpu_tests/, tools/
benchmarks/               the sweep CSVs behind every published number
monitoring/               Prometheus + Grafana; the only place aggregate numbers live
tests/                    607 tests; 572 run on CPU, 35 opt in with -m heavy
papers/                   reading notes on what this borrows from
scripts/archive/modal/    the instruments behind the 2026 A100/A10G evidence — provenance, not run
```

---

## Status

Phases 1–3 (per-request telemetry, versioned corpus with open-loop replay, trace-replay simulator)
landed 2026-09-16. Next is Phase 4, cold start, gated on the Phase 0 decisions — model size,
substrate, KV bytes per token, per-class SLOs, loop authority. The first GPU job is the simulator's
hardware check: replay a corpus class with `TELEMETRY_DIR` set on a rented pod, fit the timing
model from the rows, rank-correlate a policy sweep against the same sweep in `loop simulate`.

Out of scope by design: model registry, LoRA, multi-tenant auth, gateway features. The
`session_id` threading and the `InferenceBackend` / `SchedulerInterface` / `CacheManager` seams
exist so a platform layer can be added later without rewriting the engine.
