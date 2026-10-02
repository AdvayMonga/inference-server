# Inference Server

A single-GPU LLM inference engine written from scratch in Python, and the lab that improves it:
an open environment where an agent gets a goal, a budget, tools and memory, and is measured by
a strict referee. Gemma 4 and Qwen3-MoE on PyTorch. The design is in [`ENVIRONMENT.md`](ENVIRONMENT.md).

---

## The engine (`src/inference_server/`)

- Continuous batching, iteration-level: freed slots refill immediately (`ContinuousBatchScheduler`)
- Prefill in three modes: monolithic, chunked (no head-of-line stall), batched (one forward)
- Block-paged KV cache with refcounted blocks, window-aware sliding layers
- Radix-tree `PrefixCache`: cross-session prefix sharing, so a shared system prompt is prefilled once
- Triton paged decode and prefill kernels reading K/V through block tables; split-K decode variant
- Bucketed CUDA-graph decode, `torch.compile`, int8 weight-only quantization
- Hand-written forwards: Gemma 4 (byte-identical to HF on the parity fixture) and Qwen3-MoE
- FCFS and per-session fair share (virtual token counter) with priority hooks
- Queue and KV backpressure (HTTP 429), admission deadline, preemption under KV pressure
- HF Transformers baseline backend behind the same `InferenceBackend` interface, with LRU,
  AttentionSink and H2O eviction policies
- FastAPI + SSE, `session_id` threaded end to end; OpenAI shim (`/v1/completions`,
  `/v1/chat/completions`, `/v1/models`). No built-in UI
- Sliding-window p50/p95/p99 on `/scheduler/stats`; Prometheus `/metrics` + Grafana (`monitoring/`)
- Per-request telemetry rows (`TELEMETRY_DIR`) and the engine event timeline (`TIMELINE_DIR`):
  one JSONL event per scheduler decision and a profiler range per phase, keyed by step id

## The lab (`lab/`)

What exists today. The rest of the design, and the order it lands in, is in `ENVIRONMENT.md`.

- `python -m lab.profile`: the engine in process under torch.profiler, writing a raw bundle per run
  (events, chrome trace, memory, GPU samples, provenance)
- `python -m lab.vm`: one persistent GPU VM on Verda (default, 1x H100 SXM) or Crusoe: start,
  setup, run a command on the pushed tree, fetch outputs, stop
- `python -m lab.ledger`: the append-only raw knowledge base, one JSON line per tool call;
  `seed` imports the hand-written findings in `knowledge/`
- `lab/corpus.py` and `scripts/corpus/`: the frozen workload corpus built from real public traces
- `python -m lab.session --goal ... --budget 20`: the environment runtime. One loop: while dollars
  remain, start an agent session with the goal, the ledger and the tools (`test`, `profile`,
  `ledger`, `budget`, `restore`, `note`; `bench`, `equiv` and `submit` refuse until the eval
  harness is wired). The referee is `lab/safety/`: write surfaces, the srt jail, the grader.

---

## Quickstart

```bash
uv sync --extra dev && source .venv/bin/activate   # pinned by uv.lock
cp .env.example .env                      # set MODEL_NAME; DEVICE auto-detects CUDA → MPS → CPU

uvicorn inference_server.server:app --host 0.0.0.0 --port 8000
curl localhost:8000/                      # service index: every route the server exposes

pytest -q                                 # fast suite, CPU
pytest -m heavy                           # model-heavy tests that load the real model
```

Anything speaking the OpenAI API works unmodified. `/generate` is the native surface and carries
`session_id`, priority and per-request sampling the shim does not expose.

```bash
docker compose -f monitoring/docker-compose.yml up -d   # Grafana on :3000, admin/admin
```

Key knobs (all in `.env.example`): `BACKEND=custom-cuda`, `PREFILL_MODE=batched|chunked`,
`MAX_BATCH_SIZE`, `MAX_ACTIVE_KV_TOKENS`, `SCHEDULING_POLICY=fcfs|fair`,
`CUSTOM_BACKEND_COMPILE=1`, `TELEMETRY_DIR=runs/telemetry`, `TIMELINE_DIR=runs/timeline`.

---

## Where to read next

| file | what it is |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | the lab: why the loop went, what the referee is, what gets built next |
| [`lab/README.md`](lab/README.md) | the lab's tools and their contract with the engine |
| [`CLAUDE.md`](CLAUDE.md) | how to work with Claude on this project |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | branch → PR → `ci-ok` → merge |
| [`corpus/README.md`](corpus/README.md) | the frozen workload traces and their versioning |
| [`knowledge/README.md`](knowledge/README.md) | the measured findings, and the evidence behind them |
| [`scripts/README.md`](scripts/README.md) | benchmarks, probes and GPU checks |

---

## Repository layout

```
src/inference_server/     the engine; knows nothing about the lab
src/control_plane/        router and global prefix index (multi-replica; not wired yet)
lab/                      the environment: profile, vm, ledger, corpus loader, providers/
corpus/                   frozen workload traces per class, seen / held-out, hashed
knowledge/                findings, one JSON each, plus evidence/ (experiment records, sweep CSVs)
scripts/                  bench/, probes/, gpu_tests/, corpus/, hooks/, tools/
monitoring/               Prometheus + Grafana
docs/papers/              reading notes on what this borrows from
tests/                    engine, control plane and lab tests; model-heavy ones opt in with -m heavy
```

The research loop that preceded the lab (`src/inference_server/research/`, the merge gate, the
simulator) was removed on 2026-10-02; tag `archive/research-loop` holds it.

Out of scope by design: model registry, LoRA, multi-tenant auth, gateway features. The
`session_id` threading and the `InferenceBackend` / `SchedulerInterface` / `CacheManager` seams
exist so a platform layer can be added later without rewriting the engine.
