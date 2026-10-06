# Inference Server

A single-GPU LLM inference engine written from scratch in Python. Gemma 4 and Qwen3-MoE on PyTorch.
The environment that measures and improves it (the agent lab, workloads, load regimes, correctness gate,
referee, GPU VMs) lives in [BlameGraph](https://github.com/AdvayMonga/BlameGraph).

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
| [`CLAUDE.md`](CLAUDE.md) | how to work with Claude on this project |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | branch → PR → `ci-ok` → merge |
| [`scripts/README.md`](scripts/README.md) | launch tuning, GPU checks, smoke |

---

## Repository layout

```
src/inference_server/     the engine; knows nothing about the environment
src/control_plane/        router and global prefix index (multi-replica; not wired yet)
scripts/                  bench/tune_triton_launch.py, gpu_tests/, hooks/, tools/smoke_custom.py
monitoring/               Prometheus + Grafana
docs/papers/              reading notes on what this borrows from
tests/                    engine and control plane tests; model-heavy ones opt in with -m heavy
```

The research loop (`src/inference_server/research/`, the merge gate, the simulator) was removed on
2026-10-02 (tag `archive/research-loop`); the lab, corpus and knowledge base moved to BlameGraph on
2026-10-06 (tag `archive/pre-split`).

Out of scope by design: model registry, LoRA, multi-tenant auth, gateway features. The
`session_id` threading and the `InferenceBackend` / `SchedulerInterface` / `CacheManager` seams
exist so a platform layer can be added later without rewriting the engine.
