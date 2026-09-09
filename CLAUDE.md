# Inference Server — Project Guide





---

## Coding Guidelines

**1. Think before coding.** State assumptions. Surface tradeoffs. If something is unclear, ask before implementing. If multiple interpretations exist, present them.

**2. Simplicity first.** Minimum code that solves the problem. No speculative abstractions, no unrequested config knobs, no error handling for impossible cases. If 200 lines could be 50, rewrite.

**3. Surgical changes.** Touch only what you must. No drive-by refactors, no reformatting untouched code. Remove orphans your changes created; leave pre-existing dead code alone (mention it, don't delete it). Every changed line should trace to the stated goal.

**4. Goal-driven execution.** Turn tasks into verifiable goals ("write the test, then make it pass"). State a brief plan for multi-step work and check off as you go.

**5. Explain before coding.** Before writing code, say what you're building, why you're building it, and how it works conceptually. Then code.

**6. Concise code docs.** Comments and docstrings are short one-liners. Detailed explanations go in chat, not source. After each change, update CLAUDE.md to make sure it reflects the codebase.

**7. Pedagogical mode for canonical-choice points.** Whenever we hit a place where the field/paper/canonical implementation made a non-obvious design choice explain in chat: (a) what the canonical choice is, (b) why they made it — the failure mode it solves or the win it produces, (c) what alternatives exist and the tradeoff, (d) where in our code we could customize it later in a novel way. Use simple language, concrete examples, analogies from physical intuition, and ASCII diagrams of shapes/flow when they help understanding. Goal: build the user's intuition so they could one day make these choices themselves rather than copy them. Only on the larger important ones.

While coding, put any, mid-flight tradeoffs and deferrals in [`DECISIONS.md`](./DECISIONS.md). CLAUDE.md is the canonical plan — it changes when scope changes. Pull from DECISIONS.md selectively (grep by topic), don't load it whole. 

When running into a large bug or pivoting in a new direction, refer to [`DECISIONS.md`] and see if any of those decisions play a role.

[`PLAN.md`](./PLAN.md) is the live working plan — what we're building *right now*; read it at session start and check items off as they land. Gitignored (local working state).

After every new feature update ['HANDOFF.md'] which reflects the most recent changes and provides context for any fresh session on the current state.


---

## How we work now — the research loop

**`LOOP.md` is the method. Follow it; do not improvise around it.** Ad-hoc optimisation is what
produced this project's two most expensive mistakes (a month on the wrong bottleneck from a
closed-loop harness, and a cache-hit benchmark that hid every miss-path bug).

```bash
python -m inference_server.research.loop attribute runs/<id>.json --out gaps.json
python -m inference_server.research.loop kb --status rejected     # what is already disproved
python -m inference_server.research.loop screen hypotheses.json   # cheapest falsification first, routed to a venue
python -m inference_server.research.loop budget                    # cap, ledger, venues (cloud opt-in: RESEARCH_VENUES)
python -m inference_server.research.loop judge --hyp H.json --baseline A.json --treatment B.json
```

- **Knowledge base:** `knowledge/*.json` is the source of truth. `DECISIONS.md` is a GENERATED
  view (tracked in git since 2026-09-08) — edit the JSON, then `loop index`, commit both.
- **`scripts/` layout (2026-09-08):** `bench/` (tier-4 sweeps, roofline, load_test), `probes/`
  (tier-3 probe/profile/diag), `gpu_tests/` (CUDA parity checks), `tools/` (smoke + one-shot
  migrations). `premerge_check.py` and `run_instrument.sh` stay at `scripts/` root. See
  `scripts/README.md`. Experiments land in `experiments/*.json`; raw panels
  in `runs/` (gitignored).
- **Instruments emit panels.** Anything measuring the engine writes `runs/<id>.json` via
  `research.harness`, with a validity block. A number without its harness config and workload
  regime is not evidence.
- **Merging engine changes requires an experiment.** `scripts/premerge_check.py` refuses a merge
  that touches `src/inference_server/` without an `experiments/*.json` whose four gates are
  green. Docs, tests, scripts and `research/` are exempt.
- **Negative results are output, not failure.** Write them to `knowledge/` with status
  `rejected`. Seven such entries already exist and they are what stop the loop re-treading
  dead ends.
- **Never compare across sessions.** `compare.py` enforces this; do not work around it.

## What This Project Is

A production-grade, multi-user LLM inference engine built from scratch in Python, competing with **vLLM at the engine layer**. Target: real concurrent traffic, mixed prompt/output lengths, tight tail-latency.

**Engine-layer features (all in scope):** continuous batching, chunked prefill, paged KV cache with cross-session prefix sharing, fair scheduling + preemption hooks, backpressure under memory pressure, quantization / attention kernels / KV eviction policies.

**Out of scope:** platform layer — model registry, LoRA hot-swap, multi-tenant auth/quotas, OpenAI API translation, gateway features. Engine first; platform later.

**Primary metrics:** per-user p95 TTFT/TPOT under concurrent load, and throughput at a fixed tail-latency budget. Single-request latency is one point on the curve, not the goal.

**Comparison point:** vLLM, same model, same hardware.

---

## Key Design Decisions

- **Model:** Gemma 4 E2B-it (dev) → E4B (load/bench). HuggingFace Transformers.
- **Framework:** PyTorch.
- **Optimization priority:** throughput at p95 SLO > KV capacity > single-request latency > cold start.
- **Hardware:** auto-detect CUDA → MPS → CPU. Manual override via `DEVICE`.
- **Backend abstraction (`InferenceBackend`):** server never talks to model directly. Swap = config change.
- **Session IDs everywhere:** all per-request state scoped by `session_id`. Multi-tenancy/quotas plug in later without a rewrite.
- **Pluggable scheduler (`SchedulerInterface`):** continuous batching with fairness + preemption hooks.
- **Rich request objects:** `session_id`, arrival_ts, priority, max_tokens, timeout. Never bare token lists. Extensible by field-add.
- **Benchmarking:** every optimization measured against baseline; final compare vs vLLM.
- **Modal-deployable:** no host-machine assumptions (no hardcoded paths, ports, `localhost` — env/config only). One-time setup in a startup hook (model load, KV pre-alloc), never lazy on first request. All state in-process per-container, scoped by `session_id`. FastAPI + SSE only. `/metrics` pull-based and externally scrapable. No filesystem writes on the request path.

---

## Forward-Compatibility Constraint

Platform layers must slot in later without rewriting core. Keep:
- Rich request objects (add fields, don't refactor)
- `session_id` threading end-to-end (request → server → scheduler → backend → cache)
- `InferenceBackend` / `SchedulerInterface` / `CacheManager` abstractions
- No hidden global state outside those abstractions
- Standard HTTP surface (FastAPI + SSE)

Do **not** build yet: model registry, LoRA hot-swap, multi-tenant auth/quotas, OpenAI API translation, gateway features. If a task drifts here, stop and revisit scope.

---

## Architecture

| Component | Role |
|---|---|
| `InferenceBackend` | Swap backends via config |
| `Tokenizer` | Stateless |
| `BatchProcessor` | Continuous batching loop driven by `SchedulerInterface` |
| `CacheManager` | Methods scoped by `session_id`; enforces per-session limits |
| `EvictionPolicy` | Pluggable per-session decisions |
| `BlockManager` | Session-agnostic block pool — paged KV cache |
| `RadixTree` | Session-agnostic prefix matching for cross-session sharing |
| `Scheduler` | `FairScheduler` — per-session fairness, priority, preemption hooks |
| `Sampler` | `sampling.py::sample()` — temperature, top-k, top-p, greedy. Per-request `SamplingParams` carried on `ScheduledRequest`. |
| `server.py` | Thin HTTP — session routing only |
| `config.py` | Engine params first-class; platform params deferred |
| Prometheus metrics | All labeled by `session_id` |

### Architecture docs hard rule (`docs/architecture.html`)

Any change that adds/removes/alters a feature, component, queue, endpoint, env var, metric, or data-flow path **must** update both `architecture.html` and the relevant detail page (`arch-server.html`, `arch-scheduler.html`, `arch-cache.html`, `arch-backend.html`) in the same change. Update the "Last updated" date; verify SVG text fits its rect. If a page sprawls past a few diagrams, split it.

---

## Roadmap

### ✅ Phases 0–5 — complete
Project setup, tokenization, autoregressive loop, streaming, request batching, KV cache (block manager + radix tree + cache manager + LRU/AttentionSink/H2O eviction + cache wired through single + batched generation, `/cache/stats` endpoint, `eviction_benchmark.py`). Phase 5 has 4 contained gaps tracked in DECISIONS.md — do not redo Stages 1–7.

### Phase 6 — Engine-Level Concurrency & vLLM Parity (active)

- ✅ **Continuous batching** (`ContinuousBatchScheduler`) — iteration-level scheduling, immediate slot fill
- ✅ **Scheduler** — `FairScheduler` behind `SchedulerInterface`; FCFS + Fair/VTC policies (`SCHEDULING_POLICY` env); priority hooks; per-session admission primitives
- ✅ **Backpressure** — queue-level (HTTP 429) + KV-pressure-aware admission (active-KV gate + cache-pool gate), metrics on `/scheduler/stats`
- ✅ **Load-test client** — `scripts/bench/load_test.py` + `scripts/bench/plot_load_test.py`
- ✅ **In-server traffic simulator** — `inference_server/simulator.py` mounted at `/simulate/{start,stop,status}`. N async "users" loop streaming `/generate` calls against localhost with `session_id=sim-{i}`, weighted short/medium/long prompt mix from `simulator_prompts.py`, 50–400 ms jittered think-time. `/simulate/start` returns a `warning` field when `num_users >= max_batch_size` (no headroom for the operator's own live requests). Web UI reads `/simulate/status` for the live graph.
- ✅ **CUDA-ready backend** — `TorchBackend(device=...)`; factory routes `cuda|mps|cpu`
- ✅ **bf16 weights**, `compile_model` flag wired (default off)
- ✅ **Pre-allocated per-layer KV pools** in `BlockManager` (vLLM/PagedAttention layout)
- ✅ **M1 — Custom Gemma 4 forward** (`models/gemma4.py`) — byte-identical to HF on the parity fixture. Owns embedding, dual-RoPE, GQA attention with QK/V-norm and KV sharing, GeGLU MLP, per-layer embedding gating, softcapping. Reachable via `BACKEND=custom-{cuda,mps,cpu}`.
- ✅ **M2.1 — Custom KV cache** (`models/gemma4.py::KVCache`) — per-layer (K, V) tensors threaded through the forward; one prefill then incremental single-token decode. Cache-vs-no-cache logit parity tested. Shared layers correctly read the source layer's full cached + new K/V via the rebuilt `shared_kv` dict each call.
- ✅ **M2.2 — Block-paged KV cache** (`models/paged_kv_cache.py`) — `BlockPool` per non-shared layer with refcounted blocks; `PagedKVCache` per session holds per-layer block tables + seq_lens, allocates blocks lazily on append, gathers via `index_select` for SDPA, releases blocks on session end. KV-shared layers get a `None` pool slot. Byte-identical to contiguous KVCache on parity prompt; two-session pool-sharing test verifies no leaks. `CustomTorchBackend.generate`/`stream` switched to paged; pool size knobs `CUSTOM_BACKEND_BLOCKS` / `CUSTOM_BACKEND_BLOCK_SIZE`.
- ✅ **M2.2b — Cross-session prefix sharing** (`PrefixCache` in `paged_kv_cache.py`) — `PrefixCache.lookup(token_ids) → (matched_tokens, per-layer-blocks)` returns the longest aligned-prefix hit (sharing happens at full-block boundaries only). Blocks are refcounted: storing in the cache acquires; sessions claiming the prefix acquire too; `free_all` releases. `PagedKVCache(shared_prefix=...)` seeds the block tables + seq_lens so the model runs forward only on the suffix. Plumbed into `CustomTorchBackend`: lookup → suffix-only prefill → decode → store. Test `test_prefix_cache_hit_logits_match` shows byte-identical last-position logits with shared prefix; smoke shows **3.88× speedup** on a repeat 5-token prompt at block_size=2 (0.84s → 0.22s on CPU bf16, 4 of 5 tokens hit). `last_cache_hit_tokens` populated for downstream stats. **Not yet:** eviction (entries grow unbounded); a proper radix tree (currently a dict keyed by aligned-prefix tuples — handles same-prompt and "shared system prompt" cases but not partial-block sharing).
- ✅ **M2.4 — Scheduler-facing primitives on `CustomTorchBackend`** — implements `prefill`, `prefill_lookup/chunk/store`, `decode_step_batched`, `splice_into_batched`, `remove_row_from_cache`, `kv_length` so `ContinuousBatchScheduler` drives `BACKEND=custom-*` end-to-end with paged KV + prefix sharing. Batched KV = `list[PagedKVCache]` (one per row); decode is row-by-row (one forward/row, not GPU-batched — `attention_mask` ignored since each row's paged cache holds only its real tokens). `set_cache_adapter` is a no-op (custom backend caches via its own `PrefixCache`); blocks freed on eviction via `remove_row_from_cache → free_all`. Scheduler `_evict_row` now always routes through `remove_row_from_cache` so the last row's blocks don't leak. Tests in `test_custom_backend_scheduler.py` (output matches manual greedy; no block leak across repeats; concurrent shared-prefix hits + frees). **Not yet:** real single-forward batched decode, scheduler-visible pool-pressure backpressure (pool exhaustion → request rejection), custom prefix-cache stats on `/cache/stats`.
- ⏳ **Chunked prefill — Version A (alternating)** — in progress, MPS-feasible
- 📋 **Modal deploy** — `modal_app.py` written; CLI auth + `modal deploy` + smoke-test `/health` `/ready` `/generate` (HANDOFF.md)
- 📋 **FlashAttention-2** — CUDA-only; flip `attn_implementation="flash_attention_2"` + add `flash-attn` to Modal image. Land before the load-test sweep so baseline reflects shipped config.
- 📋 **Real load-test sweep** — held until CUDA; run after FlashAttention is on
- 📋 **Swap to E4B**, flip `COMPILE_MODEL=true` on CUDA
- 📋 **vLLM head-to-head** — throughput, p50/p95/p99 TTFT/TPOT, KV utilization, saturation
- 📋 **PagedAttention kernel (decision point)** — only if head-to-head shows we're attention-bound. Requires writing our own forward pass to call vLLM's paged-attention kernel (or Triton equivalent) against our existing block pool. Big lift, model-arch-specific.
- ⏸ **Preemption, V-B mixed-batch prefill, per-session KV quotas, batch utilization %, MLX cache integration** — deferred (DECISIONS.md)

### Phase 7+ — outline

- **7. Hardware auto-detection** — CUDA/MPS/CPU, `DEVICE` override, startup summary, auto-size KV cache
- **8. Observability** — structured JSON logs, Prometheus metrics, Grafana dashboard, timing middleware
- **9. Resilience** — graceful shutdown, request timeouts, `/health` + `/ready`, error isolation
- **10. Containerization** — Dockerfile, docker-compose with Prometheus/Grafana, `.env.example`
- **11. Benchmarking** — single `scripts/benchmark.py`, vs vLLM charts, `BENCHMARKS.md`
- **12. CI** — GitHub Actions, regression gates, ruff + mypy, `DESIGN.md`

### Future extensions

- **MLX continuous batching backend** — requires custom MLX forward loop (mlx_lm doesn't expose the primitives we need)
- **Realistic load simulator** — async, distribution-drawn lengths, remote VM as load source

### Optional (pick what's interesting)

Conversation persistence (SQLite `ConversationStore`, zero hot-path coupling) · Agent/MCP integration · Quantization · Paged attention · Flash attention · Tensor parallelism · Speculative decoding · gRPC · K8s autoscaling · OpenTelemetry tracing · Web UI · Python client lib · Response style system (caveman-inspired token compression)

---

## Current Status

**Active:** Phase 6 — M2.4 done (custom backend scheduler-driven). Next: Modal deploy of `BACKEND=custom-cuda` + load-test sweep.
**Next concrete step:** deploy `BACKEND=custom-cuda` to Modal (`modal deploy modal_app.py`), then run `scripts/bench/load_test.py` + `scripts/bench/plot_load_test.py` against it. Numbers decide the next optimization: decode GPU-bound → M2.3 Triton paged-attention kernel; queueing-bound → tune scheduler; KV-pressure-bound → PrefixCache LRU + block-size tuning + scheduler-visible pool backpressure. Later: real batched `decode_step_batched`, chunked prefill V-A on CUDA, FlashAttention (ruled out for Gemma 4 — head_dim>256), E4B + `COMPILE_MODEL=true`, vLLM head-to-head.
