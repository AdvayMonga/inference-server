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

**How a change lands:** branch → PR → `ci-ok` green → review → merge. Never push to `main`.
Setup, the three CI lanes and the three evidence paths are in [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## How we work now — the research loop

**`LOOP.md` is the method. Follow it; do not improvise around it.** Ad-hoc optimisation is what
produced this project's two most expensive mistakes (a month on the wrong bottleneck from a
closed-loop harness, and a cache-hit benchmark that hid every miss-path bug).

```bash
python -m inference_server.research.loop attribute runs/<id>.json --out gaps.json
python -m inference_server.research.loop kb --status rejected     # what is already disproved
python -m inference_server.research.loop kb --regime cold_start   # what applies to a regime (--situation k=v narrows by validity_range)
python -m inference_server.research.loop screen hypotheses.json   # cheapest falsification first, flags known dead ends
python -m inference_server.research.loop simulate --class steady_interactive --config '{"policy":"fair"}'  # tier 1: policy hypotheses, no GPU
python -m inference_server.research.loop no-claim --why "..."      # engine diff that claims no behaviour change
python -m inference_server.research.loop judge --hyp H.json --baseline A.json --treatment B.json
python scripts/bench/replay_trace.py --class steady_interactive --split seen --base-url URL  # open-loop corpus replay
RUNPOD_API_KEY=... scripts/tools/run_on_runpod.py scripts/tools/venue_smoke.py --gpu '...'  # rent a GPU, run one instrument, terminate
```

- **Workload corpus (2026-09-16):** `corpus/` holds frozen traces for three classes (`cold_start`,
  `steady_interactive`, `long_context`), each split `seen` / `heldout`, hashed into a
  `corpus_version` that `replay_trace.py` stamps on every panel and `compare.py` refuses to
  compare across. Optimise against `seen`; a win must replicate on `heldout`. A changed trace
  is a new version (`scripts/tools/build_corpus.py`), never an edit. Class SLOs are placeholders
  until the plan's Phase 0 decision. See `corpus/README.md`.
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
  that touches `src/inference_server/` without an `experiments/*.json` whose five gates are
  green — or a correctness-fix or no-claim record (see CONTRIBUTING.md). Docs, tests,
  scripts, `static/` and `research/` are exempt.
- **Negative results are output, not failure.** Write them to `knowledge/` with status
  `rejected`. Nine such entries already exist and they are what stop the loop re-treading
  dead ends.
- **Never compare across sessions.** `compare.py` enforces this; do not work around it.

## What This Project Is

A single-GPU LLM inference engine written from scratch in Python, plus the research loop that
improves it — on its way to becoming a **regime-adaptive, multi-replica serving system whose
policies are learned offline by the loop and selected at runtime by a controller**. The plan of
record is the three-plane design below; the first regime is cold start, because a replica's
first thirty seconds are deterministically different from its steady state and the regime is
detected by reading the replica's age, no classifier needed.

### Three planes

| plane | clock | role | status |
|---|---|---|---|
| **Data** — `src/inference_server/` minus `research/` | microseconds | serves tokens; emits one telemetry row per request | built (see *Engine already built*) |
| **Control** — router, global prefix index, replica launcher, controller, policy table | seconds | routes, selects a policy, manages replica lifecycle | does not exist yet (Phases 5–7) |
| **Improvement** — `research/`, `corpus/`, `knowledge/`, `scripts/` | hours | turns telemetry into validated configs; changes the other two | built; extended by Phases 1–3 |

Flow: the data plane emits telemetry → the improvement plane turns it into validated configs →
the control plane selects among them at runtime. The knowledge base is both research artifact
and production policy store. Nothing in the data plane ever calls a model for judgment;
judgment is compiled into a table offline.

### The metric

**Primary:** GPU-seconds per session, subject to p95 TTFT under a fixed per-class ceiling
(placeholders in `corpus/manifest.json` until Phase 0 sets them). Cold start is a latency/cost
tradeoff; pinning one and optimising the other avoids inventing weights, counts idle warm-pool
time, and is measurable entirely from outside the system. **Guardrail:** warm-path TTFT/TPOT
stay within their current distance of vLLM. The vLLM head-to-head is the guardrail measurement,
not the objective.

**Objective hierarchy** — higher overrides lower; a win that breaks a level above is not a win:
1. **Hard invariants** — output equivalence against the reference path, memory ceiling, no crashes.
2. **Guardrails** — no workload class regresses beyond epsilon; warm-path distance to vLLM.
3. **Primary metric** — the one number maximised.

### Where we stand

A100-80GB, Gemma 4 E4B, 32 concurrent, June 2026 (`benchmarks/`): this engine **1151 tok/s,
TPOT p50 24 ms, TTFT p50 107 ms**; vLLM 0.23 **2628 tok/s, 11 ms, 48 ms**. 9× over naive HF,
2.3× behind vLLM. The loop's attribution says the rest of that gap is per-decode-step cost at
low batch and long-prompt prefill compute — kernel and fusion work on vLLM's home turf — so we
stopped racing it and kept parity as a guardrail. Three loop findings constrain everything:

- The TTFT tail is prefill **compute** (203 ms of 213 ms at rate 1), not queueing. Every scheduling lever died at once.
- A100 draws are bimodal (2.3× TPOT spread on identical config). Only simultaneous A/B arms are evidence.
- A 10.8× isolated kernel win was worth nothing end to end.

**Out of scope:** platform layer — model registry, LoRA hot-swap, multi-tenant auth/quotas, gateway features. Engine first; platform later. **Narrow exception:** `openai_shim.py` serves `/v1/completions`, `/v1/chat/completions` and `/v1/models`. It exists so one load generator can drive both us and vLLM in the head-to-head, and so Open WebUI can be the chat frontend instead of us maintaining one. Compatibility surface, not a public API: no auth, registry, tools, logprobs, or multimodal parts. Accepts optional `X-Session-Id` / `X-Turn-Index` / `X-Trace-Id` headers for the harness (absent → auto ids) and echoes `X-Trace-Id`.

---

## Key Design Decisions

- **Model:** Gemma 4 E2B-it (dev) → E4B (load/bench). HuggingFace Transformers. Phase 0 may move the benchmark model to a ~30 GB class so cold start is dramatic.
- **Framework:** PyTorch.
- **Optimization priority:** hard invariants > guardrails (warm-path distance to vLLM, no class regresses) > GPU-seconds per session at the p95 TTFT ceiling. Cold start is the first regime; warm-path latency is a guardrail, not a goal.
- **Hardware:** auto-detect CUDA → MPS → CPU. Manual override via `DEVICE`.
- **Backend abstraction (`InferenceBackend`):** server never talks to model directly. Swap = config change.
- **Session IDs everywhere:** all per-request state scoped by `session_id`. Multi-tenancy/quotas plug in later without a rewrite.
- **Pluggable scheduler (`SchedulerInterface`):** continuous batching with fairness + preemption hooks. Ordering decisions are pure functions in `scheduling_policy.py` so the simulator shares them.
- **Rich request objects:** `session_id`, `trace_id`, `turn_index`, arrival_ts, priority, max_tokens, timeout. Never bare token lists. Extensible by field-add.
- **Benchmarking:** every engine change lands with an experiment record judged by the loop (`LOOP.md`); policy hypotheses die in the simulator before they cost a GPU; vLLM is the warm-path guardrail arm.
- **Modal-deployable:** no host-machine assumptions (no hardcoded paths, ports, `localhost` — env/config only). One-time setup in a startup hook (model load, KV pre-alloc), never lazy on first request. All state in-process per-container, scoped by `session_id`. FastAPI + SSE only. `/metrics` pull-based and externally scrapable. No filesystem writes on the request path (telemetry writes happen off the scheduler thread and only when `TELEMETRY_DIR` is set).

---

## Forward-Compatibility Constraint

Platform layers must slot in later without rewriting core. Keep:
- Rich request objects (add fields, don't refactor)
- `session_id` threading end-to-end (request → server → scheduler → backend → cache)
- `InferenceBackend` / `SchedulerInterface` / `CacheManager` abstractions
- No hidden global state outside those abstractions
- Standard HTTP surface (FastAPI + SSE)

Do **not** build yet: model registry, LoRA hot-swap, multi-tenant auth/quotas, gateway features, or any widening of the OpenAI shim past the three routes it has. If a task drifts here, stop and revisit scope.

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
| `SchedulingPolicy` | `scheduling_policy.py` — pure `fcfs_key` / `fair_key` (+ `fcfs_order` / `fair_order`) / `fair_initial_counter` / `fair_charge` over any `(session_id, priority, arrival_seq)` record; `FCFSPolicy` / `FairPolicy` are thin stateful shells. Simulator and scheduler share them. |
| `Sampler` | `sampling.py::sample()` — temperature, top-k, top-p, greedy. Per-request `SamplingParams` carried on `ScheduledRequest`. |
| `server.py` | Thin HTTP — session routing only |
| `config.py` | Engine params first-class; platform params deferred |
| `research/corpus.py` | `corpus/` manifest + traces: frozen, hashed per class (`cold_start`, `steady_interactive`, `long_context`), split seen / held-out; `corpus_version` in every panel's validity block |
| `research/simulator.py` | Tier-1 falsifier: discrete-event replay of a corpus trace through the scheduler's iteration order with a fitted `TimingModel` in place of attention; shares `scheduling_policy.py`; `loop simulate` |
| `telemetry.py` | `RequestRecord` + `RowStore` — one SQLite row per request (conditions at arrival, spans, outcome; `trace_id`/`session_id`/`turn_index`), written off the scheduler thread. Off unless `TELEMETRY_DIR` is set |
| Prometheus metrics | All labeled by `session_id` |

### Architecture docs hard rule (`docs/architecture.html`)

Any change that adds/removes/alters a feature, component, queue, endpoint, env var, metric, or data-flow path **must** update both `architecture.html` and the relevant detail page (`arch-server.html`, `arch-scheduler.html`, `arch-cache.html`, `arch-backend.html`) in the same change. Update the "Last updated" date; verify SVG text fits its rect. If a page sprawls past a few diagrams, split it.

---

## Roadmap

Phases from the plan of record. Dependencies are strict; each phase is independently demoable
and benchmarkable, so the project produces results even if later phases are cut.

### Engine already built (data plane) — do not re-plan

Continuous batching (`ContinuousBatchScheduler`, iteration-level) · monolithic / chunked / batched prefill · block-paged KV cache with refcounted blocks, window-aware sliding layers and a radix `PrefixCache` for cross-session sharing · Triton paged decode + prefill kernels (split-K decode) · bucketed CUDA-graph decode, `torch.compile`, int8 weight-only quantization · custom Gemma 4 forward, byte-identical to HF on the parity fixture · FCFS / fair (VTC) scheduling with priority hooks · queue + KV backpressure (HTTP 429), admission deadline, preemption under KV pressure · FastAPI + SSE with `session_id` end to end, OpenAI shim · Prometheus + Grafana (`monitoring/`). Detail lives in `docs/arch-*.html`; engine work measured and set aside (mixed-batch prefill, tiled prefill attention, length-grouped waves, per-session KV quotas, MLX) is in `knowledge/` as `rejected` / `deferred` entries.

### Phase 0 — decide 🔲 open

Cheap, and everything downstream depends on it.
- **Model and size.** E4B loads too fast for cold start to be dramatic; a ~30 GB class makes the story real.
- **Substrate.** Modal credits are exhausted (2026-09-07). Rented pods via `research/venues.py` are the executor today; whether SLURM is worth a second launcher, and whether it has node-local NVMe and a fast interconnect (decides migrate-vs-recompute in Phase 6).
- **KV bytes per token** for the chosen model — drives every threshold in Phases 5–6.
- **SLO ceiling per workload class.** `corpus/manifest.json` holds placeholders.
- **Loop authority in the adaptive phase:** policy-only, or code too.

### Phase 1 — per-request telemetry ✅ (PR #18, 2026-09-16)

- ✅ `telemetry.py`: one row per request — conditions at arrival snapshotted in `enqueue()` before any work (so a 429 still records the load it saw), spans from the scheduler timestamps, outcome on every terminal path. `trace_id` / `turn_index` on `ScheduledRequest`, echoed by `/generate` and the shim. SQLite, one file per run under `TELEMETRY_DIR`, written off the scheduler thread; `tests/test_telemetry.py` holds per-request cost under 200 µs.
- ⏸ **Deferred** (`kb-20260915-2c4513a1`): CUDA-event device spans; prefix-cache state *at arrival* (only post-hoc `cache_hit_tokens` today); block alloc/free/evict counters; KV high-water; CUDA-graph hit/miss by bucket; preemptions caused vs suffered; detokenize and per-chunk prefill spans. `config_id` is None until a policy registry exists (Phase 7).

### Phase 2 — corpus and harness (PRs #17, #21)

- ✅ `corpus/`: three classes (`cold_start`, `steady_interactive`, `long_context`), each split `seen` / `heldout`, hashed into a `corpus_version` that rides in the validity block and `compare.py` refuses across. `scripts/bench/replay_trace.py` replays a trace open-loop on the client's own clock, posting through the shim with `X-Session-Id` / `X-Turn-Index` / `X-Trace-Id` so its CSV row joins the telemetry row.
- ⏳ **Total accounting** from process start (the panel has peak host/device memory; process-start-to-first-token is not yet in it).
- ⏳ **Noise floor** procedure per class per hardware, banded and filed in `knowledge/`; the A100 bimodality quantified rather than remembered.
- ⏳ **Held-out replication in the merge gate.** ≥3 runs per arm is enforced today; a confirmed win replaying on `heldout` before merge is not.

### Phase 3 — simulator ✅ built (PRs #20, #22; KB regime fields #19, #23)

- ✅ `scheduling_policy.py`: ordering decisions as pure functions the scheduler and simulator share. `research/simulator.py`: discrete-event trace replay through the loop's iteration order with attention replaced by a `TimingModel`; `loop simulate --class <cls> --config '{...}'`; panels carry `harness="simulator"` and cannot be compared to hardware. `fit_timing_model` (from replay/telemetry rows) and `rank_correlation` exist. Knowledge entries carry `regime` + `validity_range`; `loop kb --regime / --situation` queries them.
- ⏳ **Fit the timing model from real rows** — today's default is the `PLACEHOLDER_A100_E4B` coefficients.
- ⏳ **Hardware rank-correlation validation** as a scheduled experiment; tier-1 results are suspended if it drifts.
- ⏸ v1 does not model preemption, chunked prefill, prefix-cache eviction, or cache-held blocks against the pool.

### Phase 4 — cold start 📋 (first result against the thesis metric)

Snapshot/restore baseline on the chosen substrate; layer-ordered weight streaming with prefill overlap; graph-capture cache; cold TTFT decomposition by component; GPU-seconds per session at fixed SLO — warm pool vs scale-to-zero vs snapshot. Stretch: warm draft-model hedging.

### Phase 5 — router and multi-replica 📋 (needs two GPUs)

`ReplicaLauncher` interface (rented-pod and local implementations); centralized global prefix index; prefix-aware session-affine router with a locality-vs-load knob, searched in the simulator first and confirmed on hardware.

### Phase 6 — session migration 📋

Session state manager (quiesce, serialize, re-layout on arrival); append-only pipelined KV copy vs multi-round token recompute behind one interface so migrate-vs-recompute is a searchable knob; copy-verify-switch atomicity. Demo: the crossover plot vs context length and interconnect.

### Phase 7 — controller and policy table 📋

Policy registry in the data plane (cheap knobs swappable at a step boundary); policy table as a view on the knowledge base; safe default trained on minimax regret; controller with replica-age regime detection, hysteresis, dwell time, fallback logging; pinned vs adaptive mode in the harness; current-policy readout and override. Demo: controller-on vs best fixed config across a mixed-regime traffic mix — if the controller cannot beat the best single config, that is a publishable negative result.

### Stretch — probably not this year

Blue-green reconfiguration with cross-config migration · disaggregated prefill/decode · spot-eviction handling · multi-tenant LoRA.

**Standing risk:** scope creep back into the engine-vs-vLLM race. The warm path is a guardrail, never the objective.

---

## Current Status

**Built:** Phases 1–3 (PRs #17–#23, merged 2026-09-16). 517 tests; 483 run on CPU, 34 model-heavy ones opt in with `-m heavy`. `knowledge/` holds 69 entries.
**Next:** Phase 4, gated on the Phase 0 decisions. The first GPU job is the simulator's hardware check: run the engine with `TELEMETRY_DIR` set on a rented pod (`scripts/tools/run_on_runpod.py`, needs `RUNPOD_API_KEY`; smoke the venue first with `scripts/tools/venue_smoke.py`), replay a corpus class with `scripts/bench/replay_trace.py`, `fit_timing_model` on the telemetry rows, then a policy sweep in `loop simulate` vs the same sweep on hardware → `rank_correlation`, filed in `knowledge/`.
**GPU budget:** Modal credits ran out on 2026-09-07. The `*_modal.py` instruments and `run_instrument.sh` still exist; new GPU work runs cheap tiers locally first and only then on a rented pod through `research/venues.py`.
