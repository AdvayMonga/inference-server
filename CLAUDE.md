# CLAUDE.md

## Recency
- My most recent direction overrides anything written down — docs, memory, old plans.
  Plans are hypotheses; the vision changes as we learn.
- If something written conflicts with what I said recently, follow me and flag the stale source.
  Don't quote old plans back at me as a reason not to change course.
- Exception: measured findings (BlameGraph `RUNS.md` and its ledger) stand until re-measured. If a new
  direction runs into one, raise it once, then do what I decide.

## How we discuss
- When I'm thinking out loud, give your opinion and push back; don't just agree.
- Keep answers short. Recommend one option, don't survey.
- Explain the non-obvious design choices (what, why, alternatives), simply, only on the big ones.

## Context
- Don't maintain process docs (plans, handoffs, logs). Git history, PRs, and memory carry state.
- Derive what the project is from the code, git log, and recent memory — not from assumptions.

## Models
- Hand-written forward: `models/qwen3_moe.py` (Qwen3-MoE only; the Gemma 4 forward it was first
  written for was removed 2026-10-17). Qwen3-30B-A3B is the benchmark model (owner's Phase 0
  decision, 2026-10-01).
- Direction (2026-10-04): one engine that adapts to load across every corpus class; cold start is
  one class among them, no longer the headline. The headline metric is not decided yet.

## Strict referee, free player (the bitter lesson)
- Harness is code and strict: sandbox, grader, measurement protocol, output equivalence,
  hidden held-out data, budgets. It defines what counts as a win and keeps it honest.
- Strategy is not harness. The loop gets facts only — the raw ledger, its measurements, the limits
  it is judged against — never heuristics or advice, in prompts or anywhere (BlameGraph, 2026-10-06),
  and there is no knowledge base of beliefs, hand-written or the agent's (2026-10-09). Only
  integrity failures (gaming, escaping) become hard rules; a bad idea is what the lab is for.
- When we design the harness, flag anything that drifts from referee into strategy.

## This repo is the engine; the environment is BlameGraph
- This repo holds only the engine and its stack: `src/` (engine, control plane), its tests, `monitoring/`,
  and engine tools in `scripts/` (Triton launch tuning, CUDA parity checks, smoke). The agent's workspace is
  exported from here; it may write `src/inference_server/` and add `tests/test_*.py`, nothing else.
- Everything that runs, measures or judges the engine lives in github.com/AdvayMonga/BlameGraph (split
  2026-10-06; tag `archive/pre-split` here holds the combined tree): the lab (session loop, agent, tools,
  ledger, workspace, sandbox/grader, budget), the GPU VMs (`python -m lab.vm`, Verda/Nebius/Crusoe), the
  workload corpus, `knowledge/`, load regimes, the correctness gate, canaries and the referee. It finds this
  repo via `LAB_ENGINE_REPO` (default `../inference-server`) and runs its code with `.venv/bin/python`.
- `/v1/completions` accepts token-id prompts and vLLM's `prompt_logprobs: k` (teacher-forced scoring via
  `CustomTorchBackend.score_logprobs`), the output-equivalence surface the environment uses.
- Serving on the H200 (short tier, measured 2026-10-18, BlameGraph `RUNS.md` has every run): single_stream
  TPOT p99 6.0 ms (was 21 on 2026-10-15), saturated 3.8 req/s (was 0.097), long_prompt 12-14 req/s (was invalid);
  vLLM 4.1 ms / 5.8 / 68. Defaults: mixed prefill (`PREFILL_MODE`, 8192-token steps capped at `MAX_PREFILL_TOKENS`
  rows x longest), async decode, FlashInfer paged prefill attention (`CUSTOM_BACKEND_PREFILL_ATTN`, fa3), decode
  CUDA graphs per row bucket, Triton MoE path for decode-sized batches (`models/moe_kernel.py`: one-launch router,
  grouped GEMMs, scatter epilogue) and CUTLASS grouped GEMM with a gather epilogue above it. The batch-1 decode step
  is ~20 launches a layer and kernel-time-bound. `CUSTOM_BACKEND_QUANT=fp8` experts: correct, ~4% at wide batch.
- Launch profiles (2026-10-14): `src/inference_server/profiles.toml` holds `[general]` plus optional tables named after
  BlameGraph's regimes; `profiles.py` writes `[general]` and the table named by `ENGINE_PROFILE` into the env before
  `config.py` loads settings, so a profile overrides the launch env. BlameGraph serves a regime with its own table in
  its own launch and gates that config with its own equiv. Ships empty: one config for everything.
- `TIMELINE_DIR` turns on the engine event timeline (`timeline.py`): one JSONL event per scheduler
  decision and a profiler range per phase, both keyed by step id. Off by default, no-op when off.
