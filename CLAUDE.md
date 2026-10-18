# CLAUDE.md

## Recency
- My most recent direction overrides anything written down — docs, memory, old plans.
  Plans are hypotheses; the vision changes as we learn.
- If something written conflicts with what I said recently, follow me and flag the stale source.
  Don't quote old plans back at me as a reason not to change course.
- Exception: measured findings (BlameGraph `knowledge/`) stand until re-measured. If a new direction
  runs into one, raise it once, then do what I decide.

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
- Strategy is not harness. Heuristics about *how* to find a win live only in prompts, as advice
  that never blocks. Test: if the model got 10x smarter, would this rule be unnecessary? Then
  it's a heuristic — keep it soft or drop it.
- The knowledge base informs, never forbids. Only integrity failures (gaming, escaping) become
  hard rules; a bad idea is what the lab is for.
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
- Serving on the H200 (measured 2026-10-15, Verda, bf16, short tier): single_stream TPOT p99 15.6 ms (was 21),
  saturated 0.37 req/s (was 0.097), long_prompt 2.66 req/s (was invalid); vLLM 4.1 ms / 5.8 / 68. The decode step
  is launch-bound: ~4,500 kernels per step before `models/fused.py` (RMSNorm, add+RMSNorm, RMSNorm+RoPE, SiLU*up,
  routing; `CUSTOM_BACKEND_FUSED=0` disables), ~2,300 after; batch-1 step 24 → 13.3 ms. Prefill attention goes
  through torch SDPA (`models/prefill_attention.py`): the Triton paged prefill kernel ran at ~5 TFLOP/s against
  SDPA's ~600 (`CUSTOM_BACKEND_PREFILL_ATTN=triton` keeps it). KV pools take `KV_CACHE_MEMORY_FRACTION` (0.9)
  of the memory free after a measured prefill of `MAX_PREFILL_TOKENS` (32768) tokens, which also caps a batched
  prefill wave (rows x longest prompt); 90% of raw free memory OOMed on long prompts. `MAX_BATCH_SIZE` 128,
  scheduling `fair`. Streaming: a request shed before its first token is a 429, after it the stream ends short.
  Off by default, measured: `CUSTOM_BACKEND_QUANT=fp8` experts (correct, no gain while launch-bound, frees 26 GB);
  `CUSTOM_BACKEND_MOE_KERNEL_ROWS` > 0 (Triton grouped GEMM, `models/moe_kernel.py`: exact in eager mode, a
  device-side assert inside the decode CUDA graph under load; the CUTLASS grouped GEMM it replaces is 3.5 ms of
  the 13 ms step). Correctness gate on this state: inconclusive at concurrency 32 (12/1,585 unanswered as 429s,
  ratio 0.9951 on the rest); re-run at 16. Tag `loop-baseline` = the engine before all of this.
- Launch profiles (2026-10-14): `src/inference_server/profiles.toml` holds `[general]` plus optional tables named after
  BlameGraph's regimes; `profiles.py` writes `[general]` and the table named by `ENGINE_PROFILE` into the env before
  `config.py` loads settings, so a profile overrides the launch env. BlameGraph serves a regime with its own table in
  its own launch and gates that config with its own equiv. Ships empty: one config for everything.
- `TIMELINE_DIR` turns on the engine event timeline (`timeline.py`): one JSONL event per scheduler
  decision and a profiler range per phase, both keyed by step id. Off by default, no-op when off.
