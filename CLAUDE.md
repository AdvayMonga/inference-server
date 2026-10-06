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
- Hand-written forwards: `models/gemma4.py` (Gemma 4) and `models/qwen3_moe.py` (Qwen3-MoE);
  `CustomTorchBackend` picks one by the HF `model_type`. Qwen3-30B-A3B is the benchmark model
  (owner's Phase 0 decision, 2026-10-01); its path runs decode CUDA graphs (2026-10-04) but not torch.compile or int8 yet.
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
- `TIMELINE_DIR` turns on the engine event timeline (`timeline.py`): one JSONL event per scheduler
  decision and a profiler range per phase, both keyed by step id. Off by default, no-op when off.
