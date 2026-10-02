# CLAUDE.md

## Recency
- My most recent direction overrides anything written down — docs, memory, old plans.
  Plans are hypotheses; the vision changes as we learn.
- If something written conflicts with what I said recently, follow me and flag the stale source.
  Don't quote old plans back at me as a reason not to change course.
- Exception: measured findings (`knowledge/`) stand until re-measured. If a new direction
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
  for MoE cold start (owner's Phase 0 decision, 2026-10-01); its path runs without CUDA graphs, torch.compile or int8 for now.

## Strict referee, free player (the bitter lesson)
- Harness is code and strict: sandbox, grader, measurement protocol, output equivalence,
  hidden held-out data, budgets. It defines what counts as a win and keeps it honest.
- Strategy is not harness. Heuristics about *how* to find a win live only in prompts, as advice
  that never blocks. Test: if the model got 10x smarter, would this rule be unnecessary? Then
  it's a heuristic — keep it soft or drop it.
- The knowledge base informs, never forbids. Only integrity failures (gaming, escaping) become
  hard rules; a bad idea is what the lab is for.
- When we design the harness, flag anything that drifts from referee into strategy.

## The lab
- `lab/` is the open environment (design: `ENVIRONMENT.md`). The research loop it replaced was
  removed on 2026-10-02; tag `archive/research-loop` keeps it. `python -m lab.profile` writes a
  raw measurement bundle per run.
- The workload corpus (`corpus/`) is built from real public traces: BurstGPT arrivals and
  sessions, WildChat-1M conversations with their real assistant replies, four classes
  (`cold_start`, `steady_interactive`, `long_context`, `spike`) split seen/heldout by alternating
  trace weeks. `scripts/corpus/fetch_traces.py` + `build_corpus.py` rebuild it; the loader is
  `lab/corpus.py`; any change is a new `corpus_version`. See `corpus/README.md`.
- `python -m lab.session` is the runtime: `lab/session.py` loops sessions over a dollar budget
  (`lab/budget.py`), `lab/agent.py` is the one place a model is called (Claude via the Agent SDK,
  provider seam), `lab/tools.py` the agent's metered tools (each snapshots the workspace and writes
  the ledger), `lab/workspace.py` the exported engine copy with content-addressed snapshots, and
  `lab/safety/` the referee (surfaces, jail, grader). Lab tools run outside the jail; the agent's
  own shell and file tools run inside it. `LAB_NO_JAIL=1` waives the jail (tests, trusted box).
- `python -m lab.vm` drives one persistent GPU VM (`LAB_VM_*` env) on `lab/providers/` verda
  (default, 1x H100 SXM) or crusoe (1x A100): start, setup, run a command on the pushed tree and
  fetch outputs, stop (Verda: hibernate). Credentials stay in env or the provider CLI's config.
- `lab/ledger.py` is the raw knowledge base: append-only JSONL, one record per tool call, written by
  tools only; `claim` holds the agent's untrusted words; held-out records carry one aggregate per metric.
  `python -m lab.ledger seed` imports `knowledge/` as `finding` records. `lab/ledger/` is gitignored.
- `TIMELINE_DIR` turns on the engine event timeline (`timeline.py`): one JSONL event per scheduler
  decision and a profiler range per phase, both keyed by step id. Off by default, no-op when off.
