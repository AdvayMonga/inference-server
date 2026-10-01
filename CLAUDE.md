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
  `CustomTorchBackend` picks one by the HF `model_type`. Qwen3-30B-A3B is the Phase 0 benchmark
  model for MoE cold start; its path runs without CUDA graphs, torch.compile or int8 for now.

## The research loop
- The method is `src/inference_server/research/method/graph.py`: a fixed graph of code, agent
  and human nodes. `method/state.py` saves each turn as append-only JSON and refuses
  out-of-order records. `method/runner.py` walks one turn: it picks each node's inputs (the
  latest record of each type it `takes`), calls it, saves the result; `pause` = supervised mode.
- The loop makes `perf:` changes only: engine code (`src/inference_server/` minus `research/`)
  plus new test files; never existing tests, the harness, or anything else. Other kinds are
  human sessions.
- The referee is `research/safety/`: `change_kinds.py` (deny-by-default write surfaces),
  `jail.py` (sandbox-runtime settings), `grader.py` (audit + jailed pristine tests).
- `research/agents/`: `brief.py` renders an agent node's context (its input records, knowledge,
  the graph map); `call.py` is the one place a model is called — Agent SDK CLI inside the srt
  jail, env wiped, write hook on. `research/nodes/` holds one module per node (`build.py`,
  `check.py`, `review_1.py`). `review_1` runs `/code-review` on the clean tree (a two-commit
  repo: base, change) read-only; code turns its findings into the verdict. `check` runs the suite with `-m "not heavy and not needs_host"`: tests marked
  `needs_host` need what the jail forbids (held-out corpus, git, sockets) and run in CI only.
- Agents run inside nodes and never edit `graph.py`. Failures go back to `build` as context;
  only a security violation or an anomalous win stops the turn for me. Every agent may leave a
  `note` for me, which is recorded and never changes the path. Budgets per turn live in
  `graph.py`: 6 rounds (reviewer send-backs), 2 experiments, a node-visit backstop.
- The workload corpus (`corpus/`) is built from real public traces: BurstGPT arrivals and
  sessions, WildChat-1M conversations with their real assistant replies (multi-turn requests
  carry `messages`), four classes (`cold_start`, `steady_interactive`, `long_context`, `spike`)
  split seen/heldout by alternating trace weeks. `scripts/tools/fetch_traces.py` +
  `build_corpus.py` rebuild it; any change is a new `corpus_version`. See `corpus/README.md`.

## Strict referee, free player (the bitter lesson)
- Harness is code and strict: sandbox, grader, measurement protocol, output equivalence,
  hidden held-out data, budgets, graph order. It defines what counts as a win and keeps it honest.
- Strategy is not harness. Heuristics about *how* to find a win live only in prompts, as advice
  that never blocks. Test: if the model got 10x smarter, would this rule be unnecessary? Then
  it's a heuristic — keep it soft or drop it.
- The knowledge base informs, never forbids. Only integrity failures (gaming, escaping) become
  hard rules; a bad idea is what the loop is for.
- When we design the harness, flag anything that drifts from referee into strategy.
