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
- Serving on the H200 (short tier, measured 2026-10-21, BlameGraph ledger runs `manual-20261020/21`): single_stream
  TPOT p99 4.47 ms, saturated 10.4 req/s, long_prompt 80, short_prompt_long_output 22.5 (24.1 at 2,048 rows), bursty
  256x at TPOT p99 26 ms; vLLM 0.32 on the same corpus 4.10 / 5.47 / 80 / 22.5 / 256x at 22 ms. Defaults: the
  backend and scheduler in a child process (`ENGINE_PROCESS`, `engine_process.py`), 1,024 rows (`MAX_BATCH_SIZE`),
  mixed prefill (`PREFILL_MODE`, 8192-token steps; chunks of <= 128 tokens run as decode rows through the decode
  graphs), async decode with first tokens read asynchronously, FlashInfer paged prefill attention
  (`CUSTOM_BACKEND_PREFILL_ATTN`), decode CUDA graphs per row bucket captured at load, the decode-step megakernel at
  batch <= 2 with its weights streamed through a TMA shared-memory ring (`MEGAKERNEL_TMA`; the KV dtype, the ring and
  blocks per SM are build-time flags of the extension), Triton MoE GEMMs up to 8,192 pairs with 16/32/64-row tiles
  and fp8 activations from 2,048 pairs (`models/moe_kernel.py`, table in `qwen3_moe.py`), CUTLASS grouped GEMMs
  above. One block table per request, shared by every layer (`paged_kv_cache.BlockAllocator`); KV admission reserves
  the p90 of recent output lengths and preempts/requeues a row that outgrows it; `KV_CACHE_DTYPE=fp8` pools are
  opt-in (2x the pool, ~25% slower large-batch steps until the attention kernel reads fp8 natively). The server
  raises its open-files limit; the scheduler runs its own GC (young every 200 steps, full every 50k).
- Speculative decoding by prompt lookup (`SPEC_DECODE_K` drafts a step, off by default; `SPEC_DECODE_ROWS` running
  rows at most, greedy rows only): `scheduler.NgramDrafter` proposes the tokens that followed the context's last
  3-gram, `backend.verify_step` runs [current] + drafts as rows at consecutive positions in one forward (the
  prefill-as-decode mechanism, K/V written causally), the longest agreeing prefix is kept; `stats()` reports
  spec_steps / spec_drafted / spec_tokens. Output-identical to plain greedy decode (tests/test_spec_decode.py).
- Launch profiles (2026-10-14): `src/inference_server/profiles.toml` holds `[general]` plus optional tables named after
  BlameGraph's regimes; `profiles.py` writes `[general]` and the table named by `ENGINE_PROFILE` into the env before
  `config.py` loads settings, so a profile overrides the launch env. BlameGraph serves a regime with its own table in
  its own launch and gates that config with its own equiv. Ships empty: one config for everything.
- `TIMELINE_DIR` turns on the engine event timeline (`timeline.py`): one JSONL event per scheduler
  decision and a profiler range per phase, both keyed by step id. Off by default, no-op when off.
