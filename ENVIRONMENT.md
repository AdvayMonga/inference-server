# The environment

Design for the next version of the research side of this repo: an open environment in which an
agent improves the inference engine, replacing the fixed loop in `research/method/`. Written
2026-10-02 from a design discussion; it describes intent, not what exists yet. Measured findings
in `knowledge/` still stand.

## Why

The current loop is a graph of nodes: hypothesize, build, check, review, measure. That order is
our idea of how to find a performance win, written into code. Our own rule says strategy is not
harness: if the model were 10x smarter, would the rule still be needed? For the graph the answer
is no. So the graph goes, and only two things stay in code:

- the referee: what counts as a win, and what the agent may not do;
- the instruments: measurements the agent can trust.

Everything about *how* to find a win leaves the code. The agent gets a goal, a budget, tools and
memory, and is free. The aim is a setup whose results improve as the model improves, and nothing
else is the priority right now.

The analogy is AlphaEvolve, not AlphaGo. AlphaGo could try millions of moves because each
evaluation was free. Here one evaluation is a test run plus a benchmark and costs minutes or
dollars. Freeing the agent does not buy scale; a cheap, trustworthy evaluator does. That is why
the eval harness is built first.

## What stays a loop

One loop remains, and it is not a strategy loop:

    while budget remains: start a session with the goal, the ledger, the tools

It exists only because a model's context fills up. The ledger carries state between sessions.

## Components

### 1. Eval harness (first)

Ground truth. Levels, cheapest first; only the top level defines a win, every lower level is a
filter the agent may run as often as it likes:

| level | catches | cost | who runs it |
|---|---|---|---|
| correctness: test suite plus output equivalence | broken semantics, drifted logits | seconds | agent, any time |
| component micro: kernel op timing, scheduler on a simulated clock, KV cache ops | local wins and regressions | seconds | agent, any time |
| short end-to-end on MPS, fixed-length templated corpus | whether a local win reaches the user-visible metric | minutes | agent, metered |
| full end-to-end on a rented GPU, natural-length corpus, seen plus held-out | the win | dollars | agent on seen; held-out only on submit |

Pieces this needs:

- **Templated corpus.** Today's prompts are raw text to an instruct model, so it stops almost
  immediately and the benchmark measures prefill only. Wrap prompts in the chat template so decode,
  scheduling and KV cache become visible. Two frozen variants: fixed output lengths for the short
  tier, natural lengths for the full tier.
- **Short tier.** A one-to-two-minute version of the end-to-end benchmark on MPS. A screen, not truth.
- **Correlation measurement.** Prove the short tier ranks changes the way the full tier does, using
  the merged perf PRs as a set of changes with known outcomes (Spearman rank). A screen that cannot
  rank is noise and the agent would optimize for noise. Re-measure when corpus, model or hardware
  changes. The simulator failed this check (rho 0.644) and is parked.
- **Noise bands** per level from repeated runs of unchanged code. A delta inside the band is not a
  result at any level.
- **Equivalence levels.** Exact token match for greedy decoding, logits within tolerance for kernel
  changes, distribution-level checks for anything touching sampling. The harness picks the level
  from what the change touched; the agent does not.
- **Kernels on MPS** are checked for correctness and gross regressions only. Timing counts on the
  target GPU.

Coverage comes from the top tier being sensitive to every part of the stack, not from having many
evals. The templated corpus is what buys that.

### 2. Measurement

Two separate things:

- **Referee numbers**: end-to-end metrics, noise bands, total accounting. Few, trusted. Exists.
- **The flame graph**: a tool for the agent to find wins. Never gates. Off the shelf: torch.profiler
  chrome traces, Nsight Systems and Compute for hardware counters, nvidia-smi for clocks and power
  per step. Custom and small: an engine event timeline (admit, prefill, decode step, evict, with
  timestamps) joinable to the profiler trace, so a kernel timing sits inside the scheduler decision
  that caused it. On MPS most of this is unavailable; the real sensor is the rented GPU.

Rule: give the agent the raw trace, not a digest. A pre-chewed summary (today's `attribute.py`)
is us deciding what matters. Keep digests as a convenience on request, never as the only view.

### 3. Knowledge base (last)

Most of it is free: the ledger the environment writes anyway (diff, score, trace, test result per
attempt) is the raw knowledge base. Distillation over it is the part that depends on everything
else and is built last. Now: record everything raw and lossless so distillation has data. Models
fail cold at "improve inference"; by attempt two hundred they are not cold, and that is what the
ledger provides. The hand-written findings in `knowledge/` become seed records.

### 4. CI

Small. Runs the harness's own tests, the `needs_host` tests the jail cannot run, and re-measures a
submitted win so a PR carries a reproducible number rather than the agent's claim.

### 5. Safety and leakage

Actions: jail, write surfaces (deny by default), grader. Exists, untouched.

Data, three holes to keep closed:

- the held-out split is unreadable from the jail;
- held-out results come back as one aggregate number, never per request;
- measurement code is unwritable.

Secrecy of the grader's logic is not needed. Goodhart risk lives in the metrics, not in whether
the agent can read `gates.py`.

### 6. Environment runtime

The thing that gives the agent its tools: the session loop over budget, ledger writes on every
tool call, snapshot and restore, submit, spawn. Small, but it is the product.

### 7. Compute plumbing

A GPU venue (exists: `venues.py`) and an eval queue in front of it, so parallel sessions share one
GPU. Without the queue, spawn is either expensive or stalled.

### 8. Scoreboard for the environment

Wins per dollar, per model version, on the same budget. The only way to know whether the
environment scales with intelligence. First row: this environment against the current graph on
equal budget. With it, the baseline advancement path: a win becomes a PR, a human merges, the next
session measures against the new base.

## Tools

The agent has bash, read, edit and write inside the jail; the write surfaces say what it may
touch. The tools are metered commands, and every one writes its result to the ledger whether the
agent likes it or not.

| tool | does |
|---|---|
| test | pristine test run, the grader's own |
| equiv | output equivalence against base, at the level the harness chose |
| bench --tier micro, short, full | measured, noise-banded; full rents a GPU; seen split only |
| profile | the raw trace, and a digest on request |
| review | code review of its own change, optional |
| ledger | past attempts, diffs, scores; `restore <attempt>` puts the workspace back |
| knowledge | measured findings |
| budget | what is left |
| submit | audit, pristine tests, held-out bench, record; the only thing that produces a win |
| spawn | a sub-session with the same jail and a slice of the budget; the model decides how to fan out |

Snapshots happen on every bench and submit, so git stays out of the jail.

## Where it lives

A new top-level package, `lab/`, written from scratch. It imports nothing from `research/`:
ideas and data carry over, modules do not. The lab treats the engine as a black box behind its
HTTP API (a launch command, a port, a health check) and its write surface for the agent is
`src/inference_server/` alone.

What carries over as ideas and data: the measured findings in `knowledge/` as seed ledger records;
clock state travels with every number (determinism) and cost is counted from process start
(accounting); the corpus, regenerated as templated traces.

`research/` stays untouched as the baseline for the head to head. Once the scoreboard has that
first row, it is deleted in one PR.

## Order

1. Eval harness and measurement: ground truth and sensor, useful to any design.
2. Environment runtime with the ledger, and the tool CLI (test, equiv, bench short, ledger, submit).
3. CI.
4. Bench full, eval queue, spawn.
5. Scoreboard, and the head to head against the current graph.
6. Knowledge distillation.

Safety is not a phase; it is in every PR.

## Cautions

- Eval cost is the bottleneck for both designs. Thousands of attempts are not available until the
  short tier is validated and cheap.
- The environment gives a weak model nothing to lean on. That is the point, and it is why the head
  to head is the first experiment, not an afterthought.
- Measurement on MPS is thin; a number from MPS is a fact about MPS.
