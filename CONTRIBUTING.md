# Contributing

The engine is measured, not argued about. That is what most of this document is about.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
git config core.hooksPath scripts/hooks     # pre-push runs CI's lint + tests in ~10s
```

The hook needs the venv on `PATH` and says so rather than passing silently. `git push
--no-verify` or `SKIP_HOOKS=1 git push` when you mean it.

## How a change lands

```
branch  →  push  →  PR  →  ci-ok green  →  review  →  merge
```

Never push to `main`. CI runs an alarm step that fails loudly on a direct push, which is a
smoke detector, not a door.

## The three lanes

`ci-ok` is the only required check. It aggregates the lanes and goes green when a lane is
*skipped* — skipping is normal, since each lane only runs when its own paths change.

| lane | runs when you touch | what it is |
|---|---|---|
| `loop` | `research/`, `knowledge/`, `experiments/`, `tests/test_research_*` | the research package's own tests, plus a check that `DECISIONS.md` still matches `knowledge/`. Installs pytest and ruff and **nothing else**: if `research/` ever imports the engine, this lane goes red. That is the seam the whole project rests on. One sanctioned exception: `research/simulator.py` imports `inference_server.scheduling_policy`, stdlib-only pure functions the simulator must share with the scheduler or its conclusions drift. |
| `engine` | everything else | ruff, the full fast suite, and `premerge_check.py`. CPU only — the model-heavy tests are deselected and no kernel runs. |
| `gpu` | never automatically | the CUDA correctness gate on a rented RunPod GPU. Weekly, or dispatch it by hand. |

Anything the classifier does not recognise routes to `engine`: safe, not fast. Markdown,
`docs/` and `runs/` gate nothing.

### The GPU lane

Triton compiles only on CUDA, so **no lane above can execute a kernel**. That is not a
detail: `_paged_decode_kernel` once referenced a name that did not exist in its scope and was
dead for three commits, on the decode path used above 128 concurrent sequences. Dispatch the
GPU lane before merging anything under `src/inference_server/models/` — `ci-ok` prints a
warning when a PR touches it.

The gate is `scripts/gpu_tests/cuda_gate.py`: every check in `scripts/gpu_tests/checks.py`
(paged decode parity, sliding window, no-recompile, paged prefill parity) on one box, emitting
a `{"gate": ...}` verdict and exiting by it. The archived `*_modal.py` wrappers under
`scripts/archive/modal/gpu_tests/` call the same checks; nothing runs them any more.

**Venue** (Settings → Secrets → Actions):

| secret | what happens |
|---|---|
| `RUNPOD_API_KEY` | `run_on_runpod.py` rents one pod (default `NVIDIA GeForce RTX 4090` — the gate needs no 80 GB), runs `cuda_gate.py`, terminates it, and exits with the verdict. |
| unset | on the weekly tick the lane says so and exits 0; dispatched by hand it fails. |

There is no second venue. The Modal fallback was removed on 2026-09-16 with the credits long
gone; see `scripts/archive/modal/README.md`.

A stuck run is bounded by the job's `timeout-minutes`, and because a killed runner never
reaches `run_instrument`'s `finally`, a last step with `if: always()` runs
`scripts/tools/runpod_reap.py --name ci-cuda-gate --since <job start>`: it terminates pods
with that exact name started after the job began, and nothing else.

**Pod-name convention:** CI owns `ci-cuda-gate` (`run_on_runpod.py --name ci-cuda-gate`).
Humans keep the launcher's default, `inference-server-instrument`, so a pod you started by
hand during the weekly window is never the reaper's. A row the API returns malformed is
logged and skipped; a same-named pod whose start time is unreadable is terminated, because
"cannot tell how old" must not mean "keeps billing".

By hand, with a GPU type:

```bash
gh workflow run ci.yml --ref main -f runpod_gpu='NVIDIA GeForce RTX 4090'
```

## Evidence for an engine change

A diff under `src/inference_server/` — excluding `research/` and `static/` — needs a
committed record on the branch, or `premerge_check.py` refuses it. Prose in the PR body is
not one of the three paths. Each record vouches for **one commit**, with no engine change
after it.

| your change | the record | how |
|---|---|---|
| makes something faster | A/B experiment, five green gates | `loop judge` over ≥3 runs per arm. **Do not rebase after measuring** — the record names a sha, and rebasing invalidates it. |
| fixes a bug | `regression_test` + `engine_sha_base` | hand-write it; model on `experiments/exp-20260908-9497648e.json`. The gate **re-runs** the test, so confirm it fails with the fix reverted. |
| changes no behaviour | a no-claim record | `python -m inference_server.research.loop no-claim --why "..." --sha <sha>` — renames, dead imports, comments, type hints. |

Touched no engine file? The gate passes on its own; say so in the PR.

Read `LOOP.md` before running an experiment, and `loop kb --status rejected` before proposing
an optimisation — nine things in there are already disproved.

## Docs

CLAUDE.md's hard rule: a change to a component, endpoint, env var, metric or data-flow path
updates `docs/architecture.html` **and** the relevant `arch-*.html` in the same PR.
