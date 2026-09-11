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
| `loop` | `research/`, `knowledge/`, `experiments/`, `tests/test_research_*` | the research package's own tests, plus a check that `DECISIONS.md` still matches `knowledge/`. Installs pytest and ruff and **nothing else**: if `research/` ever imports the engine, this lane goes red. That is the seam the whole project rests on. |
| `engine` | everything else | ruff, the full fast suite, and `premerge_check.py`. CPU only — the model-heavy tests are deselected and no kernel runs. |
| `gpu` | never automatically | `scripts/gpu_tests/` on a rented CUDA box via Modal. Weekly, or dispatch it by hand. |

Anything the classifier does not recognise routes to `engine`: safe, not fast. Markdown,
`docs/` and `runs/` gate nothing.

### The GPU lane

Triton compiles only on CUDA, so **no lane above can execute a kernel**. That is not a
detail: `_paged_decode_kernel` once referenced a name that did not exist in its scope and was
dead for three commits, on the decode path used above 128 concurrent sequences. Dispatch the
GPU lane before merging anything under `src/inference_server/models/` — `ci-ok` prints a
warning when a PR touches it. Needs `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in repo secrets.

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
