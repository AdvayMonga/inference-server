# Contributing

The engine is measured, not argued about. The environment ([BlameGraph](https://github.com/AdvayMonga/BlameGraph)) is where that happens.

## Setup

```bash
uv sync --extra dev && source .venv/bin/activate    # exact versions from uv.lock
git config core.hooksPath scripts/hooks     # pre-push runs CI's lint + tests in ~10s
```

`uv.lock` pins every dependency; Linux takes the CPU torch wheel, macOS the PyPI one (MPS).
The GPU VM (BlameGraph's `python -m lab.vm setup`) installs the CUDA wheel at the same locked version.

The hook needs the venv on `PATH` and says so rather than passing silently. `git push
--no-verify` or `SKIP_HOOKS=1 git push` when you mean it.

## How a change lands

```
branch  →  push  →  PR  →  ci-ok green  →  review  →  squash merge
```

Never push to `main`. Every PR gets a review pass before merge; the ones that came from an
agent get it twice.

## CI

`ci-ok` is the only required check. One lane feeds it: `engine`, ruff and the full fast suite (engine and
control plane). CPU only; model-heavy tests are deselected and no kernel runs.

Triton compiles only on CUDA, so no lane can execute a kernel. Before merging anything under
`src/inference_server/models/`, run the checks in `scripts/gpu_tests/checks.py` on the environment's
GPU VM, from BlameGraph: `python -m lab.vm run -- python scripts/gpu_tests/checks.py`.

## Evidence for an engine change

Today: a PR with the measurement behind it (a profile bundle, a benchmark CSV under
BlameGraph's `knowledge/evidence/`, or a ledger record) and a reviewer who reads it. The lab's `submit` tool
will make this mechanical once the eval harness is wired; until then the number in the PR body
is the claim and the reviewer is the gate.

Two rules that outlive any tooling. A number is a fact about a config and a machine: say both.
And a delta inside the run-to-run noise of the same setup is not a result, however many
replicates it has.
