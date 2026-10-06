## What changed


## Why


## Evidence
<!-- A change under src/inference_server/ is measured, not argued: link the profile bundle,
     the benchmark in BlameGraph `knowledge/evidence/`, or the ledger record, and say what config and
     machine it ran on. Touched no engine file? Say so. -->


## Checks
- [ ] `pytest -q` and `ruff check .` pass locally
- [ ] models/ change: `scripts/gpu_tests/checks.py` run on the environment's GPU VM (BlameGraph `python -m lab.vm`)
