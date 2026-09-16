# Archived Modal instruments

These 36 files are the instruments that produced this project's A100 and A10G evidence between
May and September 2026, plus `modal_app.py`, the one-container-per-GPU deployment they were
written against. They ran on [Modal](https://modal.com) because free credits made it the cheapest
way to reach a GPU. The credits ran out on 2026-09-07 and Modal is no longer the platform for
this project: it runs gVisor, so CRIU / `cuda-checkpoint` is structurally out of a tenant's
reach, and its GPU memory snapshots are alpha, documented as incompatible with multi-GPU, and
documented as not helping when weight loading dominates. The reasoning is recorded in
`knowledge/` under the `modal` + `venue` tags: `kb-20260916-57d2bb4a` (why we left) and
`kb-20260916-d12c9170` (the privilege ladder behind it).

**They are kept, not deleted, because 13 `knowledge/*.json` entries cite them by path as the
provenance for a measured number.** Deleting them would orphan the audit trail that
`DECISIONS.md` is generated from. `benchmarks/README.md` cites several of them the same way.

**They are not expected to run.** There are no Modal credits, `modal` is no longer an extra in
`pyproject.toml`, and nothing in CI or in the research loop calls them. Live GPU work goes
through `src/inference_server/research/venues.py` (RunPod) — see `LOOP.md` and `CONTRIBUTING.md`.

Layout mirrors where each file used to live: `bench/`, `probes/`, `gpu_tests/` under
`scripts/`, and `modal_app.py` at the repo root.

## Running one anyway

If a historical number ever has to be reproduced, from the repo root:

```bash
pip install 'modal>=1.0' && modal setup        # the extra was removed; install it by hand
scripts/run_instrument.sh scripts/archive/modal/bench/bench_serving_modal.py
```

`run_instrument.sh` stamps the provenance (`RESEARCH_ENGINE_SHA`, `RESEARCH_RUN_GROUP`) that
makes a panel attributable; it is kept at `scripts/` root and has no other caller.
`gpu_tests/test_paged_kernel_modal.py` and `gpu_tests/test_paged_prefill_kernel_modal.py`
additionally need `PYTHONPATH=scripts/gpu_tests`, because `checks.py` deliberately stayed
behind with the live `cuda_gate.py` rather than being copied here.
