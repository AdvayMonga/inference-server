# knowledge/ — the knowledge base

One JSON file per finding. This is the loop's memory and the source of truth for every design
decision; [`DECISIONS.md`](../DECISIONS.md) at the repo root is a **generated** view of it.

```bash
python -m inference_server.research.loop kb --status rejected     # the dead ends — read first
python -m inference_server.research.loop kb --tags prefill cache  # what is known about an area
python -m inference_server.research.loop kb --regime cold_start   # what applies to a workload class
python -m inference_server.research.loop kb --situation model=gemma-4-e4b,hardware=A100-80GB   # what covers my situation
python -m inference_server.research.loop index                    # regenerate DECISIONS.md
```

Edit the JSON, never the markdown. The record type is `KnowledgeEntry` in
[`research/schemas.py`](../src/inference_server/research/schemas.py).

| field | meaning |
|---|---|
| `status` | `open` (live), `deferred` (waiting on a trigger), `resolved` (shipped), `rejected` (measured and disproved), `obsolete` |
| `tags` | grep handles: `prefill`, `kv`, `kernel`, `graph`, `loop`, ... |
| `summary` | the finding, with the numbers that support it |
| `evidence` | `run_id` / `experiment_id` / `url` references — a claim without one is an opinion |
| `triggers` | what would reopen or close this entry |
| `superseded_by` | id of the entry that replaced this one |
| `supersedes` | id of the entry this one replaced — never delete an entry, supersede it |
| `regime` | workload class the finding applies to: `cold_start`, `steady_interactive`, `long_context` |
| `validity_range` | the bounds it was measured over; `covers()` in `kb.py` answers "does my situation fall inside" |
| `mechanism` | one sentence on why it worked or did not, so a near-miss hypothesis can reason about transfer |
| `transfer_checked` | re-run on a second model or hardware, and the result (`{"model": ..., "held": ...}`) |

`validity_range` is a plain dict. Conventional keys: `model`, `hardware`, `corpus_version`,
`engine_sha_range`, `context_tokens`, `concurrency`. A value is a scalar (equality), a list of
scalars (membership — `"hardware": ["A100", "H100"]` means measured on either), or a 2-list of
numbers (inclusive range — `"concurrency": [1, 16]`). Anything else fails `validate()`. Keys an
entry leaves out are unconstrained; an entry with no range at all is *unscoped* and `loop kb
--situation` tags it so. A situation outside the stated bounds is **not covered** — there is no
nearest-entry fallback, because extrapolating a finding past where it was measured is the
failure this field exists to stop.

`regime` is validated against `KnowledgeEntry.REGIMES`, which mirrors the workload classes in
[`corpus/manifest.json`](../corpus/manifest.json). Adding a workload class means adding it in
both places. Entries written before these fields existed were backfilled on 2026-09-16 by
`scripts/tools/backfill_kb_regime.py` (its mapping table is the record of why each entry got
what it got); a new entry sets `regime` at write time, or says why it has none.

## Subdirectories: machine-readable companions

`load_entries()` globs `knowledge/*.json` only, so these are data files the loop reads directly
rather than entries in the base. Each one has a `knowledge/` entry that tells its story.

| dir | what | written by | read by |
|---|---|---|---|
| `timing/` | the simulator's fitted `TimingModel` coefficients, keyed `<model>-<hardware>-<sha>` | `scripts/tools/fit_timing_from_runs.py` | `research/simulator.py` |
| `noise/` | the measured run-to-run spread with NOTHING changed, keyed `<harness>-<class>-<model>-<hardware>` | `loop band --run-group <null group>` | `research/compare.py` — a delta inside the band is `inconclusive`, never a win |

A band applies to exactly the situation it names: `find_band` matches harness, workload class,
model and hardware together and has no nearest-entry fallback, for the same reason `covers()`
has none.

Negative results are first-class here. A `rejected` entry is what stops the loop re-trying
something that has already been measured and found not to matter.
