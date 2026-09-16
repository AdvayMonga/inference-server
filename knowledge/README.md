# knowledge/ — the knowledge base

One JSON file per finding. This is the loop's memory and the source of truth for every design
decision; [`DECISIONS.md`](../DECISIONS.md) at the repo root is a **generated** view of it.

```bash
python -m inference_server.research.loop kb --status rejected     # the dead ends — read first
python -m inference_server.research.loop kb --tags prefill cache  # what is known about an area
python -m inference_server.research.loop kb --regime cold_start   # what applies to a workload class
python -m inference_server.research.loop kb --situation model=E4B,concurrency=8   # what covers my situation
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

`validity_range` is a plain dict. Conventional keys: `model`, `hardware`, `corpus_version`
(scalars, matched by equality) and `context_tokens`, `concurrency`, `engine_sha_range`
(2-lists, `[lo, hi]` inclusive). Keys an entry leaves out are unconstrained. A situation outside
the stated bounds is **not covered** — there is no nearest-entry fallback, because extrapolating
a finding past where it was measured is the failure this field exists to stop.

Negative results are first-class here. A `rejected` entry is what stops the loop re-trying
something that has already been measured and found not to matter.
