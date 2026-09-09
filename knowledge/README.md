# knowledge/ — the knowledge base

One JSON file per finding. This is the loop's memory and the source of truth for every design
decision; [`DECISIONS.md`](../DECISIONS.md) at the repo root is a **generated** view of it.

```bash
python -m inference_server.research.loop kb --status rejected     # the dead ends — read first
python -m inference_server.research.loop kb --tags prefill cache  # what is known about an area
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

Negative results are first-class here. A `rejected` entry is what stops the loop re-trying
something that has already been measured and found not to matter.
