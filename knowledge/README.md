# knowledge/ — the knowledge base

One JSON file per finding: the project's measured memory and the source of truth for every design
decision. The lab's ledger (`lab/ledger.py`) imports them as `finding` records.

```bash
python -m lab.ledger seed                       # import these findings into the lab's ledger
python -m lab.ledger list --kind finding        # read them back, one JSON line each
```

Edit the JSON, never the markdown. The record type is `KnowledgeEntry` in
`lab/ledger.py`.

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
| `validity_range` | the bounds it was measured over; a situation outside them is not covered |
| `mechanism` | one sentence on why it worked or did not, so a near-miss hypothesis can reason about transfer |
| `transfer_checked` | re-run on a second model or hardware, and the result (`{"model": ..., "held": ...}`) |
| `suspended_metrics` / `suspended_tiers` | panel fields this entry says currently **cannot be measured**, and at which falsification tiers |

`validity_range` is a plain dict. Conventional keys: `model`, `hardware`, `corpus_version`,
`engine_sha_range`, `context_tokens`, `concurrency`. A value is a scalar (equality), a list of
scalars (membership — `"hardware": ["A100", "H100"]` means measured on either), or a 2-list of
numbers (inclusive range — `"concurrency": [1, 16]`). Anything else fails `validate()`. Keys an
entry leaves out are unconstrained; an entry with no range at all is *unscoped*. A situation outside the stated bounds is **not covered** — there is no
nearest-entry fallback, because extrapolating a finding past where it was measured is the
failure this field exists to stop.

`regime` is validated against `KnowledgeEntry.REGIMES`, which mirrors the workload classes in
[`corpus/manifest.json`](../corpus/manifest.json). Adding a workload class means adding it in
both places. Entries written before these fields existed were backfilled on 2026-09-16 (the mapping table
is in tag `archive/research-loop`, `scripts/tools/backfill_kb_regime.py`); a new entry sets
`regime` at write time, or says why it has none.

## Suspensions — what cannot currently be measured

`status` says what we believe; `suspended_metrics` says what we are **not able to find out**, so
the two are orthogonal. `kb-20260919-94acfdb8` is `open` — live, waiting on a trigger — and at the
same time suspends `ttft_p95` at tier 1, because the simulator ranks that metric at rho 0.644
against a 0.683 critical value and therefore cannot falsify a p95-TTFT hypothesis without a GPU.

A suspension is **scoped to a metric and a tier**, never to a whole entry — that would block
everything its subject touches, get worked around, and a gate people route around is worse than
no gate.

Metric and tier are also *all* the gate enforces. An entry's `regime` and `validity_range` are
printed next to a block so a reader can see where the finding came from, but `suspends()` does
**not** check them: a `Hypothesis` carries no regime to check against. That is the right
behaviour for every suspension we hold today, because each describes the **instrument** — the
simulator has no termination model on any hardware — so blocking that metric-tier pair
everywhere is exactly right. A suspension that is genuinely hardware-specific ("this kernel
misbehaves only on Blackwell") would instead over-block every machine, and wants a
`Hypothesis.regime` field before it can be scoped honestly. Not built: no such suspension exists.

`loop screen` checks live suspensions against **every** entry, not only the ones `related()`
surfaces: a suspension is a property of the instrument, not of the subject, so a prefix-cache
hypothesis predicting `ttft_p95` at tier 1 is exactly as unfalsifiable as a scheduling one. It
exits 1 and names the entry and its scope.

**Lifting one:** fix the instrument, file the record that shows it measuring again, and point
`supersedes` / `superseded_by` at each other. `suspensions()` returns only entries that are
neither superseded nor `obsolete`, so the gate opens the moment the successor lands. Never edit
the fields away — that erases the record that the gate was ever there. This has already happened
by hand once: `kb-20260917-c07eb94b` suspended `ttft_p95`, `kb-20260918-9fc68282` lifted it, and
`kb-20260919-94acfdb8` re-imposed it.

## Subdirectories: machine-readable companions

`load_entries()` globs `knowledge/*.json` only, so these are data files the loop reads directly
rather than entries in the base. Each one has a `knowledge/` entry that tells its story.

| dir | what | written by | read by |
|---|---|---|---|
| `timing/` | the simulator's fitted `TimingModel` coefficients, keyed `<model>-<hardware>-<sha>` | retired with the loop (2026-10-02); data kept | nothing now |
| `noise/` | the measured run-to-run spread with NOTHING changed, keyed `<harness>-<class>-<model>-<hardware>` | retired with the loop (2026-10-02); data kept | the lab's eval harness, once wired: a delta inside the band is `inconclusive`, never a win |

A band applies to exactly the situation it names: `find_band` matches harness, workload class,
model and hardware together and has no nearest-entry fallback, for the same reason `covers()`
has none.

Negative results are first-class here. A `rejected` entry is what stops the loop re-trying
something that has already been measured and found not to matter.
