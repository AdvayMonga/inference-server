# Research Loop — Specification

The loop is the scientific method with the looseness removed from everything except the
hypothesis. Every iteration measures the SAME panel and runs the SAME steps, so results are
comparable across iterations. Only the hypothesis under test varies.

---

## Why this is rigid

Every wrong turn in this project came from an iteration that differed from the last one:

| what happened | cost |
|---|---|
| Harness was closed-loop; measured the wrong bottleneck | ~1 month optimizing decode when prefill bound |
| `POOL_SIZE=64` made it a cache-HIT benchmark | hid every cache-miss bug, incl. a total pool leak |
| `MAX_BATCH_SIZE` 256 vs 32 across runs | 2.4x TPOT difference read as a regression |
| Compared to a number from another session | 957 vs 1151 tok/s for identical config |
| Parity gate compared across batch widths | false failure; hours chasing a non-bug |

**Rule: if two iterations do not measure the same panel under the same harness config, their
results may not be compared.** The loop enforces this rather than trusting discipline.

---

## Step 0 — VITALS (fixed panel, every iteration, no exceptions)

Emitted as one structured record. Never subsetted, never extended ad hoc — extending the panel
is a versioned change that invalidates comparison to prior iterations.

### Service level
| field | why it is in the panel |
|---|---|
| `tok_s_within_slo`, `slo_ttft_ms`, `slo_tpot_ms` | the project's definition of success |

> **Record every point; derive late.** One panel is one *rate*, but the headline metrics —
> max throughput inside the latency budget, and the saturation knee — are properties of a whole
> *sweep*. They are NOT computed by the instrument. `session.sweep_headline()` derives them at read
> time from the rate-points, which is why a new question can be asked of old panels. An instrument
> that reduces its own data to one summary number has already decided what mattered, in the place
> that is hardest to revisit. For ten iterations these two lived only in a printed table and were
> `None` in every machine record, so the loop judged secondary metrics exclusively.
| `ttft_p50`, `ttft_p95` | headline latency |
| `ttft_queue_p50/p95`, `ttft_prefill_p50/p95` | **the split that rules out whole classes of fix** — measured 21ms queue vs 203ms prefill, which killed every scheduling lever at once |
| `tpot_p50`, `tpot_p95` | decode health |
| `saturation_knee_rate` | where it breaks |

### Pressure and scheduling
`wave_sizes` histogram · `active_size` · `pending_depth` · `pending_high_water` ·
`kv_admit_blocked` · `total_rejected` · `total_expired` · `total_preempted` ·
`total_iteration_errors`

> `wave_sizes` alone killed two planned projects (length-grouped waves, varlen mixed-batch) by
> showing 83–91% of prefill waves are K=1. Without it both looked obviously correct.

### Cache
`hit_rate` · `lookups` · `entries` · `evictions` · `blocks_held` / `max_blocks` ·
`pool_free_blocks` / `pool_total_blocks` · `pool_utilization`

### Cost attribution
`prefill_ms` by bucket · `decode_ms_per_step` by bucket · `graph_capture_s` ·
`pct_of_memory_roof` · `ridge_batch` (from `roofline.py`)

### Resources
`peak_gpu_mem_gb` · `peak_host_rss_gb` · `wall_s` · `gpu_cost_usd`

### Validity block — the panel is INVALID without it
| field | catches |
|---|---|
| `engine_sha`, `dirty` | attributing a result to the wrong code |
| `harness`, `harness_config` (pool size, rates, duration, batch, KV blocks, compile, graph flags) | the `POOL_SIZE=64` and `MAX_BATCH_SIZE` classes of error |
| `n_samples`, `stderr`, `run_id`, `started_at` | judging an effect smaller than noise |
| `workload_regime` — `cache_hit_heavy` \| `cache_miss_heavy` \| `mixed` | quoting a prefill number without its regime |
| `concurrency_observed` | claiming a concurrency the run never reached |

---

## Step 1 — ATTRIBUTE

Rank where the time actually goes, against the analytical ceiling. Output: ordered gap list,
each with a magnitude and the evidence for it. No hypotheses yet.

## Step 2 — HYPOTHESIZE

Inputs: the gap list + the knowledge base (`DECISIONS.md`, 48 entries, mostly negative results).
Each candidate MUST carry:
- predicted direction **and magnitude**, written before measuring
- the **cheapest experiment that could falsify it**
- which gap it attacks
- a knowledge-base check: has this been tried and rejected?

> Roughly half of the hypotheses in this project were wrong, and the wrong ones were the
> expensive ones. Recording the prediction before the measurement is what makes that visible
> instead of being quietly rationalised afterwards.

## Step 3 — SCREEN (cheapest falsification first)

| tier | cost | example |
|---|---|---|
| 1. static / arithmetic | seconds | padding-waste calc; bucket-ladder simulation |
| 2. CPU repro | seconds | the KV block leak was proven on CPU in ~2s |
| 3. single-GPU probe | 5–10 min | `probe_*` scripts |
| 4. full sweep | 15–30 min, real $ | `bench_serving`, `bench_stress` |

Never enter tier N+1 while a tier-N test could still falsify the hypothesis.

## Step 4 — EXPERIMENT

- own branch + worktree, one variable changed
- **arms run simultaneously**, never sequentially, never against a stored number
- every panel tagged `arm=<name>` in its notes; all arms share one `RESEARCH_RUN_GROUP`
- **>=3 runs per arm, interleaved ABBA** — not one arm's runs then the other's
- baseline arm re-run in the same session even if a baseline "already exists"

## The interface is the library, not the CLI

The loop's consumer is an agent, not a person at a shell. Marshalling arguments into a subprocess
and parsing printed text back is friction, and a second surface is a second thing to keep in sync —
`loop judge` silently kept taking one panel per arm long after single-run arms stopped being able
to clear the significance gate. So the procedure lives in `research/session.py` and the CLI is a
shim over it.

```python
from inference_server.research.session import judge_group
j, exp = judge_group(hypothesis, "iter11-prefill", branch="perf/x", cost_usd=0.55)
```

`judge_group` loads the run group, splits the arms, and raises `NotMeasurable` before judging
anything if: an arm is untagged, there are not exactly 2 arms, an arm has <3 runs, the arms ran in
blocks rather than alternating, a panel came from a dirty tree, or **engine files changed since the
panels were measured**. That last one is the rebase rule below, enforced instead of remembered.

The one thing that stays a script is `scripts/premerge_check.py`: it runs at merge time, where
there is no agent in the loop. Instruments stay scripts too — they run in another process, on
another machine.

## Step 5 — JUDGE (all five gates, in order)

1. **Validity** — did the harness exercise the thing that changed? (`wave_sizes` for a wave
   change; `workload_regime` for a cache change.) Fail ⇒ result discarded, not "inconclusive".
2. **Sanity** — did a metric this change *cannot* affect move anyway? A prefill change cannot
   alter decode speed. Iteration 3 produced a CONFIRMED "-48.2%" from two different machines
   while TPOT p50 differed 95.3 vs 46.4ms between arms that only differed in a prefill flag.
   Declare `sanity_metrics` on every hypothesis; if one moves, the arms are contaminated and
   the headline number is not evidence, however significant it looks.
3. **Significance** — effect exceeds the variance budget on the metric predicted beforehand,
   measured across **replicate runs** (>=3 per arm after discarding each arm's first run as
   warmup). Never from a single run per arm: the panel's `stderr` is request-to-request scatter
   *within* a run, and this harness's *run-to-run* null spread is 3.2x on `ttft_prefill_p95`
   and 6.5x on `ttft_p95`. Using the former as the latter authorised a merge on noise once.
   Arms must also be order-alternated — three identical back-to-back runs measured
   1494 / 401 / 229 ms, so whichever arm runs second wins.
4. **Correctness** — fast suite green; parity gates **width-matched**, never across batch shapes;
   no new `total_iteration_errors`.
5. **Cost** — startup, memory and $ regressions declared, not just latency.

## Step 6 — RECORD (always, both outcomes)

A disproved hypothesis is a successful iteration and is written up with the same weight as a
win. Negative results are what stop the loop re-treading dead ends — capture ordering, mixed-batch
prefill, and length-grouped waves were each rejected on measurement and must never be retried
blind.

## Step 7 — MERGE, THEN RE-MEASURE

Merge only on five green gates. Then **re-run Step 0**: the bottleneck moves. It moved three
times in one day here — decode → prefill dispatch → prefill compute — and a plan written before
the move was stale immediately.

---

## Invariants

1. The panel is fixed. Changing it is versioned and breaks comparison to prior iterations.
2. No cross-session comparison. Ever.
3. Cheapest falsification first.
4. Negative results are first-class output.
5. Validity is judged before significance.
6. Re-measure after every merge.
7. The loop may conclude *"the thing you asked me to optimise does not matter"* — and that is a
   successful iteration. It happened twice here, each time killing a planned project.

## Workflow rule: never rebase after measuring

An experiment vouches for one commit sha. Rebasing the branch after the A/B run moves
every engine file to a new sha, and `premerge_check.py` correctly refuses the merge —
the measurement no longer describes the code being merged. Land any main-side changes
you need *first*, then measure the final commit. Cost of learning this: one extra A100
run (iter9 → iter10, ~$0.55).
