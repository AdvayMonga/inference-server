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
- repeat to `n` until `stderr` fits the variance budget
- baseline arm re-run in the same session even if a baseline "already exists"

## Step 5 — JUDGE (all four gates, in order)

1. **Validity** — did the harness exercise the thing that changed? (`wave_sizes` for a wave
   change; `workload_regime` for a cache change.) Fail ⇒ result discarded, not "inconclusive".
2. **Significance** — effect exceeds the variance budget on the panel's primary metric.
3. **Correctness** — fast suite green; parity gates **width-matched**, never across batch shapes;
   no new `total_iteration_errors`.
4. **Cost** — startup, memory and $ regressions declared, not just latency.

## Step 6 — RECORD (always, both outcomes)

A disproved hypothesis is a successful iteration and is written up with the same weight as a
win. Negative results are what stop the loop re-treading dead ends — capture ordering, mixed-batch
prefill, and length-grouped waves were each rejected on measurement and must never be retried
blind.

## Step 7 — MERGE, THEN RE-MEASURE

Merge only on four green gates. Then **re-run Step 0**: the bottleneck moves. It moved three
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
