# scripts/ — instruments and tools

Everything that *measures* the engine lives here. The engine (`src/inference_server/`) knows
nothing about these; the research loop (`src/inference_server/research/`) knows only the panel
contract they emit. Instruments write `runs/<id>.json` via `research.harness` alongside their
human-readable tables.

`*_modal.py` scripts run on a rented GPU through [Modal](https://modal.com); launch them with
`run_instrument.sh` so the panel carries the right engine SHA and run group. Everything else
runs locally.

| folder | what | LOOP.md tier |
|---|---|---|
| `bench/` | sweeps and A/Bs that produce a number worth recording | 4 (full sweep) |
| `probes/` | one-question diagnostics: where does the time go, is X even true | 3 (single-GPU probe) |
| `gpu_tests/` | parity and correctness checks that need CUDA (kernels, graphs, compile) | correctness gate |
| `tools/` | local smoke runs and one-shot migrations | — |
| `hooks/` | git hooks: `pre-push` runs CI's lint+tests locally (`git config core.hooksPath scripts/hooks`) | — |
| `premerge_check.py` | the merge gate: refuses an engine change with no green experiment record | step 7 |
| `run_instrument.sh` | Modal launcher that stamps provenance (`RESEARCH_ENGINE_SHA`, `RESEARCH_RUN_GROUP`) | step 4 |
| `tools/run_on_runpod.py` | RunPod launcher: rent, sync, install, run, parse, terminate. Same provenance stamps | step 4 |
| `tools/runpod_reap.py` | terminates pods named `inference-server-instrument` (optionally only those started after `--since`) — CI's `if: always()` cleanup for a runner killed mid-run | — |

## bench/

| script | measures |
|---|---|
| `bench_serving.py` | open-loop client for `/v1/completions`: Poisson arrivals, realistic lengths, SLO-gated |
| `replay_trace.py` | open-loop replay of one `corpus/` trace (class, split) on the harness's own clock; panel stamped with `corpus_version`, per-request CSV beside it |
| `replay_corpus_runpod.py` | the first GPU job: serves the engine on the pod with `TELEMETRY_DIR` on, replays `(class, split, rate_scale)` via `replay_trace` in-process, returns panels + per-request rows + telemetry rows in one payload. Speaks the venue contract; see the timing-model recipe under tools/ |
| `bench_serving_modal.py` | the primary SLO curve — `bench_serving.py` co-located with the engine on an A100 |
| `bench_stress_modal.py` | overload and KV-pressure stress: does the engine degrade or break |
| `bench_load_sweep_modal.py` | closed-loop concurrency sweep of our engine (+ an HF baseline arm) |
| `bench_vllm_sweep_modal.py` | the vLLM arm of the head-to-head, same sweep |
| `bench_chunked_prefill_modal.py` | chunked vs monolithic prefill: head-of-line stall on decoders |
| `bench_decode_batch_modal.py` | full-batch decode: bandwidth-bound check + max-batch graph |
| `bench_decode_buckets_modal.py` | bucketed decode CUDA graphs vs one max-batch graph |
| `bench_prefill_graph_modal.py` / `bench_prefill_compile_modal.py` | K=1 prefill: graphed / compiled vs eager |
| `bench_lm_head_slice_modal.py` | lm_head over every prefill position vs the last one only |
| `bench_quant_modal.py` | int8 weight-only vs bf16 decode throughput |
| `roofline.py` | analytical decode ceiling and per-step breakdown (attribution input) |
| `load_test.py` / `plot_load_test.py` | HTTP concurrency sweep against a running server, and its plots |
| `eviction_benchmark.py` / `fairness_benchmark.py` / `kv_pressure_benchmark.py` | CPU-runnable policy checks: LRU vs sink vs H2O, FCFS vs fair, backpressure |
| `baseline_benchmark.py` | the original single-request baseline (kept for the record) |

## probes/

`probe_*` ask one question before anything is built (is prefill near its ceiling? does a mixed
step pay?). `profile_*` break a step down by kernel. `diag_*` chase a specific numerical
divergence. `repro_*` and `sweep_*` are what they say.

## gpu_tests/

Parity tests that only mean something on CUDA: paged decode/prefill kernels vs SDPA, split-K,
tiled prefill, CUDA-graph and `torch.compile` output parity, E4B load, vLLM compatibility. The
CPU-runnable parity tests live in `tests/`.

`checks.py` holds the paged decode/prefill checks as plain functions; `cuda_gate.py` runs all
of them on one box as a venue instrument (`{"gate": ...}` payload, exit status = verdict, no
Vitals panel) and is what the CI `gpu` lane runs on RunPod. `test_paged_kernel_modal.py` and
`test_paged_prefill_kernel_modal.py` call the same functions through Modal. With no CUDA the
gate reports every check skipped and fails, so it can be dry-run locally:

```bash
PYTHONPATH=src python scripts/gpu_tests/cuda_gate.py                                    # any CUDA box
RUNPOD_API_KEY=... scripts/tools/run_on_runpod.py scripts/gpu_tests/cuda_gate.py --gpu 'NVIDIA GeForce RTX 4090'
```

## tools/

`smoke_custom.py`, `test_scheduler.py`, `test_batch_cache.py` are local end-to-end smoke runs.
`build_corpus.py` generated the frozen traces in `corpus/` once, from a fixed seed; rerun it only to
cut a new corpus version (see `corpus/README.md`).
`migrate_decisions_to_kb.py` and `backfill_experiments.py` are the one-shot migrations that
turned prose notes into `knowledge/` and `experiments/` records; `backfill_kb_regime.py` stamped
`regime` and `validity_range` onto the entries that predate those fields (2026-09-16), and its
mapping table is the record of why each got what it got. All kept for provenance.

`fit_timing_from_runs.py` turns a `replay_corpus_runpod.py` run group into the simulator's
`TimingModel` (fit from the telemetry rows; mapping in its docstring) and, with `--validate`,
reports the Spearman rank correlation of `ttft_p95` / `tpot_p50` between the hardware panels and
the simulator over the same `(class, split, rate_scale)` configs — the check that makes tier 1
trustworthy (notes 04-simulator.md). The whole recipe, in order:

```bash
# 1. rent + replay (~15-20 min on an A100 80GB PCIe incl. provisioning and the model download)
RUNPOD_API_KEY=... scripts/tools/run_on_runpod.py scripts/bench/replay_corpus_runpod.py
#    -> runs/<run_id>.json per replay, runs/<run_group>/<class>-<split>-x<scale>.{rows,telemetry}.csv
#    REPLAY_CLASSES / REPLAY_SPLITS / REPLAY_RATE_SCALES (default steady_interactive / seen / 1,2,4)
#    `--timeout` is the instrument's wall budget on the pod (default 3600s); a run killed by it
#    prints no closing payload marker, so the rental yields NOTHING. Raise it before turning
#    CUSTOM_BACKEND_COMPILE on: the decode ladder at MAX_BATCH_SIZE=256 costs ~21 min on a cold
#    box (kb-20260902-008), which is why this instrument defaults it to 0.
# 2. fit, locally
python scripts/tools/fit_timing_from_runs.py <run_group>
#    -> knowledge/timing/<model>-<hardware>-<sha>.json
# 3. validate: same configs through `research.simulator`, rank correlation vs hardware
python scripts/tools/fit_timing_from_runs.py <run_group> --validate
```

`venue_smoke.py` is the first instrument that speaks the venue contract. It checks that the tree
synced, that provisioning installed what the engine imports, that a GPU is really there and can
run a kernel, and that provenance survived the hop. It emits **no** Vitals panel on purpose: it
measures the transport, not the engine, and a panel that looks like a measurement without being
one is what this repo keeps getting burned by. Run it before any instrument that costs real time:

```bash
RUNPOD_API_KEY=... scripts/tools/run_on_runpod.py scripts/tools/venue_smoke.py \
    --gpu 'NVIDIA GeForce RTX 4090'
```
