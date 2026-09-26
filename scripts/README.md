# scripts/ — instruments and tools

Everything that *measures* the engine lives here. The engine (`src/inference_server/`) knows
nothing about these; the research loop (`src/inference_server/research/`) knows only the panel
contract they emit. Instruments write `runs/<id>.json` via `research.harness` alongside their
human-readable tables.

GPU work goes to a rented pod through `tools/run_on_runpod.py` (`research/venues.py`), which
tries to pin the SM clock, reads the device either way, and exports the result as
`RESEARCH_DEVICE_STATE` so the instrument's panel records which machine it ran on and whether
its clocks were locked — a pod is a container, so the lock is always refused and recorded. The
`*_modal.py` instruments that produced the 2026 A100/A10G evidence were archived on 2026-09-16
under `archive/modal/` — kept as the provenance 13 `knowledge/` entries cite, not expected to
run. Everything else here runs locally.

| folder | what | LOOP.md tier |
|---|---|---|
| `bench/` | sweeps and A/Bs that produce a number worth recording | 4 (full sweep) |
| `probes/` | one-question diagnostics: where does the time go, is X even true | 3 (single-GPU probe) |
| `gpu_tests/` | parity and correctness checks that need CUDA (kernels, graphs, compile) | correctness gate |
| `tools/` | local smoke runs and one-shot migrations | — |
| `archive/modal/` | the Modal instruments, kept for provenance (see its README); nothing calls them | — |
| `hooks/` | git hooks: `pre-push` runs CI's lint+tests locally (`git config core.hooksPath scripts/hooks`) | — |
| `premerge_check.py` | the merge gate: refuses an engine change with no green experiment record | step 7 |
| `run_instrument.sh` | the archived Modal launcher; stamps provenance (`RESEARCH_ENGINE_SHA`, `RESEARCH_RUN_GROUP`). Only `archive/modal/` targets it | step 4 |
| `tools/run_on_runpod.py` | RunPod launcher: rent, sync, install, run, parse, terminate. Same provenance stamps | step 4 |
| `tools/runpod_reap.py` | terminates pods named `inference-server-instrument` (optionally only those started after `--since`) — CI's `if: always()` cleanup for a runner killed mid-run | — |

## bench/

| script | measures |
|---|---|
| `bench_serving.py` | open-loop client for `/v1/completions`: Poisson arrivals, realistic lengths, SLO-gated |
| `replay_trace.py` | open-loop replay of one `corpus/` trace (class, split) on the harness's own clock; posts through `/v1/chat/completions` so the model's chat template is applied (`--prompt-format raw` for the old `/v1/completions` path); panel stamped with `corpus_version`, `prompt_format` and a verified `chat_template` fingerprint, per-request CSV beside it |
| `replay_corpus_runpod.py` | the first GPU job: serves the engine on the pod with `TELEMETRY_DIR` on, replays `(class, split, rate_scale)` via `replay_trace` in-process, returns panels + per-request rows + telemetry rows in one payload. Speaks the venue contract; see the timing-model recipe under tools/ |
| `roofline.py` | analytical decode ceiling and per-step breakdown (attribution input) |
| `load_test.py` / `plot_load_test.py` | HTTP concurrency sweep against a running server, and its plots |
| `eviction_benchmark.py` / `fairness_benchmark.py` / `kv_pressure_benchmark.py` | CPU-runnable policy checks: LRU vs sink vs H2O, FCFS vs fair, backpressure |
| `baseline_benchmark.py` | the original single-request baseline (kept for the record) |
| `coldstart_load.py` | one cold `GemmaForCausalLM.from_hf` per PROCESS, as one panel (`wall_s` = load time, stage split in `harness_config.stage_ms`). Replicates are repeated processes; `RESEARCH_ARM` tags the arm. Tier 2, runs on any box |
| `serve_accounted.py` | the engine's uvicorn server plus a resource sidecar rewritten on every sample (so a SIGKILLed server still leaves one): the server's own peak RSS, peak device memory (exact on CUDA; sampled on MPS, which has no peak counter, by a thread that is GIL-starved under load — its coverage is recorded and a short one is named in the panel) and storage read bytes (Linux only). Not run directly; `replay_local.py` launches it so `research/` never has to import torch to know what a run cost |
| `replay_local.py` | serves the engine on THIS box and replays corpus traces against it, **one fresh server process per run**. Every panel carries a total-accounting block (wall clock from the server's launch, idle time, sessions served, the sidecar's resources) and the run prints the primary metric — GPU-seconds per session at the class's p95 TTFT ceiling. `--null N` is the noise-floor experiment (the same config as both arms, ABBA); `--configs FILE` is a sweep of engine configs, one run each. Fills the `ttft_queue_*` / `ttft_prefill_*` split from the telemetry rows and writes the `runs/<group>/` layout `fit_timing_from_runs.py` reads |
| `tune_triton_launch.py` | tier 3, CUDA only: sweeps the paged-attention kernels' launch configs (num_warps × num_stages × decode SPLITS × tiled-prefill BLOCK_M) per decode graph bucket and prefill bucket, checks each against the default launch at the parity tolerance, and writes the launch table the engine reads at load (`CUSTOM_BACKEND_LAUNCH_TABLE`). See *Tuning the Triton launch table* below |

The eleven A100/A10G sweeps (`bench_serving_modal.py`, `bench_stress_modal.py`,
`bench_load_sweep_modal.py`, `bench_vllm_sweep_modal.py`, `bench_chunked_prefill_modal.py`,
`bench_decode_batch_modal.py`, `bench_decode_buckets_modal.py`, `bench_prefill_graph_modal.py`,
`bench_prefill_compile_modal.py`, `bench_lm_head_slice_modal.py`, `bench_quant_modal.py`) moved
to `archive/modal/bench/`. `benchmarks/README.md` says which CSV each one produced.

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
Vitals panel) and is what the CI `gpu` lane runs on RunPod. The archived
`archive/modal/gpu_tests/test_paged_kernel_modal.py` and `..._prefill_kernel_modal.py` call the
same functions; `tests/test_cuda_gate.py` still asserts they do, so a copy cannot drift in. With
no CUDA the gate reports every check skipped and fails, so it can be dry-run locally:

```bash
PYTHONPATH=src python scripts/gpu_tests/cuda_gate.py                                    # any CUDA box
RUNPOD_API_KEY=... scripts/tools/run_on_runpod.py scripts/gpu_tests/cuda_gate.py --gpu 'NVIDIA GeForce RTX 4090'
```

`check_e4b_stop_tokens_modal.py` is a one-shot Modal instrument, not part of the gate: it
narrowed the E4B half of `kb-20260918-5906bc13` by replaying the June head-to-head's 8 raw
token-id prompts on `google/gemma-4-E4B-it` alone (plain `transformers`, greedy, unbatched, 100
steps, nothing stopping early) and recording where `1` / `106` / `50` first appear, so the two
engines' stop sets can be compared on the same rollout. `--out FILE` writes every generated id as
JSON; that artifact is the evidence (`knowledge/evidence/e4b-stop-tokens-20260919.json`) and the
entry's table is generated from it by `summarise()`, so the two cannot drift. Cheapest card that
fits (L4, 24 GB), weights on the `hf-cache` volume, torch/transformers pinned so a re-run is the
same experiment; `--dry-run` prints the plan without starting a GPU. Needs the `modal` extra. It
measures the *model* under HF decode, not our engine, which is why the entry keeps a trigger to
re-check under `custom-cuda` at the real batch widths.

## tools/

`smoke_custom.py`, `test_scheduler.py`, `test_batch_cache.py` are local end-to-end smoke runs.
`build_corpus.py` generated the frozen traces in `corpus/` once, from a fixed seed; rerun it only to
cut a new corpus version (see `corpus/README.md`).
`migrate_modal_citations.py` re-pointed the 13 `knowledge/` entries that cite a `*_modal.py`
instrument at `archive/modal/` when those scripts were archived (2026-09-16); it is idempotent.
`migrate_decisions_to_kb.py` and `backfill_experiments.py` are the one-shot migrations that
turned prose notes into `knowledge/` and `experiments/` records; `backfill_kb_regime.py` stamped
`regime` and `validity_range` onto the entries that predate those fields (2026-09-16), and its
mapping table is the record of why each got what it got. All kept for provenance.

### Tuning the Triton launch table

No `@triton.autotune` in the engine: it benchmarks on first call, which is cold-start time. The
engine instead reads a table at model load when `CUSTOM_BACKEND_LAUNCH_TABLE` points at one
(`models/launch_table.py`), keyed on (kernel, decode rows N or prefill tokens S, head_dim,
block_size, GPU name) for one model. Any launch with no entry is byte-for-byte today's launch, so
with the variable unset nothing changes. Blocked on RunPod funding; once funded:

```bash
RUNPOD_API_KEY=... MODEL_NAME=google/gemma-4-E4B-it MAX_BATCH_SIZE=256 \
    scripts/tools/run_on_runpod.py scripts/bench/tune_triton_launch.py --gpu 'NVIDIA A100 80GB PCIe'
#   -> knowledge/timing/triton-launch-<gpu>-<sha>.json   (the table; do not commit yet)
#   -> runs/<group>/triton_launch_sweep.json             (every config's ms, diff, exactness)
```

`MAX_BATCH_SIZE` / `CUSTOM_BACKEND_COMPILE` must match the serving config, since they set the
decode-bucket ladder being tuned. Entries are written only where a config beats the default
launch by 5% in isolation.

**Before the table is committed**, it is a behaviour change and needs an experiment
(kb-20260905-a474d802 says sweep first, then A/B; this is the sweep). Run the E2E A/B on one pod,
arm A without and arm B with `CUSTOM_BACKEND_LAUNCH_TABLE=knowledge/timing/<table>.json`
(it passes through `replay_corpus_runpod.py`), ≥3 runs per arm on `steady_interactive/seen`,
replicate on `heldout`, and `loop judge` it. Dispatch the CI `gpu` lane with the table set too.
An isolated kernel win is not an end-to-end win (the 10.84x tiled-prefill kernel lost E2E).

### The two local recipes

```bash
# the noise floor: a null experiment, then the band it implies (notes 03)
PYTHONPATH=src python scripts/bench/replay_local.py --class cold_start --null 6
PYTHONPATH=src python -m inference_server.research.loop band --run-group <that group>
#   -> knowledge/noise/<harness>-<class>-<model>-<hardware>.json, which
#      compare.significance_replicated then consults on every judged experiment

# the simulator's rank check against real hardware (notes 04)
PYTHONPATH=src python scripts/bench/replay_local.py --configs scripts/bench/configs_timing_fit.json
python scripts/tools/fit_timing_from_runs.py <fit group> --out knowledge/timing/<name>.json
PYTHONPATH=src python scripts/bench/replay_local.py --configs scripts/bench/configs_sim_validation.json
python scripts/tools/fit_timing_from_runs.py <validation group> --validate \
    --timing knowledge/timing/<name>.json
```

Two run groups, on purpose. The timing model is fitted from `configs_timing_fit.json`
(`cold_start/heldout` at three rate scales, plus one `long_context/heldout` run at
MAX_BATCH_SIZE=1 so a few long prompts prefill alone and give the slope leverage) and
rank-checked against `configs_sim_validation.json` (nine `cold_start/seen` configs spanning
MAX_BATCH_SIZE, rate scale, policy and the admission deadline). Held out by the corpus's own
seen/heldout split — fitting and validating on the same runs would report how well the model
reproduces its own training set.

Do not fit from a class this box cannot serve. `steady_interactive` and `long_context` at x1
shed 128 of 136 and 36 of 40 requests here, and the survivors' telemetry carries 40s "prefill"
spans (one request behind seven others in a K=8 wave) and 98s "decode steps". Those are invalid
inputs, not noisy ones. See `kb-20260917-c07eb94b`.

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
