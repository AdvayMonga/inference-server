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

## bench/

| script | measures |
|---|---|
| `bench_serving.py` | open-loop client for `/v1/completions`: Poisson arrivals, realistic lengths, SLO-gated |
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

## tools/

`smoke_custom.py`, `test_scheduler.py`, `test_batch_cache.py` are local end-to-end smoke runs.
`migrate_decisions_to_kb.py` and `backfill_experiments.py` are the one-shot migrations that
turned prose notes into `knowledge/` and `experiments/` records; kept for provenance.
