# scripts/

Instruments that measure the engine, and small tools. The engine knows nothing about them.
GPU work runs on the lab's VM: `python -m lab.vm run --fetch lab/runs -- <command>`.

| folder | what |
|---|---|
| `bench/` | sweeps and A/Bs: `bench_serving.py`, `load_test.py`, `baseline_benchmark.py`, eviction, fairness and KV-pressure benchmarks, `roofline.py` (analytical ceiling, CPU), `tune_triton_launch.py` (kernel launch table, CUDA) |
| `probes/` | one-question diagnostics: `probe_compile.py`, `probe_prefill_ceiling.py` |
| `gpu_tests/` | `checks.py`: paged decode and prefill parity, sliding window, no-recompile. Needs CUDA; run before merging a `models/` change |
| `corpus/` | `fetch_traces.py` (pinned raw public traces) and `build_corpus.py` (the frozen, hashed corpus in `corpus/`); the loader is `lab/corpus.py` |
| `hooks/` | `pre-push`: what CI runs, before the push |
| `tools/` | `smoke_custom.py`, scheduler and batch-cache checks |

The Modal-era instruments and the RunPod launcher behind the 2026 A100/A10G evidence were
removed on 2026-10-02 with the research loop; tag `archive/research-loop` keeps them as provenance.
