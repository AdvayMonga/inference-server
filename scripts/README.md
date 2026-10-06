# scripts/

Engine tools. Measuring and judging the engine is the environment's job: [BlameGraph](https://github.com/AdvayMonga/BlameGraph)
(load regimes, correctness gate, profiler, GPU VM). GPU work runs on its VM: `python -m lab.vm run -- <command>`.

| folder | what |
|---|---|
| `bench/` | `tune_triton_launch.py`: sweeps the paged-attention kernels' launch configs and writes the table the engine loads (CUDA) |
| `gpu_tests/` | `checks.py`: paged decode and prefill parity, sliding window, no-recompile. Needs CUDA; run before merging a `models/` change |
| `hooks/` | `pre-push`: what CI runs, before the push |
| `tools/` | `smoke_custom.py`: the real server startup path with the custom backend, over HTTP |

The load generators, probes and policy benchmarks that lived here were superseded by BlameGraph's regimes and
removed on 2026-10-06 (tag `archive/pre-split`); the Modal-era instruments went on 2026-10-02 (`archive/research-loop`).
