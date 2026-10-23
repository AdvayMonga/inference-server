# scripts/

Engine tools.

| script | what |
|---|---|
| `bench/tune_triton_launch.py` | (CUDA) sweeps the paged-attention kernels' launch configs and writes the table the engine loads |
| `gpu_tests/checks.py` | (CUDA) paged decode and prefill parity, no-recompile; run before merging a `models/` change |
| `tools/smoke_custom.py` | the real server startup path with the custom backend, over HTTP (MPS by default) |
