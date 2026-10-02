# lab

The environment that measures, and later grades, changes to the engine. Design in
`ENVIRONMENT.md` at the repo root. Imports nothing from `src/inference_server/research/`.

## Contract with the engine

- The engine is `src/inference_server/`. It is the only thing the agent may write.
- Black-box tools (bench, later) launch the server with `python -m inference_server.server`,
  wait on `GET /health`, and talk to it over its OpenAI-compatible HTTP API.
- White-box tools (profile) build the backend and scheduler in process, because torch.profiler
  has to live in the process it traces. Profiling never gates anything.
- Engine instrumentation the lab reads: `TIMELINE_DIR` turns on the event timeline
  (`src/inference_server/timeline.py`), `TELEMETRY_DIR` the per-request rows.

## Tools

    python -m lab.profile --backend custom-mps --requests 8 --max-tokens 32 --out lab/runs

The engine configuration comes from the same env vars the server reads (`MAX_BATCH_SIZE`,
`PREFILL_MODE`, ...), so a profile measures what is served; `--backend` and `--model` override
`BACKEND` and `MODEL_NAME`. Writes one bundle directory per run, raw files only:

    events.jsonl   engine event timeline: scheduler decisions and phases, per step
    trace.json     torch.profiler chrome trace; phase ranges carry the step id
    memory.json    device allocator stats and peak host RSS
    gpu.csv        nvidia-smi samples at 100 ms (CUDA hosts only)
    stats.json     the scheduler's own counters at the end of the run
    meta.json      git sha, torch, device, clock state, engine settings, workload hash, window
