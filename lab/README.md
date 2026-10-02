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

Writes one bundle directory per run (see `lab/bundle.py` for the files). Raw files only.
