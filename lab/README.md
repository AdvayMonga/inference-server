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

## GPU arm

One persistent Crusoe VM, started and stopped by hand, driven through the `crusoe` CLI. A
stopped on-demand VM keeps its disk and bills nothing. Auth is the CLI's own
(`crusoe config init`, or `CRUSOE_ACCESS_KEY_ID` / `CRUSOE_SECRET_KEY` / `CRUSOE_DEFAULT_PROJECT`);
nothing here stores a key.

    python -m lab.crusoe status
    python -m lab.crusoe start                 # create (first time) or start, wait for ssh
    python -m lab.crusoe setup                 # lab/vm-setup.sh: venv, CUDA torch, Nsight, counter and clock checks
    python -m lab.crusoe run --fetch lab/runs -- \
        env BACKEND=custom-cuda python -m lab.profile --requests 8
    python -m lab.crusoe stop

`run` rsyncs the working tree minus `.git` and everything `.gitignore` excludes (what is on disk
here is what gets measured, secrets and weights stay home), runs the command in the repo dir with
`.venv/bin` first on PATH, and brings `--fetch` (relative to the repo) back under `lab/runs/`. The
VM is left running; `stop` is yours. Do not run `uv sync` on the box: it would put CPU torch back.

| env | default | |
|---|---|---|
| `LAB_VM` | `lab-gpu` | VM name |
| `LAB_VM_TYPE` | `a100-80gb.1x` | `python -m lab.crusoe types` lists what the account can rent; single-GPU H100 is not listed by Crusoe at the time of writing |
| `LAB_VM_LOCATION` | `us-east1-a` | |
| `LAB_VM_IMAGE` | `ubuntu22.04-nvidia-slurm:latest` | an image with the NVIDIA driver; `crusoe compute images list` |
| `LAB_VM_USER` | `ubuntu` | |
| `LAB_VM_KEYFILE` | `~/.ssh/id_ed25519.pub` | public key given to the VM on create |
| `LAB_VM_DIR` | `~/inference-server` | where the tree lands |
