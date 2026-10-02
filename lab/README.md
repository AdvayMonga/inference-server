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

One persistent VM, started and stopped by hand, on the cloud `LAB_VM_PROVIDER` names:
`verda` (default; 1x H100 SXM, REST API) or `crusoe` (1x A100 PCIe, `crusoe` CLI). Same
commands either way:

    python -m lab.vm status
    python -m lab.vm types                  # what the account can rent right now
    python -m lab.vm start                  # create (first time) or start, wait for ssh
    python -m lab.vm setup                  # lab/vm-setup.sh: venv, CUDA torch, Nsight, counter and clock checks
    python -m lab.vm run --fetch lab/runs -- \
        env BACKEND=custom-cuda python -m lab.profile --requests 8
    python -m lab.vm stop                   # ends GPU billing; the disk survives

`run` rsyncs the working tree minus `.git` and everything `.gitignore` excludes (what is on disk
here is what gets measured, secrets and weights stay home), runs the command in the repo dir with
`.venv/bin` first on PATH, and brings `--fetch` (relative to the repo) back under `lab/runs/`. The
VM is left running; `stop` is yours. Do not run `uv sync` on the box: it would put CPU torch back.

Credentials never live in the repo. Verda: `VERDA_CLIENT_ID` and `VERDA_CLIENT_SECRET` (console >
Keys > Cloud API credentials) in your shell. Crusoe: `crusoe config init`.

Verda's `stop` is its hibernate: GPU billing ends, the OS volume stays and is billed as storage;
Verda's own shutdown keeps billing the GPU and is never used. Verda prices are dynamic; `status`
shows the current rate.

| env | verda default | crusoe default | |
|---|---|---|---|
| `LAB_VM_PROVIDER` | `verda` | | |
| `LAB_VM` | `lab-gpu` | `lab-gpu` | VM name (Verda: hostname) |
| `LAB_VM_TYPE` | `1H100.80S.30V` | `a100-80gb.1x` | `types` lists what is rentable |
| `LAB_VM_LOCATION` | `FIN-01` | `us-east1-a` | |
| `LAB_VM_IMAGE` | `ubuntu-24.04-cuda-12.8-open-docker` | `ubuntu22.04-nvidia-slurm:latest` | an image with the NVIDIA driver |
| `LAB_VM_DISK_GB` | `200` | fixed by type | OS volume size (Verda) |
| `LAB_VM_USER` | `root` | `ubuntu` | |
| `LAB_VM_KEYFILE` | `~/.ssh/id_ed25519.pub` | same | Verda: uploaded on first use |
| `LAB_VM_DIR` | `~/inference-server` | same | where the tree lands |
