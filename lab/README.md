# lab

The environment that measures, and later grades, changes to the engine. Design in
`ENVIRONMENT.md` at the repo root.

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

## Ledger

`lab/ledger.py`: one JSON line per tool call in `lab/ledger/ledger.jsonl`, the raw knowledge base.
Not committed for now. Tools write it through `ledger.append`; the agent reads it and never writes it
(the jail's write surface is `src/inference_server/` only).

    python -m lab.ledger seed                      # import knowledge/*.json as `finding` records
    python -m lab.ledger list --kind bench         # JSONL; also --session, --snapshot
    python -m lab.ledger show ev-20261002-7f3a91

A record: `kind` (test, equiv, bench, profile, submit, finding), `session`, `snapshot`,
`parent_snapshot`, `base`, `change`, `config` (everything the number is a fact about), `metrics`,
`gates`, `raw`, `cost`, and `claim`, which holds the agent's own hypothesis and note: untrusted, never
used for a verdict. The writer stamps `id`, `at` and `schema`. Heavy raw files (diffs, bundles) go
through `ledger.put_blob` and are referenced by their content-hashed path under `blobs/`.
A held-out record (`config.split == "heldout"`) carries no `raw` and one aggregate per metric
(`base`, `new`, `delta_pct`, `band_pct`, `verdict`); the writer refuses anything finer. A `"split":
"heldout"` anywhere else in a record (outside `claim`) is refused, so held-out data can't hide under
`result`; tools write held-out results with `Toolbox._record_heldout`.

## Runtime

    python -m lab.session --goal "cut decode step time on the steady_interactive class" --budget 20

One loop: while dollars remain, a fresh agent session gets the goal, the budget left, the last
ledger records and the tools, and is free. Its shell and file tools run inside the srt jail on an
exported copy of the engine (no git history, held-out data removed); it may write
`src/inference_server/` and add `tests/test_*.py`, nothing else. The lab's own tools run out here
with the referee's rights, snapshot the workspace and write the ledger on every call:

| tool | does |
|---|---|
| `test` | lint and the fast suite on a pristine two-commit copy of the workspace, jailed |
| `profile` | `lab.profile` on the pristine copy, jailed; the bundle goes into the ledger as a blob |
| `ledger` | read records (this run and earlier ones) |
| `budget` | dollars left |
| `restore` | workspace back to a snapshot id (`base` resets) |
| `note` | a note for the human; recorded, changes nothing |
| `equiv`, `bench`, `submit` | refuse with the reason until the eval harness is wired |

A session ends when the agent says `stop`, its per-session cap is spent, or it times out. The
run ends on budget, on `stop`, or on a write-surface violation (the one hard rule). Model cost
comes from the provider's own accounting; `LAB_MODEL` picks the model, `LAB_AGENT_PROVIDER` the
provider (only `claude` today, through the Agent SDK CLI; `lab/agent.py` is the seam for others).
The jail is sandbox-runtime (`npm install -g @anthropic-ai/sandbox-runtime`); `LAB_NO_JAIL=1`
waives it for tests and a box you trust.

## Canaries

    python -m lab.canary slow_decode,kv_leak --port 8000

Serves the engine with deliberate regressions applied from outside (`lab/canary.py`
monkeypatches the backend and scheduler in the server process; the engine tree never changes,
so the player cannot edit a canary away). The eval harness must flag every one; a canary nothing
catches is a blind spot. `slow_decode` (+3 ms per step), `slow_start` (+20 s after load),
`kv_leak` (reservations never freed), `admit_fewer` (one fewer row per admission pass, config
unchanged in stats).

## GPU arm

One persistent VM, started and stopped by hand, on the cloud `LAB_VM_PROVIDER` names:
`verda` (default; 1x H200, REST API), `nebius` (1x H200, `nebius` CLI) or `crusoe` (1x A100 PCIe, `crusoe` CLI). Same
commands either way:

    python -m lab.vm status
    python -m lab.vm types                  # what the account can rent right now
    python -m lab.vm start                  # create (first time) or start, wait for ssh
    python -m lab.vm setup                  # lab/vm-setup.sh: venv, CUDA torch, Nsight, counter and clock checks
    python -m lab.vm run --fetch lab/runs -- \
        env BACKEND=custom-cuda python -m lab.profile --requests 8
    python -m lab.vm stop                   # ends all billing (Verda: deletes the VM and its disk)

`run` rsyncs the working tree minus `.git` and everything `.gitignore` excludes (what is on disk
here is what gets measured, secrets and weights stay home), runs the command in the repo dir with
`.venv/bin` first on PATH, and brings `--fetch` (relative to the repo) back under `lab/runs/`. The
VM is left running; `stop` is yours. Do not run `uv sync` on the box: it would put CPU torch back.

Credentials never live in the repo. Verda: `VERDA_CLIENT_ID` and `VERDA_CLIENT_SECRET` (console >
Keys > Cloud API credentials) in your shell. Nebius: `nebius profile create`. Crusoe: `crusoe config init`.

Nebius: `stop` is a real stop (boot disk kept, GPU billing ends) and `start` resumes it. The VM is
created with a `lab` user from cloud-init (Nebius refuses root). Platform `gpu-h200-sxm`, preset
`1gpu-16vcpu-200gb` (`LAB_VM_PLATFORM`, `LAB_VM_TYPE`), image family `ubuntu24.04-cuda13.0`
(`LAB_VM_IMAGE`), the project's first subnet unless `LAB_VM_SUBNET` is set, `LAB_VM_PROJECT` for a
project other than the profile's.

Verda's `stop` deletes the instance and its OS volume: Verda's hibernate hides the instance from
its own API so it cannot be restored, and its shutdown keeps billing the GPU. Each session therefore
starts on a fresh VM; `setup` takes about a minute plus weight downloads. The image ships nvcc 12.8,
which vLLM's DeepGEMM FP8 path refuses (needs 12.9+; serve with `VLLM_USE_DEEP_GEMM=0`).

| env | verda default | crusoe default | |
|---|---|---|---|
| `LAB_VM_PROVIDER` | `verda` | | |
| `LAB_VM` | `lab-gpu` | `lab-gpu` | VM name (Verda: hostname) |
| `LAB_VM_TYPE` | `1H200.141S.44V` | `a100-80gb.1x` | `types` lists what is rentable |
| `LAB_VM_LOCATION` | `FIN-02` | `us-east1-a` | |
| `LAB_VM_IMAGE` | `ubuntu-24.04-cuda-12.8-open-docker` | `ubuntu22.04-nvidia-slurm:latest` | an image with the NVIDIA driver |
| `LAB_VM_DISK_GB` | `400` | fixed by type | OS volume size (Verda) |
| `LAB_VM_USER` | `root` | `ubuntu` | |
| `LAB_VM_KEYFILE` | `~/.ssh/lab_ed25519.pub` | same | must have no passphrase (ssh runs in BatchMode); Verda: uploaded on first use |
| `LAB_VM_DIR` | `~/inference-server` | same | where the tree lands |
