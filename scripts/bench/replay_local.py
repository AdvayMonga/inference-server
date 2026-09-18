#!/usr/bin/env python
"""Serve the engine on THIS box and replay corpus traces against it. One fresh server per run.

The local counterpart of `replay_corpus_runpod.py`, for the two jobs that need real hardware but
not a rented one:

    # the noise floor (notes/03): the SAME config as both arms, interleaved ABBA
    PYTHONPATH=src python scripts/bench/replay_local.py --class cold_start --null 6

    # a config sweep: one run per config, for the simulator's rank check (notes/04)
    PYTHONPATH=src python scripts/bench/replay_local.py --configs scripts/bench/configs_local.json

One server PROCESS per run, not one per invocation. Sharing a process would warm the prefix
cache across replicates, so run 1 would be `cache_miss_heavy` and run 6 `cache_hit_heavy` — two
different workloads averaged into one "band". Restarting costs ~20s and buys an independent
measurement, which is the whole point of a replicate.

Outputs, deliberately the layout the rest of the loop already reads:

    runs/<run_id>.json            one Vitals panel per run (arm=<name> in its notes)
    runs/<run_id>.csv             replay_trace's per-request rows
    runs/<group>/<label>.telemetry.csv   the engine's own per-request rows
    runs/<group>/engine_env.json  what the servers ran under

Every run also carries a **total accounting** block (notes/03): wall clock from the server
process's launch — model load, warm-up, idle and shutdown included, not from the first request —
plus the server's own peak RSS, device memory and storage reads, which reach the panel through
the sidecar `serve_accounted.py` writes at exit. That is what the primary metric printed after
each run divides: GPU-seconds per session at the class's p95 TTFT ceiling.

`ttft_queue_*` / `ttft_prefill_*` — the split LOOP.md calls the one that rules out whole classes
of fix — are filled from the telemetry rows, because the client cannot see where inside the
server its TTFT went. That is why the panel is built AFTER the server is stopped: SIGTERM is
what flushes the row store. `harness_config["split_source"]` records that those two came from
inside the engine while everything else came from the client's own clock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay_trace as rt  # noqa: E402
from replay_corpus_runpod import (  # noqa: E402
    free_port,
    make_client,
    read_telemetry,
    start_server,
    stop_server,
    tail,
    wait_ready,
)

from inference_server.research import harness as H  # noqa: E402
from inference_server.research.accounting import Accounting  # noqa: E402
from inference_server.research.corpus import load_trace  # noqa: E402
from inference_server.research.schemas import Vitals  # noqa: E402
from inference_server.research.session import primary_metric  # noqa: E402

SERVE_ENTRY = [str(Path(__file__).resolve().parent / "serve_accounted.py")]

# The engine every run starts from. Every knob `fit_timing_from_runs.SIM_KNOBS` needs is pinned
# explicitly, even where it is the engine's own default — an unrecorded knob means the simulator
# is asked to model a scheduler nobody can reconstruct.
#
# BACKEND=custom-mps, not mps: custom-* is the engine this project ships and the one the A100
# plan deploys, so a timing model fitted against anything else describes a code path we do not
# run. It costs ~2x the wall time of the HF backend on this box (row-by-row decode), which
# cold_start can afford.
#
# CUSTOM_BACKEND_BLOCKS and KV_CACHE_NUM_BLOCKS are set to the same number on purpose: the
# custom backend owns its own pool and ignores the CacheManager, but SIM_KNOBS reads
# KV_CACHE_NUM_BLOCKS for the simulator's `kv_blocks`, so they must agree or the simulated pool
# is not the served one.
ENGINE_BASE = {
    "BACKEND": "custom-mps",
    "DEVICE": "mps",
    "MODEL_NAME": "google/gemma-4-E2B-it",
    "MAX_BATCH_SIZE": "8",
    "PREFILL_MODE": "batched",
    "CUSTOM_BACKEND_BLOCKS": "4096",
    "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
    "CUSTOM_BACKEND_BLOCK_SIZE": "16",
    "KV_CACHE_NUM_BLOCKS": "4096",
    "KV_CACHE_BLOCK_SIZE": "16",
    "MAX_ACTIVE_KV_TOKENS": "0",        # unbounded: the simulator gates on blocks, not tokens
    "MAX_QUEUE_SIZE": "1000",
    "MAX_QUEUE_WAIT_S": "30.0",
    "SCHEDULING_POLICY": "fcfs",
    "PREFIX_CACHE_IMPL": "radix",
    "LOG_FORMAT": "text",
    "LOG_LEVEL": "WARNING",
}

# Recorded per panel, so a later run can be told apart from this one by machine as well as code.
ENGINE_RECORDED = tuple(ENGINE_BASE)

# harness_config keys compare.py already refuses across. Anything the sweep varies must land in
# one of these or two different configs would read as comparable.
CONFIG_TO_HARNESS_KEY = {
    "MODEL_NAME": "model",
    "MAX_BATCH_SIZE": "max_batch_size",
    "PREFILL_MODE": "prefill_mode",
    "CUSTOM_BACKEND_BLOCKS": "blocks",
    "CUSTOM_BACKEND_SLIDING_BLOCKS": "sliding_blocks",
    "MAX_QUEUE_WAIT_S": "max_queue_wait_s",
    "MAX_QUEUE_SIZE": "max_queue_size",
    "SCHEDULING_POLICY": "scheduling_policy",
    "PREFIX_CACHE_IMPL": "prefix_cache_impl",
}

WARMUP_PROMPT = ("Warm-up only. Reply with one word.")     # NOT from the corpus: a trace prompt
#                                                            here would leave the replay hitting
#                                                            its own prefix cache.


# --------------------------------------------------------------------------- identity

def hardware_name() -> str:
    """Apple silicon reports no GPU name; the SoC is the identity that matters here."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if torch.backends.mps.is_available():
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True).stdout.strip()
            return f"{brand or platform.machine()} (MPS)"
    except Exception:                       # noqa: BLE001 — identity only, never fatal
        pass
    return f"{platform.machine()} (cpu)"


def stamp_device_state(hardware: str) -> None:
    """What the RunPod launcher exports for an instrument; here the instrument is the launcher.

    `clocks_locked` is False, not None: macOS exposes no way to pin the GPU clock, so this is a
    measured refusal rather than an unknown, and compare.py should treat it as one.
    """
    os.environ.setdefault("RESEARCH_DEVICE_STATE", json.dumps({
        "gpu_name": hardware,
        "host_id": platform.node(),
        "clocks_locked": False,
        "lock_attempted": False,
        "provider": "local",
    }))


# --------------------------------------------------------------------------- run plan

@dataclass
class RunSpec:
    """One run: one server process, one replay."""

    label: str
    arm: str
    cls: str = "cold_start"
    split: str = "seen"
    rate_scale: float = 1.0
    env: dict[str, str] = field(default_factory=dict)      # overrides on ENGINE_BASE

    def engine_env(self) -> dict[str, str]:
        return {**ENGINE_BASE, **{k: str(v) for k, v in self.env.items()}}


def null_plan(cls: str, split: str, replicates: int, rate_scale: float) -> list[RunSpec]:
    """ABBA over two arms of the SAME config — a true null. `session.arms_for` requires the
    alternation, and it is also what stops a monotonic warm-up drift landing on one arm."""
    arms = ["baseline", "treatment"]
    plan = []
    for i in range(replicates):
        pair = arms if i % 2 == 0 else arms[::-1]          # AB BA AB BA ...
        for arm in pair:
            plan.append(RunSpec(label=f"null-{arm}-{i}", arm=arm, cls=cls, split=split,
                                rate_scale=rate_scale))
    return plan


def configs_plan(path: Path) -> list[RunSpec]:
    """A JSON list of {label, class, split, rate_scale, env} — one run each."""
    out = []
    for raw in json.loads(Path(path).read_text()):
        out.append(RunSpec(label=raw["label"], arm=raw.get("arm", raw["label"]),
                           cls=raw.get("class", "cold_start"), split=raw.get("split", "seen"),
                           rate_scale=float(raw.get("rate_scale", 1.0)),
                           env={k: str(v) for k, v in (raw.get("env") or {}).items()}))
    return out


# --------------------------------------------------------------------------- one run

def flatten_cache_stats(raw: dict[str, Any]) -> dict[str, Any]:
    """`/cache/stats` has two shapes: the CacheManager's flat one, and the custom backend's
    `{"backend":..., "prefix_cache": {...}}`. Reading only the flat one leaves `hit_rate` None,
    which `infer_regime` turns into `synthetic` — a panel that claims it served nothing."""
    inner = raw.get("prefix_cache")
    return dict(inner) if isinstance(inner, dict) else dict(raw)


def split_from_telemetry(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | None]:
    """p50/p95 of the queue and prefill halves of TTFT, over this replay's rows only.

    The client owns every other metric in the panel; these two it cannot see, because the split
    happens inside the scheduler. Measured 21ms queue vs 203ms prefill once killed every
    scheduling lever for TTFT p95 at a stroke, which is why the panel carries them at all.
    """
    mine = [r for r in rows
            if str(r.get("trace_id", "")).startswith(prefix + "-")
            and r.get("terminal_state") == "ok"]
    queue = [float(r["queue_wait_s"]) * 1000 for r in mine if r.get("queue_wait_s") is not None]
    prefill = [float(r["prefill_s"]) * 1000 for r in mine if r.get("prefill_s") is not None]
    return {
        "ttft_queue_p50": round(H.pct(queue, 0.50), 2) if queue else None,
        "ttft_queue_p95": round(H.pct(queue, 0.95), 2) if queue else None,
        "ttft_prefill_p50": round(H.pct(prefill, 0.50), 2) if prefill else None,
        "ttft_prefill_p95": round(H.pct(prefill, 0.95), 2) if prefill else None,
    }


async def warmup(client, n: int) -> int:
    from bench_serving import one_request
    done = 0
    for i in range(n):
        s = await one_request(client, WARMUP_PROMPT, 4,
                              headers={"X-Session-Id": f"warmup-{i}", "X-Turn-Index": "0",
                                       "X-Trace-Id": f"warmup-{i}"})
        done += s.error is None
    return done


async def replay_once(base_url: str, spec: RunSpec, *, warmup_n: int, drain_timeout_s: float):
    manifest, trace = load_trace(spec.cls, spec.split)
    prefix = rt.new_trace_prefix(spec.cls, spec.split)
    async with make_client(base_url) as client:
        if warmup_n:
            await warmup(client, warmup_n)
        res = await rt.run_replay(client, trace, rate_scale=spec.rate_scale,
                                  drain_timeout_s=drain_timeout_s, trace_prefix=prefix)
        sched = await rt.fetch_stats(client, "/scheduler/stats")
        cache = flatten_cache_stats(await rt.fetch_stats(client, "/cache/stats"))
    return manifest, res, sched, cache, prefix


def build_panel(spec: RunSpec, env: dict[str, str], manifest, res, sched, cache, prefix,
                telemetry: list[dict[str, Any]], hardware: str,
                accounting: Accounting | None = None) -> Vitals:
    cls = manifest.classes[spec.cls]
    panel = rt.build_panel(res, cls, corpus_version=manifest.corpus_version, split=spec.split,
                           rate_scale=spec.rate_scale, scheduler_stats=sched, cache_stats=cache,
                           trace_prefix=prefix)
    for k, v in split_from_telemetry(telemetry, prefix).items():
        setattr(panel, k, v)
    hc = panel.validity.harness_config
    hc.update({CONFIG_TO_HARNESS_KEY[k]: env[k] for k in CONFIG_TO_HARNESS_KEY if k in env})
    hc.update({
        "gpu": None,                    # no CUDA device; `hardware` carries the real identity
        "hardware": hardware,
        "backend": env["BACKEND"],
        "device": env["DEVICE"],
        "run_label": spec.label,
        "split_source": "telemetry (ttft_queue_*, ttft_prefill_*); client clock for the rest",
        # What fit_timing_from_runs.sim_config needs. Per panel, because one local run group
        # deliberately holds several engine configs — unlike a rented run, where the whole pod
        # ran one.
        "engine_env": {k: env[k] for k in ENGINE_RECORDED if k in env},
    })
    panel.validity.notes = (f"arm={spec.arm} label={spec.label} local replay on {hardware}, "
                            f"fresh server per run, trial={H.trial_id()}")
    if accounting is not None:
        accounting.apply(panel)
    return panel


def build_accounting(sidecar_path: Path, *, wall_from_launch_s: float, serving_wall_s: float,
                     sessions: int) -> Accounting:
    """What the whole run cost, from the two vantage points that can each see half of it.

    The instrument owns the clock — `wall_from_launch_s` starts before `start_server` and ends
    after the server has exited, so model load, warm-up, idle and shutdown are all inside it.
    That is the "wall clock from process start, not from first successful request" term, and it
    is why `Vitals.wall_s` (the replay window) is NOT reused for it.

    The server owns its own resident set, its device and its file reads, so those arrive through
    the sidecar `serve_accounted.py` writes at exit. A missing sidecar (server died before
    writing one) leaves those terms None and NAMED in `unmeasured`, never zeroed.
    """
    side: dict[str, Any] = {}
    if sidecar_path.exists():
        try:
            side = json.loads(sidecar_path.read_text())
        except json.JSONDecodeError:
            side = {}
    unmeasured: dict[str, str] = {}
    if not side:
        unmeasured["server resources"] = (f"no accounting sidecar at {sidecar_path}; the server "
                                          f"exited without writing one")
    if side.get("peak_device_mem_gb") is None:
        unmeasured.setdefault("peak_device_mem_gb", "no CUDA or MPS device in the server process")
    if side.get("storage_read_bytes") is None:
        unmeasured.setdefault("storage_read_bytes",
                              side.get("storage_read_note") or "not readable on this platform")
    # The sampler is a Python thread inside the server, so it is starved of the GIL exactly when
    # the engine is busy. Coverage short of the run is a PARTIAL measurement, and saying which
    # window a peak covers is the difference between a lower bound and a wrong number.
    sampled_to = side.get("sampled_to_uptime_s")
    if sampled_to is not None and sampled_to < 0.9 * wall_from_launch_s:
        unmeasured["peak_device_mem_gb (coverage)"] = (
            f"sampled only the first {sampled_to:.0f}s of a {wall_from_launch_s:.0f}s run "
            f"({side.get('device_mem_samples')} samples); the sampler thread is GIL-starved "
            f"while the engine computes, so the peak is a lower bound over that window")
    return Accounting(
        wall_s_from_process_start=round(wall_from_launch_s, 3),
        serving_wall_s=round(serving_wall_s, 3),
        sessions_served=sessions,
        peak_host_rss_gb=side.get("peak_host_rss_gb"),
        peak_device_mem_gb=side.get("peak_device_mem_gb"),
        device_mem_source=side.get("device_mem_source"),
        storage_read_bytes=side.get("storage_read_bytes"),
        unmeasured=unmeasured,
    )


def run_one(spec: RunSpec, *, runs_dir: Path, group_dir: Path, hardware: str,
            warmup_n: int, drain_timeout_s: float, ready_timeout_s: float) -> Vitals | None:
    env = spec.engine_env()
    telemetry_dir = Path(tempfile.mkdtemp(prefix=f"telemetry-{spec.label}-"))
    env["TELEMETRY_DIR"] = str(telemetry_dir)
    sidecar = telemetry_dir / "accounting.json"
    env["ACCOUNTING_SIDECAR"] = str(sidecar)
    log_path = telemetry_dir / "server.log"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"

    print(f"\n== {spec.label} ({spec.arm}) {spec.cls}/{spec.split} x{spec.rate_scale:g} "
          f"mbs={env['MAX_BATCH_SIZE']} policy={env['SCHEDULING_POLICY']} "
          f"deadline={env['MAX_QUEUE_WAIT_S']}s ==", flush=True)
    t0 = time.perf_counter()               # the process's START: everything after is its cost
    proc = start_server(env, port, log_path, entry=SERVE_ENTRY)
    try:
        if not wait_ready(base_url, proc, ready_timeout_s):
            print("\n".join(["  server never became ready; log tail:"] + tail(log_path)),
                  file=sys.stderr, flush=True)
            return None
        print(f"  ready in {time.perf_counter() - t0:.1f}s", flush=True)
        out = asyncio.run(replay_once(base_url, spec, warmup_n=warmup_n,
                                      drain_timeout_s=drain_timeout_s))
    finally:
        stop_server(proc)                 # SIGTERM, so the lifespan flushes the telemetry rows
    wall_from_launch_s = time.perf_counter() - t0

    manifest, res, sched, cache, prefix = out
    telemetry = read_telemetry(telemetry_dir)
    summary = res.summary(manifest.classes[spec.cls])
    rt.print_summary(manifest.classes[spec.cls], summary)
    if not summary["n_ok"]:
        print("  no request succeeded; no panel for this run", flush=True)
        return None

    accounting = build_accounting(
        sidecar, wall_from_launch_s=wall_from_launch_s, serving_wall_s=res.wall_s,
        sessions=len({r.session_id for r in res.rows if r.error is None}))
    print(accounting.summary(), flush=True)
    panel = build_panel(spec, env, manifest, res, sched, cache, prefix, telemetry, hardware,
                        accounting=accounting)
    print(primary_metric(panel).format(), flush=True)
    path = H.emit(panel, label=spec.label, runs_dir=runs_dir)
    rt.write_rows(path.with_suffix(".csv"), res.rows)
    write_csv(group_dir / f"{spec.label}.telemetry.csv", telemetry)
    return panel


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", default="cold_start")
    ap.add_argument("--split", default="seen", choices=("seen", "heldout"))
    ap.add_argument("--null", type=int, default=0, metavar="N",
                    help="null experiment: N replicates of the SAME config per arm, ABBA")
    ap.add_argument("--configs", help="JSON list of run specs; one run per config")
    ap.add_argument("--rate-scale", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0,
                    help="discarded warm-up requests per run; 0 (default) keeps a cold_start "
                         "replay describing a genuinely cold replica")
    ap.add_argument("--drain-timeout-s", type=float, default=300.0)
    ap.add_argument("--ready-timeout-s", type=float, default=600.0)
    ap.add_argument("--runs-dir", default=str(H.RUNS_DIR))
    args = ap.parse_args(argv)

    if bool(args.null) == bool(args.configs):
        ap.error("pass exactly one of --null N or --configs FILE")

    plan = (null_plan(args.cls, args.split, args.null, args.rate_scale) if args.null
            else configs_plan(Path(args.configs)))
    hardware = hardware_name()
    stamp_device_state(hardware)
    runs_dir = Path(args.runs_dir)
    group_dir = runs_dir / H.run_group()
    group_dir.mkdir(parents=True, exist_ok=True)

    print(f"-- {len(plan)} run(s) on {hardware}, run_group={H.run_group()} --", flush=True)
    panels: list[Vitals] = []
    failed: list[str] = []
    for spec in plan:
        try:
            p = run_one(spec, runs_dir=runs_dir, group_dir=group_dir, hardware=hardware,
                        warmup_n=args.warmup, drain_timeout_s=args.drain_timeout_s,
                        ready_timeout_s=args.ready_timeout_s)
        except Exception as e:            # noqa: BLE001 — one dead run, not a dead plan
            import traceback
            traceback.print_exc()
            p, e = None, f"{type(e).__name__}: {e}"
            print(f"  {spec.label} FAILED: {e}", file=sys.stderr, flush=True)
        if p is None:
            failed.append(spec.label)
        else:
            panels.append(p)

    (group_dir / "engine_env.json").write_text(json.dumps({
        "engine_env": ENGINE_BASE,
        "hardware": hardware,
        "plan": [[s.cls, s.split, s.rate_scale] for s in plan],
        "runs": {s.label: {"arm": s.arm, "class": s.cls, "split": s.split,
                           "rate_scale": s.rate_scale, "engine_env": s.engine_env()}
                 for s in plan},
    }, indent=2, sort_keys=True))

    print(f"\n{len(panels)}/{len(plan)} run(s) produced a panel; run_group={H.run_group()}")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
    print(f"group dir: {group_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
