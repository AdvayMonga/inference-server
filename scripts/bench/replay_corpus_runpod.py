#!/usr/bin/env python
"""Serve the engine on a rented GPU, replay corpus classes against it, bring the rows home.

The first GPU job of the plan: the simulator's timing model has to be FITTED from telemetry
rows measured on real hardware, and then VALIDATED by ranking configurations the way the
hardware does (notes 04-simulator.md). This instrument produces both inputs in one rental.

    scripts/tools/run_on_runpod.py scripts/bench/replay_corpus_runpod.py
    REPLAY_CLASSES=steady_interactive,long_context REPLAY_RATE_SCALES=1,2 \\
        scripts/tools/run_on_runpod.py scripts/bench/replay_corpus_runpod.py

On the pod: start `inference_server.server` as a uvicorn subprocess on 127.0.0.1 with
`TELEMETRY_DIR` set (one SQLite row per request), wait for `/ready`, warm up, then replay each
`(class, split, rate_scale)` in-process through `replay_trace.run_replay` — the client measures
TTFT/TPOT on its own clock, the engine's stats fill the panel's pressure fields only. Stop the
server (SIGTERM, so the lifespan flushes the row store), read the SQLite file(s) back, and
partition the rows per replay by the `<class>-<split>-<nonce>-` trace-id prefix that
`replay_trace` mints. The warm-up's rows carry a different prefix and are dropped by that join.

Warm-up is sequential single requests, not an open-loop burst: with `CUSTOM_BACKEND_COMPILE=1`
the first requests trigger lazy torch.compile, and a burst behind that compile becomes one huge
prefill wave (the 143GiB OOM that killed a Modal run).

One server process serves every replay, so the prefix cache warms across them: a `seen` trace
replayed at x1 then x2 hits on the second pass. The panel's `workload_regime` records that
honestly; restarting the server per replay would cost a model load each.

Payload (the venue contract, `research/venues.py`):

    {"panels": [Vitals dicts], "engine_env": {...}, "hardware": "...",
     "replays": [{"class", "split", "rate_scale", "trace_prefix", "summary",
                  "rows": [replay_trace.Row dicts], "telemetry_rows": [RequestRecord dicts]}]}

If the server never becomes ready the payload carries `"error"` and the server log's last 50
lines instead, and the exit code is non-zero.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay_trace as rt  # noqa: E402
from bench_serving import one_request  # noqa: E402
from inference_server.config import Settings  # noqa: E402
from inference_server.research.corpus import load_trace  # noqa: E402
from inference_server.research.venues import emit_payload  # noqa: E402

# Engine env the server subprocess gets. Defaults mirror modal_app.py so a panel measured here
# is comparable to the Modal ones; anything already in the environment wins.
#
# Every knob the simulator needs is pinned EXPLICITLY, even where the value is just the engine's
# own default. An unset knob is not recorded in engine_env, and fit_timing_from_runs then has to
# guess it: MAX_QUEUE_SIZE unset meant --validate simulated a 256-deep queue against a server
# that ran 1000-deep, which rejects far earlier than reality in exactly the overload regime the
# rank correlation is supposed to judge.
ENGINE_DEFAULTS = {
    "BACKEND": "custom-cuda",
    "MODEL_NAME": "google/gemma-4-E4B-it",
    "MAX_BATCH_SIZE": "256",
    "PREFILL_MODE": "batched",
    # 0, unlike modal_app.py. This is a calibration run, and the decode-graph compile ladder at
    # MAX_BATCH_SIZE=256 costs ~21 minutes on a cold box (knowledge/kb-20260902-008.json) — with
    # the model download and three rate scales that risks the venue's run timeout, and a run that
    # is killed mid-print emits NO payload, so the whole rental yields nothing. The timing model
    # then describes eager decode; a refit with compile on is a second, cheaper run (the weights
    # are cached by then) once this one has proved the pipeline.
    "CUSTOM_BACKEND_COMPILE": "0",
    "CUSTOM_BACKEND_BLOCKS": "8192",
    "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
    "KV_CACHE_NUM_BLOCKS": "16384",
    "MAX_ACTIVE_KV_TOKENS": "200000",
    "LOG_FORMAT": "text",
    # Engine defaults, pinned so they are recorded rather than inherited. See SIM_KNOBS in
    # scripts/tools/fit_timing_from_runs.py, which refuses to simulate without them.
    "MAX_QUEUE_SIZE": str(Settings.max_queue_size),
    "MAX_QUEUE_WAIT_S": str(Settings.max_queue_wait_s),
    "KV_CACHE_BLOCK_SIZE": str(Settings.kv_cache_block_size),
    "SCHEDULING_POLICY": str(Settings.scheduling_policy),
}
# Passed through when set; never defaulted. HF_TOKEN reaches the server but never the payload.
ENGINE_PASSTHROUGH = ("PREFILL_CHUNK_SIZE", "WAVE_WINDOW_MULT", "CONTEXT_WINDOW",
                      "CUSTOM_BACKEND_PREFILL_GRAPH", "HF_TOKEN", "HF_HOME",
                      "TORCHINDUCTOR_CACHE_DIR", "CUSTOM_BACKEND_LAUNCH_TABLE")
SECRET_KEYS = ("HF_TOKEN",)

WARMUP_MAX_TOKENS = 16


# ------------------------------------------------------------------ configuration

def engine_env(base: dict[str, str]) -> dict[str, str]:
    """The env the server runs under: defaults, passthrough, and a fresh TELEMETRY_DIR."""
    env = dict(ENGINE_DEFAULTS)
    for k in list(ENGINE_DEFAULTS) + list(ENGINE_PASSTHROUGH):
        if k in base:
            env[k] = base[k]
    env["TELEMETRY_DIR"] = base.get("TELEMETRY_DIR") or tempfile.mkdtemp(prefix="telemetry-")
    return env


def public_env(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if k not in SECRET_KEYS}


def parse_plan(base: dict[str, str]) -> list[tuple[str, str, float]]:
    """(class, split, rate_scale) triples from REPLAY_CLASSES / REPLAY_SPLITS / REPLAY_RATE_SCALES."""
    classes = [c for c in base.get("REPLAY_CLASSES", "steady_interactive").split(",") if c]
    splits = [s for s in base.get("REPLAY_SPLITS", "seen").split(",") if s]
    scales = [float(x) for x in base.get("REPLAY_RATE_SCALES", "1,2,4").split(",") if x]
    return [(c, s, r) for c in classes for s in splits for r in scales]


def hardware_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                                # noqa: BLE001 — identity only, never fatal
        return "unknown"


# ------------------------------------------------------------------ the server subprocess

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(env: dict[str, str], port: int, log_path: Path,
                 entry: list[str] | None = None) -> subprocess.Popen:
    """uvicorn in a subprocess, engine env applied, stdout+stderr to `log_path`.

    `entry` replaces the `-m uvicorn ...` part for a caller that needs the server started a
    different way — `replay_local.py` passes `serve_accounted.py`, which is the same app plus a
    resource sidecar. Default unchanged, so the pod path starts exactly the process it always did.
    """
    cmd = [sys.executable, *(entry or ["-m", "uvicorn", "inference_server.server:app"]),
           "--host", "127.0.0.1", "--port", str(port), "--log-level", "info"]
    full = {**os.environ, **env, "PYTHONPATH": str(REPO / "src")}
    log = open(log_path, "w")
    return subprocess.Popen(cmd, env=full, stdout=log, stderr=subprocess.STDOUT)


def wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: float,
               poll_s: float = 2.0) -> bool:
    """Poll /ready until 200. False if the server exits or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"{base_url}/ready", timeout=5.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(poll_s)
    return False


def stop_server(proc: subprocess.Popen, timeout_s: float = 60.0) -> None:
    """SIGTERM so uvicorn runs the lifespan shutdown (scheduler.stop -> telemetry flush)."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def tail(path: Path, n: int = 50) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def make_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, limits=httpx.Limits(max_connections=1024))


# ------------------------------------------------------------------ telemetry read-back

def read_telemetry(directory: str | Path) -> list[dict[str, Any]]:
    """Every row from every `<run_id>.sqlite` the server wrote (one per server process)."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob("*.sqlite")):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            rows += [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY arrival_ts")]
        finally:
            conn.close()
    return rows


def attach_telemetry(replays: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Join by trace-id prefix: `<trace_prefix>-<i>` is the key a replay row and a telemetry
    row share. Warm-up rows match no replay and are dropped here."""
    for rep in replays:
        pre = rep["trace_prefix"] + "-"
        rep["telemetry_rows"] = [r for r in rows if str(r.get("trace_id", "")).startswith(pre)]


# ------------------------------------------------------------------ the replays

async def warmup(client: httpx.AsyncClient, cls_name: str, split: str, n: int,
                 prompt_format: str = rt.DEFAULT_PROMPT_FORMAT) -> int:
    """Sequential single requests, awaited one at a time, so lazy compile sees an empty queue.

    Same route as the replay it precedes: the chat route encodes a longer prompt and takes a
    different code path into the tokenizer, so warming the other one warms the wrong thing."""
    _, trace = load_trace(cls_name, split)
    prefix = rt.new_trace_prefix("warmup", split)
    done = 0
    for i, req in enumerate(trace[:n]):
        s = await one_request(client, req.prompt, min(req.max_tokens, WARMUP_MAX_TOKENS),
                              headers={"X-Session-Id": f"warmup-{i}", "X-Turn-Index": "0",
                                       "X-Trace-Id": f"{prefix}-{i}"},
                              prompt_format=prompt_format)
        done += s.error is None
    return done


async def run_plan(client: httpx.AsyncClient, plan: list[tuple[str, str, float]],
                   replays: list[dict[str, Any]], panels: list[dict[str, Any]], *,
                   warmup_n: int, drain_timeout_s: float, settle_s: float = 2.0,
                   prompt_format: str = rt.DEFAULT_PROMPT_FORMAT,
                   model_name: str | None = None) -> None:
    """Warm up, then one open-loop replay per (class, split, rate_scale).

    Appends into the caller's `replays` and `panels` rather than returning them, and catches per
    replay, because both are money already spent: an exception on the last config used to discard
    the panels of the ones that succeeded, and a rental that yields nothing is the expensive
    failure this whole module is arranged around. A failed replay keeps its `error` and the plan
    continues — a dead server makes the rest fail fast anyway, and says so in their errors.
    """
    if warmup_n and plan:
        ok = await warmup(client, plan[0][0], plan[0][1], warmup_n, prompt_format)
        print(f"[replay] warmup {ok}/{warmup_n} ok (discarded)", flush=True)
    for cls_name, split, rate_scale in plan:
        prefix = rt.new_trace_prefix(cls_name, split)
        rep: dict[str, Any] = {"class": cls_name, "split": split, "rate_scale": rate_scale,
                               "trace_prefix": prefix, "summary": {}, "rows": [],
                               "telemetry_rows": [], "error": None}
        replays.append(rep)
        try:
            manifest, trace = load_trace(cls_name, split)
            cls = manifest.classes[cls_name]
            print(f"-- replay {cls_name}/{split} x{rate_scale:g} ({len(trace)} requests, "
                  f"trace ids {prefix}-<i>) --", flush=True)
            res = await rt.run_replay(client, trace, rate_scale=rate_scale,
                                      drain_timeout_s=drain_timeout_s, trace_prefix=prefix,
                                      prompt_format=prompt_format)
            sched = await rt.fetch_stats(client, "/scheduler/stats")
            cache = await rt.fetch_stats(client, "/cache/stats")
            summary = res.summary(cls)
            rt.print_summary(cls, summary)
            rep["summary"] = summary
            rep["rows"] = [asdict(r) for r in res.rows]
            if summary["n_ok"]:
                panel = rt.build_panel(res, cls, corpus_version=manifest.corpus_version,
                                       split=split, rate_scale=rate_scale, scheduler_stats=sched,
                                       cache_stats=cache, trace_prefix=prefix,
                                       prompt_format=prompt_format, model_name=model_name,
                                       trace=trace)
                panel.validate()                 # fail here, in the run, not on the way home
                panels.append(panel.to_dict())
            else:
                print("  no request succeeded; no panel for this replay", flush=True)
        except Exception as e:                   # noqa: BLE001 — one dead config, not a dead run
            rep["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            print(f"[replay] {cls_name}/{split} x{rate_scale:g} FAILED: {rep['error']}",
                  file=sys.stderr, flush=True)
        await asyncio.sleep(settle_s)


# ------------------------------------------------------------------ entry point

def main() -> int:
    env = engine_env(dict(os.environ))
    plan = parse_plan(os.environ)
    warmup_n = int(os.environ.get("REPLAY_WARMUP_N", "8"))
    drain_timeout_s = float(os.environ.get("REPLAY_DRAIN_TIMEOUT_S", "300"))
    ready_timeout_s = float(os.environ.get("REPLAY_READY_TIMEOUT_S", "1800"))
    prompt_format = os.environ.get("REPLAY_PROMPT_FORMAT", rt.DEFAULT_PROMPT_FORMAT)

    telemetry_dir = Path(env["TELEMETRY_DIR"])
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    log_path = telemetry_dir / "server.log"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    payload: dict[str, Any] = {"panels": [], "replays": [], "engine_env": public_env(env),
                               "hardware": hardware_name(), "plan": plan}
    replays: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []

    print(f"[replay] starting server on {base_url} (model={env['MODEL_NAME']} "
          f"backend={env['BACKEND']} telemetry={telemetry_dir})", flush=True)
    proc = start_server(env, port, log_path)
    try:
        if not wait_ready(base_url, proc, ready_timeout_s):
            lines = tail(log_path)
            payload["error"] = (f"server never became ready within {ready_timeout_s:.0f}s "
                                f"(exit={proc.poll()})")
            payload["server_log_tail"] = lines
            print("\n".join(["[replay] server log tail:"] + lines), file=sys.stderr, flush=True)
            print(emit_payload(payload), flush=True)
            return 1
        print("[replay] server ready", flush=True)

        async def go():
            async with make_client(base_url) as client:
                await run_plan(client, plan, replays, panels, warmup_n=warmup_n,
                               drain_timeout_s=drain_timeout_s, prompt_format=prompt_format,
                               model_name=env["MODEL_NAME"])
        try:
            asyncio.run(go())
        except Exception as e:                   # noqa: BLE001 — whatever ran is still evidence
            traceback.print_exc()
            payload["error"] = f"replay plan aborted: {type(e).__name__}: {e}"
    finally:
        stop_server(proc)

    rows = read_telemetry(telemetry_dir)
    attach_telemetry(replays, rows)
    print(f"[replay] {len(rows)} telemetry rows read back; "
          + ", ".join(f"{r['class']}/{r['split']} x{r['rate_scale']:g}: "
                      f"{len(r['telemetry_rows'])}" for r in replays), flush=True)
    payload["panels"], payload["replays"] = panels, replays
    failed = [f"{r['class']}/{r['split']} x{r['rate_scale']:g}" for r in replays if r["error"]]
    if failed:
        payload["error"] = ((payload.get("error", "") + "; ") if payload.get("error") else "") \
            + f"{len(failed)} replay(s) failed: {', '.join(failed)}"
    if payload.get("error"):
        payload["server_log_tail"] = tail(log_path)
    print(emit_payload(payload), flush=True)
    # Partial evidence is still evidence and is emitted above; the exit code says it is partial.
    return 1 if payload.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
