#!/usr/bin/env python
"""Fit the simulator's TimingModel from a replay run group, then check it ranks like hardware.

    python scripts/tools/fit_timing_from_runs.py <run_group>                  # fit + write
    python scripts/tools/fit_timing_from_runs.py <run_group> --validate       # rank check

Input is what `run_on_runpod.py` wrote for a `replay_corpus_runpod.py` run: the hardware panels
`runs/<run_id>.json` whose `validity.run_group` matches, and `runs/<run_group>/*.telemetry.csv`
plus `engine_env.json`. Local, no GPU, no knowledge entry written — the fitted model goes to
`knowledge/timing/<model>-<hardware>-<sha>.json` (`--out` overrides) and the validation result
is printed for a human to file.

Row mapping, from `telemetry.RequestRecord` to what `simulator.fit_timing_model` expects. Only
`terminal_state == "ok"` rows are used.

    prefill_s      <- prefill_s          (first_token_ts - admit_ts: the span the simulator
                                          charges per prefill)
    prompt_tokens  <- prompt_tokens - cache_hit_tokens
                                         (the simulator passes UNCACHED tokens as the first
                                          prefill_s argument; a prefix hit shrinks the work)
    decode_step_s  <- decode_s / decode_steps   (rows with decode_steps > 0)
    batch_size     <- active_size + 1    (see the caveat below)

`batch_prompt_tokens` and `total_kv_tokens` are not in the telemetry row, so their coefficients
fit to 0. That is the v1 model: cost linear in the request's own size and in batch width.

The batch_size caveat, which bounds what this fit can claim. `active_size` is snapshotted ONCE,
at enqueue, before the request is admitted, and is never updated — so it is the batch width this
request ARRIVED into, not the width it decoded in. A request that arrives alone and then decodes
for 200 steps inside a batch of 30 contributes the row (batch_size=1, decode_step_s measured at
~30 rows wide). "+1" corrects only for the row's own slot. Under steady load the two are close
and the decode slope is usable; under a ramp the predictor is biased low and the fitted slope is
too flat. Fixing it needs a per-step width in the telemetry row (an engine change, not done here)
— until then, prefer rate scales that hold a steady width, and read the decode slope as a
lower bound.

Validation (--validate): for each hardware panel's (class, split, rate_scale), simulate the same
trace under a SimConfig built from `engine_env.json`, then report Spearman rank correlation of
`ttft_p95` and `tpot_p50` between hardware and simulator across the configs. Needs >= 2 configs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from inference_server.research.corpus import load_trace  # noqa: E402
from inference_server.research.schemas import Vitals  # noqa: E402
from inference_server.research.simulator import (  # noqa: E402
    SimConfig,
    TimingModel,
    fit_timing_model,
    rank_correlation,
    simulate,
)

RUNS_DIR = REPO / "runs"
TIMING_DIR = REPO / "knowledge" / "timing"
HARDWARE_HARNESS = "replay_trace"


# ------------------------------------------------------------------ loading

def load_group(runs_dir: Path, run_group: str) -> tuple[list[Vitals], dict[str, Any],
                                                         list[dict[str, Any]]]:
    """(hardware panels, engine_env.json contents, telemetry rows) for one run group."""
    panels = []
    for p in sorted(runs_dir.glob("*.json")):
        v = Vitals.load(p)
        if v.validity.run_group == run_group and v.validity.harness == HARDWARE_HARNESS:
            panels.append(v)
    group_dir = runs_dir / run_group
    meta_path = group_dir / "engine_env.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows: list[dict[str, Any]] = []
    for path in sorted(group_dir.glob("*.telemetry.csv")):
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return panels, meta, rows


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def fit_rows(telemetry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The docstring's mapping, one fit row per successful telemetry row."""
    out = []
    for r in telemetry:
        if r.get("terminal_state") != "ok":
            continue
        prefill = _num(r.get("prefill_s"))
        decode_s, steps = _num(r.get("decode_s")), int(_num(r.get("decode_steps")) or 0)
        row: dict[str, Any] = {
            "prompt_tokens": int(_num(r["prompt_tokens"]) or 0)
                             - int(_num(r.get("cache_hit_tokens")) or 0),
            "prefill_s": prefill,
            "batch_size": int(_num(r["active_size"]) or 0) + 1,
            "decode_step_s": (decode_s / steps if decode_s is not None and steps > 0 else None),
        }
        out.append(row)
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "unknown"


# ------------------------------------------------------------------ validation

# Every knob the simulated scheduler shares with the served one. Absent from engine_env is a
# REFUSAL, not a default: SimConfig's own defaults are not the engine's (max_queue_size 256 vs
# the engine's 1000), so falling back silently simulates a queue that rejects far earlier than
# the hardware did — in exactly the overload regime the rank correlation is meant to judge.
SIM_KNOBS = ("MAX_BATCH_SIZE", "MAX_QUEUE_SIZE", "MAX_QUEUE_WAIT_S", "KV_CACHE_NUM_BLOCKS",
             "KV_CACHE_BLOCK_SIZE", "SCHEDULING_POLICY", "PREFILL_MODE")


def sim_config(engine_env: dict[str, str], rate_scale: float) -> SimConfig:
    """The simulator knobs the served engine actually ran with. Raises on any knob it did not
    record — `replay_corpus_runpod.ENGINE_DEFAULTS` pins all of them for exactly this reason."""
    missing = [k for k in SIM_KNOBS if not engine_env.get(k)]
    if missing:
        raise ValueError(
            f"engine_env does not record {', '.join(missing)}, so the simulated scheduler would "
            f"not be the one that ran. Re-run the instrument (its ENGINE_DEFAULTS pin these), or "
            f"pass the values explicitly if you know what the server used.")
    return SimConfig(
        max_batch_size=int(engine_env["MAX_BATCH_SIZE"]),
        max_queue_size=int(engine_env["MAX_QUEUE_SIZE"]),
        max_queue_wait_s=float(engine_env["MAX_QUEUE_WAIT_S"]),
        kv_blocks=int(engine_env["KV_CACHE_NUM_BLOCKS"]),
        block_size=int(engine_env["KV_CACHE_BLOCK_SIZE"]),
        policy=engine_env["SCHEDULING_POLICY"],
        prefill_mode="batched" if engine_env["PREFILL_MODE"] == "batched" else "monolithic",
        rate_scale=rate_scale,
    )


def validate(panels: list[Vitals], timing: TimingModel,
             engine_env: dict[str, str]) -> dict[str, Any]:
    """Simulate every hardware config; return per-config pairs and the two Spearman rhos."""
    pairs = []
    for p in panels:
        hc = p.validity.harness_config
        cls_name, split, scale = hc["workload_class"], hc["split"], float(hc["rate_scale"])
        manifest, trace = load_trace(cls_name, split)
        s = simulate(trace, sim_config(engine_env, scale), timing).summary(
            manifest.classes[cls_name])
        pairs.append({"class": cls_name, "split": split, "rate_scale": scale,
                      "hw_ttft_p95": p.ttft_p95, "sim_ttft_p95": s["ttft_p95"],
                      "hw_tpot_p50": p.tpot_p50, "sim_tpot_p50": s["tpot_p50"]})
    pairs.sort(key=lambda d: (d["class"], d["split"], d["rate_scale"]))
    rho = {}
    for metric in ("ttft_p95", "tpot_p50"):
        hw = [d[f"hw_{metric}"] for d in pairs]
        sim = [d[f"sim_{metric}"] for d in pairs]
        ok = [i for i in range(len(pairs)) if hw[i] is not None and sim[i] is not None]
        rho[metric] = (rank_correlation([hw[i] for i in ok], [sim[i] for i in ok])
                       if len(ok) >= 2 else None)
    return {"pairs": pairs, "rank_correlation": rho}


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_group")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--out", default=None, help="timing model path; default "
                    "knowledge/timing/<model>-<hardware>-<sha>.json")
    ap.add_argument("--model", default=None, help="identity override (default: engine_env)")
    ap.add_argument("--hardware", default=None)
    ap.add_argument("--sha", default=None, help="default: the panels' engine_sha")
    ap.add_argument("--validate", action="store_true",
                    help="load the fitted model (--timing or the default --out path) and "
                         "rank-check it against the group's hardware panels")
    ap.add_argument("--timing", default=None, help="fitted model to validate")
    args = ap.parse_args(argv)

    panels, meta, telemetry = load_group(Path(args.runs_dir), args.run_group)
    engine_env = meta.get("engine_env", {})
    model = args.model or engine_env.get("MODEL_NAME", "unknown")
    hardware = args.hardware or meta.get("hardware") or "unknown"
    sha = args.sha or (panels[0].validity.engine_sha if panels else "unknown")
    out = Path(args.out) if args.out else TIMING_DIR / f"{slug(model)}-{slug(hardware)}-{sha}.json"

    if args.validate:
        path = Path(args.timing) if args.timing else out
        if not path.exists():
            print(f"[error] no timing model at {path}; fit first", file=sys.stderr)
            return 1
        if len(panels) < 2:
            print(f"[error] rank correlation needs >= 2 hardware panels in {args.run_group}, "
                  f"found {len(panels)}", file=sys.stderr)
            return 1
        timing = TimingModel.from_json(path)
        v = validate(panels, timing, engine_env)
        print(f"-- validate {path.name} against {len(panels)} hardware panel(s) "
              f"(run_group={args.run_group}) --")
        print(f"{'config':<34} {'hw ttft_p95':>12} {'sim ttft_p95':>13} "
              f"{'hw tpot_p50':>12} {'sim tpot_p50':>13}")
        for d in v["pairs"]:
            name = f"{d['class']}/{d['split']} x{d['rate_scale']:g}"
            print(f"{name:<34} {d['hw_ttft_p95']!s:>12} {d['sim_ttft_p95']!s:>13} "
                  f"{d['hw_tpot_p50']!s:>12} {d['sim_tpot_p50']!s:>13}")
        for metric, rho in v["rank_correlation"].items():
            print(f"spearman {metric}: {rho if rho is None else round(rho, 3)}")
        if engine_env.get("BACKEND", "").startswith("custom"):
            print(f"note: BACKEND={engine_env['BACKEND']} gates admission on the token budget "
                  f"MAX_ACTIVE_KV_TOKENS={engine_env.get('MAX_ACTIVE_KV_TOKENS', '?')}, while the "
                  f"simulator gates on a block pool (kv_blocks). Under KV pressure the two "
                  f"constraints differ, so disagreement there is the model's, not the engine's.")
        print("simulated: ranks configurations, makes no absolute claim; file the result as a "
              "knowledge entry by hand with this run_group as evidence")
        return 0

    rows = fit_rows(telemetry)
    if not rows:
        print(f"[error] no ok telemetry rows under {Path(args.runs_dir) / args.run_group}",
              file=sys.stderr)
        return 1
    timing = fit_timing_model(rows, model=model, hardware=hardware, engine_sha=sha,
                              fitted_from=args.run_group)
    out.parent.mkdir(parents=True, exist_ok=True)
    timing.to_json(out)
    n_pre = sum(r["prefill_s"] is not None for r in rows)
    n_dec = sum(r["decode_step_s"] is not None for r in rows)
    print(f"-- fit from {len(rows)} ok rows ({n_pre} prefill, {n_dec} decode) in "
          f"run_group={args.run_group} --")
    a, b, c = timing.prefill
    print(f"prefill_s     = {a:.6f} + {b:.3e} * prompt_tokens + {c:.3e} * batch_prompt_tokens")
    a, b, c = timing.decode
    print(f"decode_step_s = {a:.6f} + {b:.3e} * batch_size + {c:.3e} * total_kv_tokens")
    rel = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"wrote {rel} (model={model}, hardware={hardware}, sha={sha})")
    print(f"validate with: python scripts/tools/fit_timing_from_runs.py {args.run_group} "
          f"--validate" + (f" --timing {rel}" if args.out else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
