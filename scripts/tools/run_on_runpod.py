#!/usr/bin/env python
"""Launch an instrument on a rented RunPod GPU. The RunPod counterpart of run_instrument.sh.

Same provenance contract as the Modal path, for the same reason: the box has no git repo, so the
engine sha and the run_group must be stamped by the launcher or the panel cannot be attributed.

    scripts/tools/run_on_runpod.py scripts/bench/replay_corpus_runpod.py
    RESEARCH_RUN_GROUP=exp-42 scripts/tools/run_on_runpod.py <instrument>   # arm A, then arm B

Panels land in `runs/<run_id>.json`. An instrument that also returns `replays` (per-request rows
and telemetry rows, see replay_corpus_runpod.py) gets them written as CSV under
`runs/<run_group>/`, next to the panels, with the engine env the server ran under.

Arms of one experiment MUST share RESEARCH_RUN_GROUP, and — the lesson that cost us an A100 day —
should share one POD. Two pods are two machines, and two identical A100-80GB draws on Modal
differed by 2.31x on byte-identical config. Prefer an instrument that runs both arms in-process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from time import strftime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from inference_server.research.venues import PodSpec, VenueError, run_instrument  # noqa: E402


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def provenance() -> dict[str, str]:
    dirty = "1" if _git("status", "--porcelain") else "0"
    return {
        "RESEARCH_ENGINE_SHA": (os.environ.get("RESEARCH_ENGINE_SHA")
                                or _git("rev-parse", "--short", "HEAD")),
        "RESEARCH_ENGINE_DIRTY": os.environ.get("RESEARCH_ENGINE_DIRTY", dirty),
        "RESEARCH_RUN_GROUP": os.environ.get("RESEARCH_RUN_GROUP")
                              or f"grp-{strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}",
    }


def hf_token() -> tuple[str | None, str]:
    """(token, where it came from). `hf auth login` leaves it in the cache file; read that when
    the env has none so a gated model downloads without exporting the secret by hand."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "env"
    path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        tok = path.read_text().strip()
    except OSError:
        return None, "none"
    return (tok, str(path)) if tok else (None, "none")


def write_replays(payload: dict, run_group: str, runs: Path) -> list[Path]:
    """`runs/<run_group>/<class>-<split>-x<rate_scale>.{rows,telemetry}.csv` per replay, plus
    `engine_env.json`. CSV, because replay_trace.py already writes its per-request rows that way
    beside the panel, and the loop re-slices CSV."""
    out = runs / run_group
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rep in payload.get("replays", []):
        stem = f"{rep['class']}-{rep['split']}-x{rep['rate_scale']:g}"
        for kind in ("rows", "telemetry_rows"):
            rows = rep.get(kind) or []
            path = out / f"{stem}.{kind.split('_')[0]}.csv"
            with open(path, "w", newline="") as f:
                if rows:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            written.append(path)
    meta = {"engine_env": payload.get("engine_env", {}), "hardware": payload.get("hardware"),
            "plan": payload.get("plan", [])}
    (out / "engine_env.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    written.append(out / "engine_env.json")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", help="repo-relative path, e.g. scripts/bench/x.py")
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe")
    ap.add_argument("--name", default=PodSpec.name,
                    help="pod name; CI uses ci-cuda-gate so its reaper never touches a human's pod")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be rented and exit without spending")
    args = ap.parse_args()

    env = provenance()
    # Passed through so the instrument configures the engine identically to modal_app.py.
    for k in ("BACKEND", "MODEL_NAME", "MAX_BATCH_SIZE", "PREFILL_MODE", "BENCH_RATES",
              "BENCH_DURATION", "BENCH_WARMUP", "BENCH_ARM", "BENCH_POOL_SIZE",
              "CUSTOM_BACKEND_COMPILE", "CUSTOM_BACKEND_BLOCKS",
              "CUSTOM_BACKEND_SLIDING_BLOCKS", "KV_CACHE_NUM_BLOCKS"):
        if k in os.environ:
            env[k] = os.environ[k]
    for k, v in os.environ.items():
        if k.startswith(("REPLAY_", "TELEMETRY_")):
            env[k] = v
    token, source = hf_token()
    if token:
        env["HF_TOKEN"] = token

    spec = PodSpec(gpu_type_ids=[args.gpu], name=args.name)
    print(f"[provenance] sha={env['RESEARCH_ENGINE_SHA']} "
          f"dirty={env['RESEARCH_ENGINE_DIRTY']} group={env['RESEARCH_RUN_GROUP']}")
    if env["RESEARCH_ENGINE_DIRTY"] == "1":
        print("[warn] working tree is dirty — this panel cannot be attributed to a clean commit")
    if token:
        print(f"[hf] token from {source}")
    else:
        print("[warn] no HF_TOKEN and no ~/.cache/huggingface/token — a gated model will fail "
              "to download on the pod")

    if args.dry_run:
        print(f"[dry-run] would rent {args.gpu} and run {args.instrument}")
        print(f"[dry-run] env: {', '.join(sorted(env))}")
        return 0

    try:
        payload = run_instrument(args.instrument, env, repo=str(REPO), spec=spec)
    except VenueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    # An instrument that could not run reports why and exits non-zero; no panel is evidence then.
    if "error" in payload:
        print(f"[error] instrument: {payload['error']}", file=sys.stderr)
        for line in payload.get("log_tail") or []:
            print(f"    {line}", file=sys.stderr)
        return 1

    from inference_server.research.schemas import Vitals
    runs = REPO / "runs"
    panels = payload.get("panels", [])
    for d in panels:
        v = Vitals.from_dict(d)
        v.to_json(runs / f"{v.validity.run_id}.json")
    print(f"wrote {len(panels)} panel(s) to runs/ (run_group={env['RESEARCH_RUN_GROUP']})")
    if payload.get("replays"):
        for p in write_replays(payload, env["RESEARCH_RUN_GROUP"], runs):
            print(f"  {p.relative_to(REPO)}")
        print(f"fit the timing model with: python scripts/tools/fit_timing_from_runs.py "
              f"{env['RESEARCH_RUN_GROUP']}")

    # A smoke run returns no panels on purpose; print what it did return so the run is readable.
    if "smoke" in payload:
        print(json.dumps(payload["smoke"], indent=2, sort_keys=True))
    # A gate returns a verdict, not panels; its verdict is this process's exit status.
    if "gate" in payload:
        print(json.dumps(payload["gate"], indent=2, sort_keys=True))
        return 0 if payload["gate"].get("passed") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
