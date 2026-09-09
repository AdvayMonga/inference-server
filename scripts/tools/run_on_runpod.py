#!/usr/bin/env python
"""Launch an instrument on a rented RunPod GPU. The RunPod counterpart of run_instrument.sh.

Same provenance contract as the Modal path, for the same reason: the box has no git repo, so the
engine sha and the run_group must be stamped by the launcher or the panel cannot be attributed.

    scripts/tools/run_on_runpod.py scripts/bench/bench_serving_runpod.py
    RESEARCH_RUN_GROUP=exp-42 scripts/tools/run_on_runpod.py <instrument>   # arm A, then arm B

Arms of one experiment MUST share RESEARCH_RUN_GROUP, and — the lesson that cost us an A100 day —
should share one POD. Two pods are two machines, and two identical A100-80GB draws on Modal
differed by 2.31x on byte-identical config. Prefer an instrument that runs both arms in-process.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("instrument", help="repo-relative path, e.g. scripts/bench/x.py")
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be rented and exit without spending")
    args = ap.parse_args()

    env = provenance()
    # Passed through so the instrument configures the engine identically to modal_app.py.
    for k in ("BACKEND", "MODEL_NAME", "MAX_BATCH_SIZE", "PREFILL_MODE", "BENCH_RATES",
              "BENCH_DURATION", "BENCH_WARMUP", "BENCH_ARM", "BENCH_POOL_SIZE",
              "CUSTOM_BACKEND_COMPILE", "CUSTOM_BACKEND_BLOCKS",
              "CUSTOM_BACKEND_SLIDING_BLOCKS", "KV_CACHE_NUM_BLOCKS", "HF_TOKEN"):
        if k in os.environ:
            env[k] = os.environ[k]

    spec = PodSpec(gpu_type_ids=[args.gpu])
    print(f"[provenance] sha={env['RESEARCH_ENGINE_SHA']} "
          f"dirty={env['RESEARCH_ENGINE_DIRTY']} group={env['RESEARCH_RUN_GROUP']}")
    if env["RESEARCH_ENGINE_DIRTY"] == "1":
        print("[warn] working tree is dirty — this panel cannot be attributed to a clean commit")
    if "HF_TOKEN" not in env:
        print("[warn] no HF_TOKEN — a gated model will fail to download on the pod")

    if args.dry_run:
        print(f"[dry-run] would rent {args.gpu} and run {args.instrument}")
        print(f"[dry-run] env: {', '.join(sorted(env))}")
        return 0

    try:
        payload = run_instrument(args.instrument, env, repo=str(REPO), spec=spec)
    except VenueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    from inference_server.research.schemas import Vitals
    runs = REPO / "runs"
    panels = payload.get("panels", [])
    for d in panels:
        v = Vitals.from_dict(d)
        v.to_json(runs / f"{v.validity.run_id}.json")
    print(f"wrote {len(panels)} panel(s) to runs/ (run_group={env['RESEARCH_RUN_GROUP']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
