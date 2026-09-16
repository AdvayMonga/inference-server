#!/usr/bin/env python
"""Terminate pods this repo named and then lost. The CI gpu lane's `if: always()` step.

`run_instrument` terminates its pod in a `finally`, but a runner killed by `timeout-minutes`
never reaches it, and a pod bills until someone finds it in a web console. This asks the API
for every pod, keeps the ones carrying our name (`PodSpec.name`), optionally only those started
at or after `--since`, and terminates them. It never touches a pod it did not name.

    scripts/tools/runpod_reap.py --since 2026-09-16T07:00:00Z
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference_server.research.venues import (  # noqa: E402
    API_KEY_ENV,
    PodSpec,
    RunPodClient,
    VenueError,
    reap_pods,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=PodSpec.name, help="only pods with exactly this name")
    ap.add_argument("--since", default=None,
                    help="ISO-8601 UTC; only pods started at or after this (e.g. the job start)")
    args = ap.parse_args()

    if not os.environ.get(API_KEY_ENV):
        print(f"[reap] no {API_KEY_ENV} in the environment — nothing was rented, nothing to reap")
        return 0
    try:
        reaped = reap_pods(RunPodClient(), name=args.name, since=args.since)
    except VenueError as e:
        print(f"[reap] {e} — check runpod.io/console/pods by hand", file=sys.stderr)
        return 1
    print(f"[reap] {len(reaped)} pod(s) terminated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
