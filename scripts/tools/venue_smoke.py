#!/usr/bin/env python
"""Prove a rented venue can run our code, before spending an A100 hour finding out it cannot.

The first instrument that speaks the venue contract. It deliberately emits NO Vitals panel: a
smoke run measures the transport, not the engine, and a panel that looks like a measurement but
is not one is the kind of thing this repo has been burned by.

    scripts/tools/run_on_runpod.py scripts/tools/venue_smoke.py --gpu 'NVIDIA GeForce RTX 4090'

What it answers, in the order the answers stop being useful:

  1. did rsync land the tree            -> imports resolve
  2. did pip install run                -> transformers/fastapi are importable
  3. is there actually a GPU            -> torch.cuda sees a device
  4. does provenance survive the hop    -> the launcher's env vars are readable here
  5. does the payload round-trip        -> the marked block parses on the other side
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference_server.research.venues import emit_payload  # noqa: E402


def probe_imports() -> dict[str, str]:
    """Import what an instrument needs. A missing one means provisioning silently half-worked."""
    found: dict[str, str] = {}
    for mod in ("torch", "transformers", "fastapi", "httpx", "prometheus_client",
                "inference_server.server"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            found[mod] = getattr(m, "__version__", "ok")
        except Exception as e:                       # noqa: BLE001 — report, never abort
            found[mod] = f"MISSING: {type(e).__name__}: {e}"
    return found


def probe_gpu() -> dict[str, object]:
    import torch
    if not torch.cuda.is_available():
        return {"available": False, "why": "torch.cuda.is_available() is False"}
    dev = torch.cuda.get_device_properties(0)
    # A matmul, not just a device query: a pod can expose a GPU whose driver cannot run kernels.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()
    tflops = 10 * 2 * 4096 ** 3 / (time.perf_counter() - t0) / 1e12
    return {
        "available": True,
        "name": dev.name,
        "vram_gb": round(dev.total_memory / 1e9, 1),
        "capability": f"{dev.major}.{dev.minor}",
        "bf16_matmul_tflops": round(tflops, 1),
    }


def main() -> int:
    payload = {
        "panels": [],                                # a smoke run measures no engine behaviour
        "smoke": {
            "host": platform.node(),
            "python": platform.python_version(),
            "cwd": os.getcwd(),
            "imports": probe_imports(),
            "provenance": {k: v for k, v in os.environ.items()
                           if k.startswith("RESEARCH_")},
        },
    }
    try:
        payload["smoke"]["gpu"] = probe_gpu()
    except Exception as e:                           # noqa: BLE001 — a dead GPU is the result
        payload["smoke"]["gpu"] = {"available": False, "why": f"{type(e).__name__}: {e}"}

    print(emit_payload(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
