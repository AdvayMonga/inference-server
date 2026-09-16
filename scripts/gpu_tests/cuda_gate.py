#!/usr/bin/env python
"""The CUDA correctness gate as one venue instrument: every check in `checks.py`, on one box.

What the CI gpu lane runs on a rented pod, and what you run by hand on any CUDA machine:

    scripts/tools/run_on_runpod.py scripts/gpu_tests/cuda_gate.py --gpu 'NVIDIA GeForce RTX 4090'
    PYTHONPATH=src python scripts/gpu_tests/cuda_gate.py          # directly, on a CUDA box

It emits NO Vitals panel, for the same reason venue_smoke.py does not: a gate is a verdict,
not a measurement. The payload is `{"gate": {passed, checks, gpu, torch, triton}}`, and the
exit status is the verdict so a direct run needs no parser. With no CUDA device the checks
are reported as skipped and the gate fails — a gate that cannot run has not passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

import checks  # noqa: E402
from inference_server.research.venues import emit_payload  # noqa: E402


def environment() -> dict[str, str | None]:
    """What ran the checks. `gpu` is None when torch sees no CUDA device."""
    info: dict[str, str | None] = {"gpu": None, "torch": None, "triton": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:                           # noqa: BLE001 — report, never abort
        info["torch"] = f"MISSING: {type(e).__name__}: {e}"
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as e:                           # noqa: BLE001
        info["triton"] = f"MISSING: {type(e).__name__}: {e}"
    return info


def run_checks(cuda: bool) -> list[dict[str, object]]:
    """Run every check, or report each as skipped when there is no CUDA. Never raises."""
    results: list[dict[str, object]] = []
    for name, fn in checks.CHECKS.items():
        if not cuda:
            results.append({"name": name, "passed": False, "detail": "skipped: no CUDA device"})
            continue
        try:
            passed, detail = fn()
        except Exception as e:                       # noqa: BLE001 — a crash is a failed check
            passed, detail = False, f"{type(e).__name__}: {e}"
        results.append({"name": name, "passed": bool(passed), "detail": detail})
    return results


def main() -> int:
    info = environment()
    results = run_checks(cuda=info["gpu"] is not None)
    gate = {"passed": all(r["passed"] for r in results), "checks": results, **info}

    for r in results:
        print(f"[gate] {'ok  ' if r['passed'] else 'FAIL'} {r['name']}: {r['detail']}")
    if info["gpu"] is None:
        print("[gate] no CUDA device — every check skipped; the gate cannot pass here")
    print(f"[gate] {'PASSED' if gate['passed'] else 'FAILED'} on gpu={info['gpu']} "
          f"torch={info['torch']} triton={info['triton']}")

    print(emit_payload({"panels": [], "gate": gate}))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
