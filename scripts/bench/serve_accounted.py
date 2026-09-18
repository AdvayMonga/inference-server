#!/usr/bin/env python
"""Run the engine's uvicorn server and write what the PROCESS cost to a sidecar JSON at exit.

    ACCOUNTING_SIDECAR=/tmp/run.json python scripts/bench/serve_accounted.py --port 8000

Exists because of one seam: the device memory and the resident set that matter belong to the
SERVER process, and the instrument measuring it is a different process. `replay_local.py`
launches this instead of `python -m uvicorn` so those two numbers come from inside the process
that holds them, while `research/` stays free of torch (see research/accounting.py).

Everything else about the server is unchanged — same app, same env, same SIGTERM path, so the
lifespan shutdown still flushes telemetry. Without `ACCOUNTING_SIDECAR` this is exactly
`python -m uvicorn inference_server.server:app`.

Device memory on MPS is SAMPLED, not a high-water mark: torch exposes
`current_allocated_memory()` and `driver_allocated_memory()` on MPS but no peak counter, so a
spike between two samples is missed and the number is a lower bound. CUDA has a real peak
(`max_memory_allocated`) and is read once at exit. The sidecar records which it was, so nobody
reads a sampled floor as a measured peak.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SAMPLE_INTERVAL_S = 0.25


class DeviceMemorySampler:
    """Peak device memory for this process: exact on CUDA, sampled on MPS, absent elsewhere."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.peak_bytes = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if torch.cuda.is_available():
            self.kind, self.source = "cuda", "torch.cuda.max_memory_allocated (true peak)"
        elif torch.backends.mps.is_available():
            self.kind = "mps"
            self.source = (f"torch.mps.driver_allocated_memory sampled every "
                           f"{SAMPLE_INTERVAL_S}s (a lower bound: MPS exposes no peak counter)")
        else:
            self.kind, self.source = "", ""

    def start(self) -> None:
        if self.kind != "mps":
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL_S):
            try:
                self.peak_bytes = max(self.peak_bytes,
                                      self.torch.mps.driver_allocated_memory())
            except Exception:          # noqa: BLE001 — accounting must never kill the server
                return

    def result(self) -> tuple[float | None, str]:
        """(peak GB, how it was measured). (None, "") when this box has no device."""
        if self.kind == "cuda":
            return round(self.torch.cuda.max_memory_allocated() / 1e9, 3), self.source
        if self.kind == "mps":
            self._stop.set()
            try:
                self.peak_bytes = max(self.peak_bytes,
                                      self.torch.mps.driver_allocated_memory())
            except Exception:          # noqa: BLE001
                pass
            return round(self.peak_bytes / 1e9, 3), self.source
        return None, ""


def write_sidecar(path: Path, sampler: DeviceMemorySampler, started: float) -> None:
    """The three terms only this process can see: its RSS, its device, its file reads.

    ru_maxrss is BYTES on Darwin and KIBIBYTES on Linux — a 1024x error the other way.
    """
    from inference_server.research.accounting import storage_read_bytes

    peak_gb, source = sampler.result()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    read_bytes, read_note = storage_read_bytes()
    path.write_text(json.dumps({
        "peak_host_rss_gb": round(rss / (1e9 if sys.platform == "darwin" else 1e6), 3),
        "peak_device_mem_gb": peak_gb,
        "device_mem_source": source or None,
        "storage_read_bytes": read_bytes,
        "storage_read_note": read_note or None,
        "server_uptime_s": round(time.monotonic() - started, 3),
    }, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    import uvicorn

    started = time.monotonic()
    sampler = DeviceMemorySampler()
    sampler.start()
    try:
        uvicorn.run("inference_server.server:app", host=args.host, port=args.port,
                    log_level=args.log_level)
    finally:
        sidecar = os.environ.get("ACCOUNTING_SIDECAR")
        if sidecar:
            write_sidecar(Path(sidecar), sampler, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
