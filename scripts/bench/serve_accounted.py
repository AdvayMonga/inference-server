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
(`max_memory_allocated`). The sidecar records which it was, so nobody reads a sampled floor as
a measured peak.

**The sidecar is rewritten every second, not only at exit.** Measured on this box: the engine's
lifespan shutdown can outlast `stop_server`'s 60s SIGTERM grace, and the SIGKILL that follows
runs no `finally`. An exit-only sidecar is therefore missing exactly when a run is slow, which
is the run whose cost most needs accounting. A periodically-rewritten one is at most a second
stale, and the instrument names any gap rather than assuming zero either way.
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
WRITE_EVERY_N_SAMPLES = 4          # a sidecar at most ~1s stale, even if the process is killed


class ResourceRecorder:
    """This process's own resource high-water marks, rewritten to the sidecar as it runs."""

    def __init__(self, sidecar: Path) -> None:
        import torch

        self.torch = torch
        self.sidecar = sidecar
        self.started = time.monotonic()
        self.peak_bytes = 0.0
        self._stop = threading.Event()
        if torch.cuda.is_available():
            self.kind, self.source = "cuda", "torch.cuda.max_memory_allocated (true peak)"
        elif torch.backends.mps.is_available():
            self.kind = "mps"
            self.source = (f"torch.mps.driver_allocated_memory sampled every "
                           f"{SAMPLE_INTERVAL_S}s (a lower bound: MPS exposes no peak counter)")
        else:
            self.kind, self.source = "", ""

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        """Last sample, last write. Only reached on a clean shutdown — hence the periodic one."""
        self._stop.set()
        self._sample()
        self.write()

    def _loop(self) -> None:
        n = 0
        while not self._stop.wait(SAMPLE_INTERVAL_S):
            try:
                self._sample()
                n += 1
                if n % WRITE_EVERY_N_SAMPLES == 0:
                    self.write()
            except Exception:          # noqa: BLE001 — accounting must never kill the server
                return

    def _sample(self) -> None:
        if self.kind == "cuda":
            self.peak_bytes = max(self.peak_bytes, self.torch.cuda.max_memory_allocated())
        elif self.kind == "mps":
            self.peak_bytes = max(self.peak_bytes, self.torch.mps.driver_allocated_memory())

    def device_peak_gb(self) -> float | None:
        return round(self.peak_bytes / 1e9, 3) if self.kind else None

    def write(self) -> None:
        """The three terms only this process can see: its RSS, its device, its file reads.

        ru_maxrss is BYTES on Darwin and KIBIBYTES on Linux — a 1024x error the other way.
        Written via a temp file and renamed, so a reader never sees half a JSON document.
        """
        from inference_server.research.accounting import storage_read_bytes

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        read_bytes, read_note = storage_read_bytes()
        tmp = self.sidecar.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "peak_host_rss_gb": round(rss / (1e9 if sys.platform == "darwin" else 1e6), 3),
            "peak_device_mem_gb": self.device_peak_gb(),
            "device_mem_source": self.source or None,
            "storage_read_bytes": read_bytes,
            "storage_read_note": read_note or None,
            "server_uptime_s": round(time.monotonic() - self.started, 3),
        }, indent=2, sort_keys=True))
        tmp.replace(self.sidecar)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    import uvicorn

    sidecar = os.environ.get("ACCOUNTING_SIDECAR")
    recorder = ResourceRecorder(Path(sidecar)) if sidecar else None
    if recorder:
        recorder.start()
    try:
        uvicorn.run("inference_server.server:app", host=args.host, port=args.port,
                    log_level=args.log_level)
    finally:
        if recorder:
            recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
