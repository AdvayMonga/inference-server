"""NVIDIA GPU state and samples through nvidia-smi. Everything here returns None where there is no nvidia-smi."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

QUERY = ("name", "driver_version", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
         "power.limit", "power.draw", "temperature.gpu", "utilization.gpu", "utilization.memory",
         "memory.used", "memory.total")
SAMPLE = ("timestamp", "utilization.gpu", "utilization.memory", "clocks.sm", "clocks.mem",
          "power.draw", "temperature.gpu", "memory.used")


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query() -> dict | None:
    """One snapshot of the first GPU: identity, clocks and limits. The clock state that travels with a number."""
    if not available():
        return None
    out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(QUERY)}", "--format=csv,noheader,nounits",
                          "-i", "0"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return dict(zip(QUERY, [v.strip() for v in out.stdout.strip().split(",")]))


class Sampler:
    """nvidia-smi sampling into a CSV at `interval_ms`, as a subprocess so it costs the engine nothing."""

    def __init__(self, path: Path, interval_ms: int = 100):
        self.path, self.interval_ms, self._proc = Path(path), interval_ms, None

    def start(self) -> bool:
        if not available():
            return False
        self._file = open(self.path, "w")
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(SAMPLE)}", "--format=csv,nounits",
             "-lms", str(self.interval_ms), "-i", "0"], stdout=self._file, stderr=subprocess.DEVNULL)
        return True

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        self._proc.wait(timeout=5)
        self._file.close()
        self._proc = None
