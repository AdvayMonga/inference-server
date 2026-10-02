"""One directory per profile run, with the raw outputs of every layer and the provenance they were taken under.

    events.jsonl   engine event timeline (scheduler decisions, phases, per step)
    trace.json     torch.profiler chrome trace; phase ranges carry the step id
    memory.json    device allocator stats and peak host RSS
    gpu.csv        nvidia-smi samples at 100ms (CUDA hosts only)
    stats.json     the scheduler's own counters at the end of the run
    meta.json      git sha, torch, device, clock state, workload hash, wall clock window
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from lab import gpu

FILES = ("events.jsonl", "trace.json", "memory.json", "stats.json", "meta.json")


def new_dir(root: str | Path) -> Path:
    d = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))


def git_sha() -> str | None:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def workload_hash(prompts: list[list[int]], max_tokens: int) -> str:
    h = hashlib.sha256(json.dumps([prompts, max_tokens]).encode())
    return h.hexdigest()[:16]


def device_memory(device: str) -> dict:
    """Allocator view for the device the engine ran on, plus peak host RSS."""
    out: dict[str, Any] = {"device": device,
                           "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                           * (1 if sys.platform == "darwin" else 1024)}
    if device.startswith("cuda") and torch.cuda.is_available():
        out["cuda"] = torch.cuda.memory_stats()
    elif device == "mps" and torch.backends.mps.is_available():
        out["mps"] = {"current_allocated": torch.mps.current_allocated_memory(),
                      "driver_allocated": torch.mps.driver_allocated_memory()}
    return out


def meta(device: str, prompts: list[list[int]], max_tokens: int, t0: float, t1: float,
         **extra: Any) -> dict:
    return {"git_sha": git_sha(), "torch": torch.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "device": device, "gpu": gpu.query(),
            "workload_hash": workload_hash(prompts, max_tokens), "requests": len(prompts),
            "max_tokens": max_tokens, "window": {"start": t0, "end": t1, "wall_s": t1 - t0},
            **extra}
