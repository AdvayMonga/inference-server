"""Engine event timeline: one JSONL line per scheduler event, and a named profiler range per phase.

The step id is the join key. Every event carries the scheduler step it happened in, and every
phase range in a torch.profiler trace is named with the same step, so a kernel timing in the
trace sits inside the scheduler decision that caused it. Timestamps are wall clock so they line
up with the profiler trace and GPU samples taken by other tools. Off unless given a directory;
writes never touch the scheduler thread.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class Timeline:
    """Append-only `events.jsonl` in `directory`, drained by one background thread. `Timeline()` is off."""

    _STOP = object()

    def __init__(self, directory: str | Path | None = None, max_queue: int = 100_000):
        self.enabled = directory is not None
        self.step = 0
        self.events_written = 0
        self.events_dropped = 0
        if not self.enabled:
            return
        self.path = Path(directory) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, name="timeline-writer", daemon=True)
        self._thread.start()

    def begin_step(self) -> None:
        self.step += 1

    def event(self, kind: str, **fields: Any) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait({"ts": time.time(), "step": self.step, "kind": kind, **fields})
        except queue.Full:
            self.events_dropped += 1

    def phase(self, name: str, **labels: Any):
        """Wrap a block: a profiler range named `name step=N ...` plus one `phase` event with host duration."""
        if not self.enabled:
            return contextlib.nullcontext()
        return _Phase(self, name, labels)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "step": self.step,
                "events_written": self.events_written, "events_dropped": self.events_dropped}

    def close(self) -> None:
        """Flush everything queued and stop the writer. Idempotent."""
        if self.enabled and self._thread.is_alive():
            self._q.put(self._STOP)
            self._thread.join()

    def _drain(self) -> None:
        with open(self.path, "a") as f:
            while True:
                item = self._q.get()
                if item is self._STOP:
                    return
                try:
                    f.write(json.dumps(item) + "\n")
                    self.events_written += 1
                except Exception:
                    self.events_dropped += 1
                    logger.exception("timeline write failed for %s", self.path)


class _Phase:
    def __init__(self, tl: Timeline, name: str, labels: dict[str, Any]):
        self.tl, self.name, self.labels = tl, name, labels
        tag = " ".join(f"{k}={v}" for k, v in labels.items())
        self.range = torch.profiler.record_function(f"{name} step={tl.step} {tag}".rstrip())

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.range.__enter__()
        return self

    def __exit__(self, *exc):
        self.range.__exit__(*exc)
        self.tl.event("phase", name=self.name, dur_s=time.perf_counter() - self.t0,
                      ok=exc[0] is None, **self.labels)
        return False
