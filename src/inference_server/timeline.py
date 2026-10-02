"""Engine event timeline: one JSONL event per scheduler decision and a profiler range per phase, keyed by step id."""

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
    """Appends `events.jsonl` in `directory` from one background thread. `Timeline()` is off and a no-op."""

    _STOP = object()

    def __init__(self, directory: str | Path | None = None, max_queue: int = 100_000):
        self.enabled = directory is not None
        self.step = 0
        self.events_written = 0
        self.events_dropped = 0
        self._closed = False
        if not self.enabled:
            return
        self.path = Path(directory) / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")          # fails here, at construction, not in the writer
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, name="timeline-writer", daemon=True)
        self._thread.start()

    def begin_step(self) -> None:
        self.step += 1

    def event(self, kind: str, **fields: Any) -> None:
        """Wall-clock timestamp, so rows line up with profiler traces and GPU samples from other tools."""
        if not self.enabled:
            return
        if self._closed:
            self.events_dropped += 1
            return
        try:
            self._q.put_nowait({"ts": time.time(), "step": self.step, "kind": kind, **fields})
        except queue.Full:
            self.events_dropped += 1

    def phase(self, name: str, **labels: Any):
        """A profiler range named `name step=N ...` around the block, plus one `phase` event with host duration."""
        if not self.enabled or self._closed:
            return contextlib.nullcontext()
        return _Phase(self, name, labels)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "step": self.step,
                "events_written": self.events_written, "events_dropped": self.events_dropped}

    def close(self) -> None:
        """Flush everything queued and stop the writer. Idempotent; later events are counted as dropped."""
        if not self.enabled or self._closed:
            return
        self._closed = True
        self._q.put(self._STOP)
        self._thread.join()

    def _drain(self) -> None:
        failed_before = False
        with self._file as f:
            while True:
                item = self._q.get()
                if item is self._STOP:
                    return
                try:
                    f.write(json.dumps(item) + "\n")
                    self.events_written += 1
                except Exception:
                    self.events_dropped += 1
                    if not failed_before:
                        failed_before = True
                        logger.exception("timeline write failed for %s; dropping", self.path)


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
