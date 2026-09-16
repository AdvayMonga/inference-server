"""Per-request telemetry rows — the loop's senses.

Rows, not aggregates: an average cannot be re-sliced by the load it was measured under, so
every request gets one row with the conditions it arrived into, where its time went, and how
it ended. Off unless TELEMETRY_DIR is set; writes never touch the scheduler thread.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, fields
from pathlib import Path

TERMINAL_STATES = ("ok", "rejected_429", "expired", "preempted", "error")


@dataclass
class RequestRecord:
    """One row per request. Conditions are snapshotted at enqueue, before any work."""
    trace_id: str
    session_id: str
    turn_index: int
    # --- conditions at arrival ---
    arrival_ts: float               # wall clock, so rows align with a harness run
    pending_depth: int
    active_size: int                # rows holding a batch slot (decoding or mid-prefill)
    max_batch_size: int
    kv_free_blocks: int | None      # None when the backend has no cache adapter
    kv_free_frac: float | None
    concurrent_sessions: int        # distinct sessions in pending + active, excluding this one
    prompt_tokens: int
    replica_age_s: float            # seconds since scheduler start: cold start is a regime
    config_id: str | None = None    # applied policy config; no registry exists yet
    # --- spans: perf_counter deltas between scheduler boundaries ---
    queue_wait_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None
    decode_steps: int = 0
    # --- counters ---
    cache_hit_tokens: int = 0
    preempted: int = 0
    # --- outcome ---
    ttft_s: float | None = None
    tpot_s: float | None = None
    total_s: float | None = None
    tokens_out: int = 0
    terminal_state: str = ""

    def finish(self, state: str, *, enqueue_ts: float, admit_ts: float, first_token_ts: float,
               end_ts: float, tokens_out: int, cache_hit_tokens: int) -> None:
        """Fill spans and outcome from the scheduler's timestamps (0.0 = boundary never reached)."""
        self.terminal_state = state
        self.tokens_out = tokens_out
        self.cache_hit_tokens = cache_hit_tokens
        self.preempted = int(state == "preempted")
        self.total_s = end_ts - enqueue_ts
        if admit_ts:
            self.queue_wait_s = admit_ts - enqueue_ts
        if first_token_ts:
            self.ttft_s = first_token_ts - enqueue_ts
            self.prefill_s = first_token_ts - (admit_ts or enqueue_ts)
            self.decode_s = end_ts - first_token_ts
            self.decode_steps = max(tokens_out - 1, 0)   # first token comes from prefill
            if tokens_out > 1:
                self.tpot_s = self.decode_s / (tokens_out - 1)


_SQL_TYPES = {"int": "INTEGER", "float": "REAL", "str": "TEXT"}
_COLUMNS = [f.name for f in fields(RequestRecord)]
_SCHEMA = ", ".join(f"{f.name} {_SQL_TYPES[f.type.replace(' | None', '')]}"
                    for f in fields(RequestRecord))


class RowStore:
    """SQLite sink: one file per run, drained by one background thread.

    The producer side only ever put_nowait()s and counts drops — it never blocks the scheduler.
    """

    _STOP = object()

    def __init__(self, directory: str | Path, run_id: str | None = None,
                 max_queue: int = 10_000):
        self.run_id = run_id or f"run-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        self.path = Path(directory) / f"{self.run_id}.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows_written = 0
        self.rows_dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, name="telemetry-writer", daemon=True)
        self._thread.start()

    def put(self, record: RequestRecord) -> None:
        try:
            self._q.put_nowait(record)
        except queue.Full:
            self.rows_dropped += 1

    def stats(self) -> dict:
        return {"enabled": True, "run_id": self.run_id,
                "rows_written": self.rows_written, "rows_dropped": self.rows_dropped}

    def close(self) -> None:
        """Flush everything queued and stop the writer. Idempotent."""
        if self._thread.is_alive():
            self._q.put(self._STOP)
            self._thread.join()

    def _drain(self) -> None:
        conn = sqlite3.connect(self.path)   # sqlite3 connections are thread-bound: open it here
        conn.execute(f"CREATE TABLE IF NOT EXISTS requests ({_SCHEMA})")
        insert = f"INSERT INTO requests VALUES ({', '.join('?' * len(_COLUMNS))})"
        try:
            while True:
                batch = [self._q.get()]
                while len(batch) < 256:
                    try:
                        batch.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                rows = [tuple(getattr(r, c) for c in _COLUMNS)
                        for r in batch if r is not self._STOP]
                if rows:
                    conn.executemany(insert, rows)
                    conn.commit()
                    self.rows_written += len(rows)
                if any(r is self._STOP for r in batch):
                    return
        finally:
            conn.close()
