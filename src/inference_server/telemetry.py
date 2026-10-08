"""Per-request telemetry rows — the loop's senses.

Rows, not aggregates: an average cannot be re-sliced by the load it was measured under, so
every request gets one row with the conditions it arrived into, where its time went, and how
it ended. Off unless TELEMETRY_DIR is set; writes never touch the scheduler thread.
Clocks: `arrival_ts` is the only absolute time (epoch seconds); every `*_s` is a perf_counter duration.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1   # bump on any column change; also stored in the file's `meta` table
TERMINAL_STATES = ("ok", "rejected_429", "rejected_400", "expired", "preempted", "error",
                   "cancelled", "aborted")   # cancelled = client went away; aborted = engine shut down


@dataclass
class RequestRecord:
    """One row per request. Conditions are snapshotted at enqueue, before any work."""
    trace_id: str
    session_id: str
    turn_index: int
    # --- conditions at arrival ---
    arrival_ts: float               # epoch seconds at HTTP receipt, so rows align with a harness run
    pending_depth: int
    active_size: int                # rows holding a batch slot (decoding or mid-prefill)
    max_batch_size: int
    kv_free_blocks: int | None      # None when the backend has no cache adapter
    kv_free_frac: float | None
    concurrent_sessions: int        # distinct sessions in pending + active, excluding this one
    prompt_tokens: int
    replica_age_s: float            # seconds since scheduler start: cold start is a regime
    config_id: str | None = None    # applied policy config; no registry exists yet
    # --- spans: perf_counter deltas between boundaries (arrival = HTTP receipt, before tokenizing) ---
    queue_wait_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None
    decode_steps: int = 0
    # --- decode-time batch width: measured WHILE this row decoded, not at arrival ---
    # `active_size` above is the width this request ARRIVED into. These are the widths it
    # actually shared a forward pass with. Conflating the two is what flattened the simulator's
    # fitted decode slope by ~40x (kb-20260917-c07eb94b), so the names keep them apart.
    decode_batch_width_mean: float | None = None   # sum(width) / steps, over its OWN steps
    decode_batch_width_max: int = 0                # widest step it decoded in
    # --- counters ---
    cache_hit_tokens: int | None = None    # None = no prefix-cache lookup reported (not admitted, or unknown)
    prefill_mode: str | None = None         # monolithic | chunked | batched; None = never admitted
    preempted: int = 0                      # 0/1: preemption is terminal here (no resume)
    # --- outcome ---
    ttft_s: float | None = None
    tpot_s: float | None = None
    total_s: float | None = None
    tokens_out: int = 0
    terminal_state: str = ""
    schema_version: int = SCHEMA_VERSION

    def finish(self, state: str, *, enqueue_ts: float, admit_ts: float, first_token_ts: float,
               end_ts: float, tokens_out: int, cache_hit_tokens: int | None,
               decode_width_sum: int = 0, decode_width_steps: int = 0,
               decode_width_max: int = 0, arrival_mono: float = 0.0,
               prefill_mode: str | None = None) -> None:
        """Fill spans and outcome from the scheduler's perf_counter stamps (0.0 = never reached).

        The three `decode_width_*` arguments are the scheduler's running accumulators for this
        request; they are divided here rather than on the hot path.
        """
        self.terminal_state = state
        self.tokens_out = tokens_out
        self.cache_hit_tokens = cache_hit_tokens
        self.prefill_mode = prefill_mode
        self.preempted = int(state == "preempted")
        self.decode_batch_width_max = decode_width_max
        if decode_width_steps:
            # Own denominator, not decode_steps: a row evicted on EOS decoded one more step
            # than it emitted tokens, and the mean must match the steps it was summed over.
            self.decode_batch_width_mean = decode_width_sum / decode_width_steps
        arrival = arrival_mono or enqueue_ts
        self.total_s = end_ts - arrival
        self.decode_steps = decode_width_steps             # forward passes, incl. one that hit EOS
        if admit_ts:
            self.queue_wait_s = admit_ts - enqueue_ts
        if first_token_ts:
            self.prefill_s = first_token_ts - (admit_ts or enqueue_ts)
            if tokens_out:                                 # an EOS-first answer has no real token
                self.ttft_s = first_token_ts - arrival
                self.decode_s = end_ts - first_token_ts
                if tokens_out > 1:                         # TPOT = (E2E - TTFT) / (n - 1)
                    self.tpot_s = self.decode_s / (tokens_out - 1)


_SQL_TYPES = {"int": "INTEGER", "float": "REAL", "str": "TEXT"}
_COLUMNS = [f.name for f in fields(RequestRecord)]
_SCHEMA = ", ".join(f"{f.name} {_SQL_TYPES[f.type.replace(' | None', '')]}"
                    for f in fields(RequestRecord))


class RowStore:
    """SQLite sink: one file per run, drained by one background thread.

    The producer side only ever put_nowait()s and counts drops — it never blocks the scheduler.
    A `meta` table (key, value) carries schema_version, run_id, rows_written, rows_dropped and
    closed (1 once counts are final), updated after every batch.
    """

    _STOP = object()

    def __init__(self, directory: str | Path, run_id: str | None = None,
                 max_queue: int = 10_000):
        self.run_id = run_id or f"run-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
        self.path = Path(directory) / f"{self.run_id}.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows_written = 0
        self.rows_dropped = 0
        self._closed = False
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, name="telemetry-writer", daemon=True)
        self._thread.start()

    def put(self, record: RequestRecord) -> None:
        if self._closed:            # nobody drains after close: count it, don't lose it silently
            self.rows_dropped += 1
            return
        try:
            self._q.put_nowait(record)
        except queue.Full:
            self.rows_dropped += 1

    def stats(self) -> dict:
        return {"enabled": True, "run_id": self.run_id,
                "rows_written": self.rows_written, "rows_dropped": self.rows_dropped}

    def close(self) -> None:
        """Flush everything queued and stop the writer. Idempotent."""
        self._closed = True
        if self._thread.is_alive():
            self._q.put(self._STOP)
            self._thread.join()

    def _drain(self) -> None:
        conn = sqlite3.connect(self.path)   # sqlite3 connections are thread-bound: open it here
        conn.execute(f"CREATE TABLE IF NOT EXISTS requests ({_SCHEMA})")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value)")
        insert = f"INSERT INTO requests VALUES ({', '.join('?' * len(_COLUMNS))})"
        failed_before = False
        try:
            while True:
                batch = [self._q.get()]
                while len(batch) < 256:
                    try:
                        batch.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                items = [r for r in batch if r is not self._STOP]
                # A bad batch must not kill the writer: log once, count it, keep draining.
                try:
                    if items:
                        conn.executemany(insert, [tuple(getattr(r, c) for c in _COLUMNS)
                                                  for r in items])
                        conn.commit()
                        self.rows_written += len(items)
                except Exception:
                    self.rows_dropped += len(items)
                    if not failed_before:
                        failed_before = True
                        logger.exception("telemetry write failed for %s; dropping batches",
                                         self.path)
                stop = any(r is self._STOP for r in batch)
                self._write_meta(conn, closed=stop)
                if stop:
                    return
        finally:
            conn.close()

    def _write_meta(self, conn: sqlite3.Connection, closed: bool) -> None:
        try:
            conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [
                ("schema_version", SCHEMA_VERSION), ("run_id", self.run_id),
                ("rows_written", self.rows_written), ("rows_dropped", self.rows_dropped),
                ("closed", int(closed))])
            conn.commit()
        except Exception:
            logger.exception("telemetry meta write failed for %s", self.path)
