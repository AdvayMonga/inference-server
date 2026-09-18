"""Per-request telemetry rows (telemetry.py + the scheduler's emission points).

Fast: stub backend, no model. Covers every terminal state, the SQLite round-trip, off-by-default,
and the overhead budget the plan says to treat as a correctness failure.
"""

from __future__ import annotations

import asyncio
import queue
import sqlite3
import time
from dataclasses import asdict

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, QueueFullError, ScheduledRequest
from inference_server.telemetry import TERMINAL_STATES, RequestRecord, RowStore
from tests.test_chunked_prefill import FakeChunkBackend


def _rows(store: RowStore) -> list[dict]:
    store.close()
    conn = sqlite3.connect(store.path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY arrival_ts")]
    finally:
        conn.close()


def _req(loop, **kw):
    kw.setdefault("token_ids", [1, 2, 3, 4])
    kw.setdefault("max_tokens", 3)
    kw.setdefault("session_id", "s")
    return ScheduledRequest(future=loop.create_future(), **kw)


# --------------------------------------------------------------------------- terminal states

@pytest.mark.asyncio
async def test_ok_row_has_conditions_spans_and_outcome(tmp_path):
    store = RowStore(tmp_path)
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=4, telemetry=store)
    sched.start()
    loop = asyncio.get_running_loop()
    req = _req(loop, trace_id="abc123", turn_index=2, session_id="u1")
    out = await sched.submit(req)
    await sched.stop()          # closes the store

    (row,) = _rows(store)
    assert row["terminal_state"] == "ok"
    assert row["trace_id"] == "abc123" and row["session_id"] == "u1" and row["turn_index"] == 2
    # conditions: first request into an idle scheduler
    assert row["pending_depth"] == 0 and row["active_size"] == 0 and row["max_batch_size"] == 4
    assert row["concurrent_sessions"] == 0 and row["prompt_tokens"] == 4
    assert row["kv_free_blocks"] is None and row["kv_free_frac"] is None
    assert 0 <= row["replica_age_s"] < 5
    # spans and outcome
    assert row["tokens_out"] == len(out) == 3
    assert row["queue_wait_s"] >= 0 and row["prefill_s"] >= 0 and row["decode_s"] >= 0
    assert row["ttft_s"] > 0 and row["total_s"] >= row["ttft_s"]
    assert row["tpot_s"] > 0 and row["decode_steps"] == 2
    # decode-time width: this row decoded alone, so both are 1 while active_size stays 0
    assert row["decode_batch_width_mean"] == 1.0 and row["decode_batch_width_max"] == 1
    assert row["preempted"] == 0
    assert sched.stats()["telemetry"] == {
        "enabled": True, "run_id": store.run_id, "rows_written": 1, "rows_dropped": 0}


def test_429_at_enqueue_still_emits_a_row_with_the_load_it_saw(tmp_path):
    """The whole point of snapshotting in enqueue(): 'what was the load when we rejected'."""
    store = RowStore(tmp_path)
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_queue_size=1, telemetry=store)
    loop = asyncio.new_event_loop()
    try:
        sched.enqueue(_req(loop, session_id="a"))
        with pytest.raises(QueueFullError):
            sched.enqueue(_req(loop, session_id="b"))
    finally:
        loop.close()
    rows = _rows(store)
    assert [r["terminal_state"] for r in rows] == ["rejected_429"]
    assert rows[0]["pending_depth"] == 1 and rows[0]["concurrent_sessions"] == 1
    assert rows[0]["queue_wait_s"] is None and rows[0]["ttft_s"] is None
    assert rows[0]["total_s"] >= 0


def test_expired_row(tmp_path):
    store = RowStore(tmp_path)
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_queue_wait_s=0.05, telemetry=store)
    loop = asyncio.new_event_loop()
    try:
        req = _req(loop)
        sched.enqueue(req)
        req.enqueue_ts = time.perf_counter() - 10.0
        sched._shed_expired()
    finally:
        loop.close()
    (row,) = _rows(store)
    assert row["terminal_state"] == "expired"
    assert row["total_s"] >= 10.0 and row["queue_wait_s"] is None


class _PreemptOnceBackend(FakeChunkBackend):
    """First decode step over a 2-row batch fails, so the scheduler preempts the newest row."""

    def __init__(self):
        super().__init__()
        self.raised = False

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids,
                            sampling_per_row=None):
        if current_tokens.shape[0] > 1 and not self.raised:
            self.raised = True
            raise RuntimeError("out of KV blocks")
        return super().decode_step_batched(current_tokens, batched_kv, attention_mask,
                                           position_ids, sampling_per_row)


@pytest.mark.asyncio
async def test_preempted_row(tmp_path):
    store = RowStore(tmp_path)
    sched = ContinuousBatchScheduler(_PreemptOnceBackend(), max_batch_size=2, telemetry=store)
    sched.start()
    loop = asyncio.get_running_loop()
    a, b = _req(loop, session_id="old"), _req(loop, session_id="new")
    results = await asyncio.gather(sched.submit(a), sched.submit(b), return_exceptions=True)
    await sched.stop()

    rows = {r["session_id"]: r for r in _rows(store)}
    assert set(rows) == {"old", "new"}
    assert rows["old"]["terminal_state"] == "ok"
    assert rows["new"]["terminal_state"] == "preempted" and rows["new"]["preempted"] == 1
    assert any(isinstance(r, QueueFullError) for r in results)
    assert sched.stats()["total_preempted"] == 1


class _PrefillFailsBackend(FakeChunkBackend):
    def prefill(self, token_ids, session_id="default"):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_error_row(tmp_path):
    store = RowStore(tmp_path)
    sched = ContinuousBatchScheduler(_PrefillFailsBackend(), telemetry=store)
    sched.start()
    with pytest.raises(RuntimeError):
        await sched.submit(_req(asyncio.get_running_loop()))
    await sched.stop()
    (row,) = _rows(store)
    assert row["terminal_state"] == "error"
    assert row["queue_wait_s"] is not None and row["ttft_s"] is None


def test_decode_width_is_measured_while_decoding_not_at_arrival(tmp_path):
    """The distinction the field exists for: a row that ARRIVES alone can DECODE in a crowd.

    Drives the loop's phases directly so the widths are exact rather than thread-timing luck.
    """
    store = RowStore(tmp_path)
    backend = FakeChunkBackend()
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, telemetry=store)
    loop = asyncio.new_event_loop()
    try:
        solo, late = _req(loop, session_id="solo"), _req(loop, session_id="late")
        for r in (solo, late):
            r.enqueue_ts = time.perf_counter()

        solo.record = sched._arrival_record(solo)        # arrives into an idle scheduler
        kv, tok, n = backend.prefill(solo.token_ids)
        sched._promote_to_decode(solo, kv, n, tok, "cpu")
        sched._decode_step("cpu")                        # width 1 — solo is alone

        late.record = sched._arrival_record(late)        # arrives while solo is decoding
        kv, tok, n = backend.prefill(late.token_ids)
        sched._promote_to_decode(late, kv, n, tok, "cpu")
        sched._decode_step("cpu")                        # width 2
        sched._decode_step("cpu")                        # width 2

        sched._finish_row(solo, "ok")
        sched._finish_row(late, "ok")
    finally:
        loop.close()

    rows = {r["session_id"]: r for r in _rows(store)}
    # solo arrived at width 0 and decoded at 1, 2, 2 — the arrival snapshot misses all of it
    assert rows["solo"]["active_size"] == 0
    assert rows["solo"]["decode_batch_width_mean"] == pytest.approx(5 / 3)
    assert rows["solo"]["decode_batch_width_max"] == 2
    # late arrived behind one row and decoded both its steps at width 2
    assert rows["late"]["active_size"] == 1
    assert rows["late"]["decode_batch_width_mean"] == 2.0
    assert rows["late"]["decode_batch_width_max"] == 2


def test_a_row_that_never_decodes_reports_no_width(tmp_path):
    """Rejected / expired rows have no decode steps: the mean must be None, not 0."""
    store = RowStore(tmp_path)
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_queue_size=1, telemetry=store)
    loop = asyncio.new_event_loop()
    try:
        sched.enqueue(_req(loop, session_id="a"))
        with pytest.raises(QueueFullError):
            sched.enqueue(_req(loop, session_id="b"))
    finally:
        loop.close()
    (row,) = _rows(store)
    assert row["decode_batch_width_mean"] is None and row["decode_batch_width_max"] == 0


def test_every_terminal_state_is_named():
    assert set(TERMINAL_STATES) == {"ok", "rejected_429", "expired", "preempted", "error"}


# --------------------------------------------------------------------------- store

def _record(**kw) -> RequestRecord:
    base = dict(trace_id="t", session_id="s", turn_index=0, arrival_ts=1.0, pending_depth=0,
                active_size=0, max_batch_size=1, kv_free_blocks=None, kv_free_frac=None,
                concurrent_sessions=0, prompt_tokens=1, replica_age_s=0.0)
    return RequestRecord(**{**base, **kw})


def test_one_sqlite_file_per_run_and_rows_round_trip(tmp_path):
    a, b = RowStore(tmp_path), RowStore(tmp_path)
    assert a.run_id != b.run_id and a.path != b.path
    assert a.path.name == f"{a.run_id}.sqlite"

    rec = _record(trace_id="x", kv_free_blocks=7, kv_free_frac=0.5, config_id="cfg-1")
    rec.finish("ok", enqueue_ts=10.0, admit_ts=10.5, first_token_ts=11.0, end_ts=13.0,
               tokens_out=5, cache_hit_tokens=2)
    a.put(rec)
    b.close()
    (row,) = _rows(a)
    assert row == asdict(rec)
    assert row["queue_wait_s"] == 0.5 and row["prefill_s"] == 0.5 and row["decode_s"] == 2.0
    assert row["ttft_s"] == 1.0 and row["total_s"] == 3.0 and row["tpot_s"] == 0.5
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([a.path.name, b.path.name])


def test_full_queue_drops_and_counts_instead_of_blocking(tmp_path):
    store = RowStore(tmp_path)
    live = store._q
    store._q = queue.Queue(maxsize=1)      # a queue nobody drains
    store._q.put_nowait(_record())
    try:
        t0 = time.perf_counter()
        store.put(_record())
        assert time.perf_counter() - t0 < 0.05
        assert store.rows_dropped == 1
    finally:
        store._q = live
        store.close()


def test_bad_batch_is_dropped_and_the_writer_keeps_going(tmp_path, caplog):
    store = RowStore(tmp_path)
    store._q.put_nowait(object())          # not a record: the batch cannot be written
    for _ in range(200):
        if store.rows_dropped:
            break
        time.sleep(0.005)
    assert store.rows_dropped == 1 and "telemetry write failed" in caplog.text
    assert store._thread.is_alive(), "the writer survived the bad batch"
    store.put(_record(trace_id="after"))
    assert [r["trace_id"] for r in _rows(store)] == ["after"]


def test_off_by_default_writes_nothing(tmp_path, monkeypatch):
    from inference_server.config import Settings, load_settings
    monkeypatch.delenv("TELEMETRY_DIR", raising=False)
    assert Settings.telemetry_dir == "" and load_settings().telemetry_dir == ""

    sched = ContinuousBatchScheduler(FakeChunkBackend())
    loop = asyncio.new_event_loop()
    try:
        req = _req(loop)
        sched.enqueue(req)
        assert req.record is None
        assert sched.stats()["telemetry"] == {"enabled": False, "rows_written": 0,
                                              "rows_dropped": 0}
    finally:
        loop.close()
    assert list(tmp_path.iterdir()) == []


def test_trace_id_is_generated_when_absent():
    loop = asyncio.new_event_loop()
    try:
        a, b = _req(loop), _req(loop)
        assert len(a.trace_id) == 32 and a.trace_id != b.trace_id
        assert _req(loop, trace_id="mine").trace_id == "mine"
    finally:
        loop.close()


# --------------------------------------------------------------------------- overhead budget

def test_overhead_budget(tmp_path):
    """Snapshot + finish + hand-off must stay under 200us mean per request.

    That is the whole per-request cost telemetry adds to the engine (the SQLite write happens
    on the writer thread). Expected ~10us; the budget is 20x that so CI noise cannot trip it,
    while a regression to anything blocking — a synchronous write, a lock on the writer — will.
    """
    store = RowStore(tmp_path, max_queue=20_000)
    sched = ContinuousBatchScheduler(FakeChunkBackend(), telemetry=store)
    loop = asyncio.new_event_loop()
    n = 5000
    try:
        reqs = [_req(loop, session_id=f"s{i % 50}") for i in range(n)]
        for r in reqs:
            r.enqueue_ts = time.perf_counter()
        t0 = time.perf_counter()
        for r in reqs:
            r.record = sched._arrival_record(r)
            sched._finish_row(r, "ok")
        per_request = (time.perf_counter() - t0) / n
    finally:
        loop.close()
    rows = _rows(store)
    assert len(rows) == n and store.rows_dropped == 0
    assert per_request < 200e-6, f"{per_request * 1e6:.1f}us per request"
