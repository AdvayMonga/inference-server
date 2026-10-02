"""The engine event timeline: what the scheduler records, and that an off timeline records nothing."""

from __future__ import annotations

import asyncio
import json

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.timeline import Timeline
from tests.stub_backend import StubBackend


def _events(path):
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def test_off_timeline_is_a_no_op(tmp_path):
    tl = Timeline()
    tl.begin_step()
    tl.event("x", a=1)
    with tl.phase("p", b=2):
        pass
    tl.close()
    assert not tl.enabled and tl.stats()["events_written"] == 0
    assert list(tmp_path.iterdir()) == []


def test_events_and_phases_carry_the_step(tmp_path):
    tl = Timeline(tmp_path)
    tl.begin_step()
    tl.event("a", n=1)
    with tl.phase("work", batch=3):
        pass
    tl.begin_step()
    tl.event("b")
    tl.close()
    ev = _events(tmp_path)
    assert [(e["kind"], e["step"]) for e in ev] == [("a", 1), ("phase", 1), ("b", 2)]
    assert ev[1]["name"] == "work" and ev[1]["batch"] == 3 and ev[1]["dur_s"] >= 0 and ev[1]["ok"]
    assert all("ts" in e for e in ev)
    assert tl.stats() == {"enabled": True, "step": 2, "events_written": 3, "events_dropped": 0}


def test_events_after_close_are_counted_as_dropped(tmp_path):
    tl = Timeline(tmp_path)
    tl.event("a")
    tl.close()
    tl.event("b")
    tl.close()
    assert tl.stats()["events_written"] == 1 and tl.stats()["events_dropped"] == 1
    assert len(_events(tmp_path)) == 1


def test_phase_records_failure(tmp_path):
    tl = Timeline(tmp_path)
    with pytest.raises(RuntimeError):
        with tl.phase("boom"):
            raise RuntimeError
    tl.close()
    assert _events(tmp_path)[0]["ok"] is False


async def _serve(tmp_path, n=3, max_tokens=4):
    tl = Timeline(tmp_path)
    sched = ContinuousBatchScheduler(StubBackend(), max_batch_size=n, timeline=tl)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=[1, 2, 3], max_tokens=max_tokens, session_id=f"s{i}",
                                 future=loop.create_future()) for i in range(n)]
        out = await asyncio.gather(*(sched.submit(r) for r in reqs))
    finally:
        await sched.stop()
    return sched, out, _events(tmp_path)


@pytest.mark.asyncio
async def test_scheduler_writes_the_request_lifecycle(tmp_path):
    sched, out, ev = await _serve(tmp_path)
    assert all(len(o) == 4 for o in out)
    kinds = {e["kind"] for e in ev}
    assert {"enqueue", "admit", "first_token", "finish", "step", "phase"} <= kinds
    for trace_id in {e["trace_id"] for e in ev if e["kind"] == "enqueue"}:
        mine = [e["kind"] for e in ev if e.get("trace_id") == trace_id]
        assert mine.index("enqueue") < mine.index("admit") < mine.index("first_token") < mine.index("finish")
    finishes = [e for e in ev if e["kind"] == "finish"]
    assert len(finishes) == 3 and all(e["state"] == "ok" and e["tokens_out"] == 4 for e in finishes)
    steps = [e["step"] for e in ev]
    assert steps == sorted(steps) and steps[-1] >= 1
    step_rows = [e for e in ev if e["kind"] == "step"]
    assert max(e["active"] for e in step_rows) == 3
    assert {e["name"] for e in ev if e["kind"] == "phase"} >= {"admit", "evict", "decode", "prefill"}
    assert sched.stats()["timeline"]["enabled"] is True


@pytest.mark.asyncio
async def test_idle_ticks_write_nothing(tmp_path):
    sched = ContinuousBatchScheduler(StubBackend(), max_batch_size=1, timeline=Timeline(tmp_path))
    sched.start()
    await asyncio.sleep(0.35)      # several idle ticks of the 0.1s wait
    await sched.stop()
    assert sched.stats()["timeline"]["step"] == 0
    assert (tmp_path / "events.jsonl").read_text() == ""


@pytest.mark.asyncio
async def test_queue_full_rejection_has_an_enqueue_event(tmp_path):
    tl = Timeline(tmp_path)
    sched = ContinuousBatchScheduler(StubBackend(), max_batch_size=1, max_queue_size=1, timeline=tl)
    loop = asyncio.get_running_loop()
    first = ScheduledRequest(token_ids=[1], max_tokens=1, session_id="a", future=loop.create_future())
    second = ScheduledRequest(token_ids=[1], max_tokens=1, session_id="b", future=loop.create_future())
    sched.enqueue(first)                      # worker not started: it stays pending and fills the queue
    with pytest.raises(Exception):
        sched.enqueue(second)
    tl.close()
    mine = [(e["kind"], e.get("state")) for e in _events(tmp_path) if e.get("trace_id") == second.trace_id]
    assert mine == [("enqueue", None), ("finish", "rejected_429")]


@pytest.mark.asyncio
async def test_scheduler_default_timeline_is_off(tmp_path):
    sched = ContinuousBatchScheduler(StubBackend(), max_batch_size=1)
    sched.start()
    await sched.stop()
    assert sched.stats()["timeline"] == {"enabled": False, "step": 0, "events_written": 0,
                                         "events_dropped": 0}
