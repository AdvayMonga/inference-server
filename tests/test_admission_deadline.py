"""The admission deadline is enforced head-of-line only, and neither policy orders by age.

`_admit_pending` checks `max_queue_wait_s` against the request the policy hands back from
`peek_next()`, and nothing else. That is only equivalent to "shed everything that is too stale"
if the head of the queue is always the oldest request — and it is not:

    FCFS  min(pending, key=(-priority, arrival_seq))
    VTC   min(pending, key=(-priority, session_counter, arrival_seq))

Both put priority first, and VTC puts a fairness counter ahead of arrival order. So a request can
sit past its deadline indefinitely while newer, higher-priority or lighter-session requests keep
overtaking it. It is never the head, so its deadline is never evaluated.

Consequences: total_expired under-reports, stale requests keep holding queue memory, and a client
that gave up long ago is still eventually served — the server does the work and nobody reads it.
Measured context: a sweep at 128 req/s queued 2985 requests with a 30s deadline against a 200ms
TTFT budget, and reported TTFT p50 of 29.9s.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.scheduling_policy import FCFSPolicy
from tests.test_chunked_prefill import FakeChunkBackend


def _req(loop, *, session="s", priority=0, aged_s=0.0, seq=0):
    r = ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id=session,
                         future=loop.create_future(), priority=priority, arrival_seq=seq)
    r.enqueue_ts = time.perf_counter() - aged_s
    return r


def test_neither_policy_orders_by_age():
    """The assumption the head-of-line check rests on, stated as a test. If a policy ever did
    order strictly by arrival, checking only the head would be sound."""
    fcfs = FCFSPolicy()
    loop = asyncio.new_event_loop()
    try:
        old_low = _req(loop, priority=0, aged_s=100.0, seq=0)
        new_high = _req(loop, priority=5, aged_s=0.0, seq=1)
        fcfs.on_request_arrived(old_low)
        fcfs.on_request_arrived(new_high)

        assert fcfs.peek_next() is new_high, "priority outranks arrival, so the head is not oldest"
    finally:
        loop.close()


def test_a_starved_request_is_shed_even_when_it_is_never_the_head():
    """The fix, tested deterministically.

    Driving this through a RUNNING scheduler is racy — it can see the stale request as head in
    the window before the higher-priority ones arrive, and then the old head-of-line check sheds
    it for the wrong reason and the test passes against buggy code. So the sweep is exercised
    directly, with the queue arranged so the stale request is provably never the head.
    """
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=1,
                                     prefill_mode="batched", max_queue_wait_s=0.05)
    loop = asyncio.new_event_loop()
    try:
        stale = ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id="starved",
                                 future=loop.create_future(), priority=0)
        sched.enqueue(stale)
        stale.enqueue_ts = time.perf_counter() - 60.0        # 1200x over the deadline
        hot = []
        for i in range(5):
            h = ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id=f"hot{i}",
                                 future=loop.create_future(), priority=9)
            sched.enqueue(h)
            hot.append(h)

        # Precondition: the only request the old check could ever inspect is a fresh one.
        assert sched.policy.peek_next() is not stale
        assert not stale.future.done()

        sched._shed_expired()

        # Assertions are on scheduler state, not the future: delivering the rejection is
        # _reject's job and needs a running loop (covered by the head-of-line test below).
        assert sched.stats()["total_expired"] == 1, "shed from anywhere in the queue, not just head"
        assert sched.stats()["total_rejected"] == 1
        assert sched.policy.pending() == hot, "the stale request left; fresh work untouched"
        assert sched.stats()["pending_depth"] == len(hot)
    finally:
        loop.close()


def test_the_sweep_is_actually_wired_into_admission():
    """Separate from whether the sweep works: that it RUNS. Testing _shed_expired() directly
    passes even if nothing ever calls it, so this drives the real admission path instead."""
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=1,
                                     prefill_mode="batched", max_queue_wait_s=0.05)
    loop = asyncio.new_event_loop()
    try:
        stale = ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id="starved",
                                 future=loop.create_future(), priority=0)
        sched.enqueue(stale)
        stale.enqueue_ts = time.perf_counter() - 60.0
        for i in range(5):
            h = ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id=f"hot{i}",
                                 future=loop.create_future(), priority=9)
            sched.enqueue(h)

        assert sched.policy.peek_next() is not stale
        sched._admit_pending("cpu")

        assert sched.stats()["total_expired"] == 1, (
            "admission must sweep the whole queue for expiry, not only inspect its head")
        assert stale not in sched.policy.pending()
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_fresh_requests_survive_the_sweep():
    """The sweep must shed only what is actually stale — rejecting live work would be far worse
    than the bug it fixes."""
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=4,
                                     prefill_mode="batched", max_queue_wait_s=30.0)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id=f"s{i}",
                                 future=loop.create_future()) for i in range(4)]
        out = await asyncio.gather(*(sched.submit(r) for r in reqs))

        assert all(len(o) > 0 for o in out)
        assert sched.stats()["total_expired"] == 0
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_the_head_of_line_case_does_shed():
    """The case that always worked: equal priority, one session, head is oldest."""
    backend = FakeChunkBackend()
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, prefill_mode="batched",
                                     max_queue_wait_s=0.05)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        req = ScheduledRequest(token_ids=[1, 2, 3], max_tokens=2, session_id="s",
                               future=loop.create_future())
        sched.enqueue(req)
        # Backdate AFTER enqueue, which stamps enqueue_ts: the deadline is measured from it.
        req.enqueue_ts = time.perf_counter() - 10.0

        with pytest.raises(Exception) as excinfo:
            await asyncio.wait_for(req.future, timeout=5.0)
        assert "deadline" in str(excinfo.value) or "overloaded" in str(excinfo.value)
        assert sched.stats()["total_expired"] >= 1
    finally:
        await sched.stop()


def test_the_deadline_default_is_150x_the_ttft_budget():
    """Not a bug, a calibration question, recorded so it is not forgotten: shedding at 30s when
    the TTFT SLO is 200ms means a caller waits half a minute to be refused. A deadline should be
    a small multiple of the latency budget, not two orders of magnitude above it."""
    from inference_server.config import Settings

    slo_ttft_ms = 200.0                       # bench_serving_modal.SLO_TTFT_MS
    assert Settings.max_queue_wait_s == 30.0
    assert Settings.max_queue_wait_s * 1000 / slo_ttft_ms == 150.0
