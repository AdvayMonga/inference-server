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


@pytest.mark.asyncio
async def test_a_starved_request_is_never_checked_against_its_deadline():
    """The bug, pinned as a CHARACTERIZATION test — it asserts today's WRONG behaviour so the
    defect is visible and measurable, not so it is preserved.

    When expiry is fixed to sweep the whole pending set, this test must be INVERTED (the stale
    request should be shed, and total_expired should count it), not deleted.

    A low-priority request ages past the deadline but never reaches the head, so the only code
    that could shed it never looks at it.
    """
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=1,
                                     prefill_mode="batched", max_queue_wait_s=1.0)
    loop = asyncio.get_running_loop()

    stale = _req(loop, priority=0, aged_s=60.0, seq=0)          # 60x over the 1s deadline
    fresh = [_req(loop, priority=9, aged_s=0.0, seq=i + 1) for i in range(5)]
    for r in [stale] + fresh:
        sched.policy.on_request_arrived(r)

    # Whatever the policy hands back, it is never the stale request — so the deadline check in
    # _admit_pending, which only inspects peek_next(), cannot see it.
    seen = {id(sched.policy.peek_next()) for _ in range(10)}
    assert id(stale) not in seen, (
        "a request 60x past its deadline is invisible to a head-of-line-only expiry check")


@pytest.mark.asyncio
async def test_the_head_of_line_case_does_shed():
    """The mechanism itself is correct — this is a coverage gap, not a broken implementation.
    With equal priority and one session, the head IS the oldest and shedding works."""
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
