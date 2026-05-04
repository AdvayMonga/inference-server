"""Unit tests for SchedulingPolicy implementations."""

import asyncio

from inference_server.scheduler import ScheduledRequest
from inference_server.scheduling_policy import (
    FCFSPolicy,
    FairPolicy,
    create_scheduling_policy,
)


def _req(session_id: str, arrival_seq: int) -> ScheduledRequest:
    return ScheduledRequest(
        token_ids=[1, 2, 3],
        max_tokens=4,
        session_id=session_id,
        future=asyncio.Future(),
        arrival_seq=arrival_seq,
    )


def test_factory_known_policies():
    assert isinstance(create_scheduling_policy("fcfs"), FCFSPolicy)
    assert isinstance(create_scheduling_policy("fair"), FairPolicy)


def test_fcfs_returns_in_arrival_order():
    p = FCFSPolicy()
    a, b, c = _req("A", 1), _req("B", 2), _req("A", 3)
    p.on_request_arrived(a)
    p.on_request_arrived(b)
    p.on_request_arrived(c)
    assert p.pick_next() is a
    assert p.pick_next() is b
    assert p.pick_next() is c
    assert p.pick_next() is None


def test_fair_round_robin_when_counters_equal():
    p = FairPolicy()
    a1, b1, a2 = _req("A", 1), _req("B", 2), _req("A", 3)
    p.on_request_arrived(a1)
    p.on_request_arrived(b1)
    p.on_request_arrived(a2)
    # Both sessions start at 0; tiebreak by arrival_seq -> A first.
    first = p.pick_next()
    assert first.session_id == "A"
    p.on_tokens_processed(first, 5)  # A counter = 5
    # Now B (counter 0) wins.
    second = p.pick_next()
    assert second.session_id == "B"


def test_fair_late_arrival_not_starved():
    """The whole point of VTC: B arrives mid-flood and gets admitted next."""
    p = FairPolicy()
    a_reqs = [_req("A", i) for i in range(1, 51)]
    for r in a_reqs:
        p.on_request_arrived(r)

    # Drain a few of A's requests, accumulating counter.
    for _ in range(3):
        r = p.pick_next()
        p.on_tokens_processed(r, 100)  # A's counter climbs to 300

    # B arrives now — counter inherits min of active (A=300), so B starts at 300.
    b = _req("B", 100)
    p.on_request_arrived(b)

    # Next pick: A=300, B=300, tiebreak by arrival_seq.
    # A's next has arrival_seq=4, B's has 100, so A still wins this one.
    next_pick = p.pick_next()
    assert next_pick.session_id == "A"
    p.on_tokens_processed(next_pick, 100)  # A=400, B still 300

    # Now B must win.
    assert p.pick_next() is b


def test_fair_preserves_intra_session_fifo():
    p = FairPolicy()
    a1, a2, a3 = _req("A", 1), _req("A", 2), _req("A", 3)
    for r in (a3, a1, a2):  # arrived in this order
        p.on_request_arrived(r)
    # Within session A, FIFO by arrival into the deque.
    assert p.pick_next() is a3
    assert p.pick_next() is a1
    assert p.pick_next() is a2


def test_fair_empty_pick():
    assert FairPolicy().pick_next() is None
