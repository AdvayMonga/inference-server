"""The pure ordering functions reproduce FCFSPolicy/FairPolicy exactly, on plain records.

A trace-replay simulator will feed these functions its own records, never ScheduledRequest,
so the properties here are: same pick sequence as the classes, no input mutation, order
independent of input permutation, duck-typed candidates.
"""

import random
from dataclasses import dataclass

import pytest

from inference_server.scheduler import ScheduledRequest
from inference_server.scheduling_policy import (
    FCFSPolicy,
    FairPolicy,
    fair_charge,
    fair_initial_counter,
    fair_order,
    fcfs_order,
)


@dataclass(frozen=True)
class Cand:
    """A simulator-style record: the three fields ordering reads, nothing else."""
    session_id: str
    priority: int
    arrival_seq: int


def _req(c: Cand) -> ScheduledRequest:
    return ScheduledRequest(token_ids=[1], max_tokens=1, session_id=c.session_id, future=None,
                            arrival_seq=c.arrival_seq, priority=c.priority)


def _cands(rng: random.Random, n: int) -> list[Cand]:
    return [Cand(f"s{rng.randint(0, 4)}", rng.randint(0, 2), i) for i in range(n)]


@pytest.mark.parametrize("seed", range(20))
def test_fcfs_policy_matches_fcfs_order(seed):
    rng = random.Random(seed)
    cands = _cands(rng, rng.randint(0, 30))
    policy = FCFSPolicy()
    for c in cands:
        policy.on_request_arrived(_req(c))
    picked = []
    while (r := policy.pick_next()) is not None:
        picked.append(r.arrival_seq)
    assert picked == [c.arrival_seq for c in fcfs_order(cands)]


@pytest.mark.parametrize("seed", range(20))
def test_fair_policy_matches_pure_replay(seed):
    """Interleave arrivals, picks and charges; a pure-function replay must pick identically."""
    rng = random.Random(seed)
    policy, counters, waiting, seq = FairPolicy(), {}, [], 0
    for _ in range(60):
        op = rng.random()
        if op < 0.5 or not waiting:
            c = Cand(f"s{rng.randint(0, 4)}", rng.randint(0, 2), seq)
            seq += 1
            policy.on_request_arrived(_req(c))
            counters[c.session_id] = fair_initial_counter(
                c.session_id, counters, (w.session_id for w in waiting))
            waiting.append(c)
        elif op < 0.8:
            got = policy.pick_next()
            want = fair_order(waiting, counters)[0]
            assert got.arrival_seq == want.arrival_seq
            waiting.remove(want)
        else:
            sid, n = f"s{rng.randint(0, 4)}", rng.randint(1, 50)
            policy.on_tokens_processed(_req(Cand(sid, 0, 0)), n)
            fair_charge(counters, sid, n)
        assert [r.arrival_seq for r in policy.peek_window(99)] == \
            [c.arrival_seq for c in fair_order(waiting, counters)]


def test_pure_functions_do_not_mutate_inputs():
    cands = _cands(random.Random(1), 12)
    counters = {"s0": 5.0, "s1": 0.0}
    before, before_counters = list(cands), dict(counters)
    fcfs_order(cands)
    fair_order(cands, counters)
    fair_initial_counter("s9", counters, ["s0", "s1"])
    assert cands == before and counters == before_counters


def test_order_is_independent_of_input_permutation():
    cands = _cands(random.Random(2), 25)
    counters = {f"s{i}": float(i % 3) for i in range(5)}
    shuffled = list(cands)
    random.Random(3).shuffle(shuffled)
    assert fcfs_order(shuffled) == fcfs_order(cands)
    assert fair_order(shuffled, counters) == fair_order(cands, counters)
    assert fair_order(cands, counters) is not cands  # a new list, not the input


def test_tie_breaks_are_the_documented_keys():
    a, b, c = Cand("A", 0, 1), Cand("B", 0, 2), Cand("A", 5, 3)
    assert fcfs_order([a, b, c]) == [c, a, b]                         # priority, then arrival
    assert fair_order([a, b, c], {"A": 10.0, "B": 0.0}) == [c, b, a]  # priority, counter, arrival


def test_fair_initial_counter_rule():
    counters = {"idle": 1.0, "busy": 7.0, "known": 3.0}
    assert fair_initial_counter("new", counters, []) == 0.0            # nothing pending
    assert fair_initial_counter("new", counters, ["busy"]) == 7.0     # min over pending only
    assert fair_initial_counter("new", counters, ["busy", "ghost"]) == 7.0  # unknown pending ignored
    assert fair_initial_counter("known", counters, ["busy"]) == 3.0   # existing counter kept


def test_fair_charge_is_in_place_and_starts_at_zero():
    counters = {}
    fair_charge(counters, "s", 4)
    fair_charge(counters, "s", 3)
    assert counters == {"s": 7.0}
