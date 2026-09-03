"""Length-grouped prefill wave planning: saves padding without deferring anyone past their turn.

A batched prefill pads every row to Smax, so a ragged wave wastes most of its compute (~59% at
K=8 on a ShareGPT-ish length distribution). plan_wave groups similar lengths — but always
includes the window HEAD, which is what keeps it starvation-free.
"""

from __future__ import annotations

import asyncio
import random
import threading

import pytest

from inference_server.backends.base import InferenceBackend
from inference_server.scheduler import (
    ContinuousBatchScheduler,
    ScheduledRequest,
    plan_wave,
)
from inference_server.scheduling_policy import FairPolicy, FCFSPolicy


def _req(n_tokens: int, seq: int = 0, session: str = "s") -> ScheduledRequest:
    r = ScheduledRequest(token_ids=list(range(n_tokens)), max_tokens=4,
                         session_id=session, future=None)
    r.arrival_seq = seq
    return r


def _pad_cost(reqs):
    lens = [len(r.token_ids) for r in reqs]
    return max(lens) * len(lens) - sum(lens)


def test_head_is_always_included():
    """The starvation guarantee: whoever is at the front goes this wave, whatever its length."""
    window = [_req(900)] + [_req(10 + i) for i in range(20)]
    for k in (1, 2, 4, 8):
        assert window[0] in plan_wave(window, k), f"head dropped at k={k}"


def test_groups_similar_lengths_around_the_head():
    window = [_req(100), _req(900), _req(105), _req(880), _req(95), _req(910)]
    got = sorted(len(r.token_ids) for r in plan_wave(window, 3))
    assert got == [95, 100, 105]


def test_beats_arrival_order_on_padding():
    random.seed(0)
    window = [_req(random.randint(20, 800), seq=i) for i in range(32)]
    assert _pad_cost(plan_wave(window, 8)) < _pad_cost(window[:8])


def test_returns_everything_when_window_fits():
    window = [_req(10), _req(20), _req(30)]
    assert plan_wave(window, 8) == window
    assert plan_wave(window, 3) == window


def test_degenerate_inputs():
    assert plan_wave([], 4) == []
    assert plan_wave([_req(5)], 0) == []


def test_wave_is_admitted_in_policy_order():
    """Grouping picks WHICH requests share a wave, not the order they are admitted in."""
    window = [_req(100, seq=0), _req(900, seq=1), _req(105, seq=2), _req(95, seq=3)]
    got = plan_wave(window, 3)
    assert [r.arrival_seq for r in got] == sorted(r.arrival_seq for r in got)


@pytest.mark.parametrize("policy_cls", [FCFSPolicy, FairPolicy])
def test_peek_window_is_policy_ordered(policy_cls):
    policy = policy_cls()
    reqs = [_req(10, seq=i) for i in range(5)]
    for r in reversed(reqs):           # arrive out of order
        policy.on_request_arrived(r)
    assert [r.arrival_seq for r in policy.peek_window(3)] == [0, 1, 2]
    assert len(policy.peek_window(99)) == 5


@pytest.mark.parametrize("policy_cls", [FCFSPolicy, FairPolicy])
def test_pick_removes_that_specific_request(policy_cls):
    policy = policy_cls()
    reqs = [_req(10, seq=i) for i in range(3)]
    for r in reqs:
        policy.on_request_arrived(r)
    policy.pick(reqs[1])                                  # not the head
    assert [r.arrival_seq for r in policy.peek_window(9)] == [0, 2]


def test_default_peek_window_does_not_reorder():
    """A custom policy that does not override peek_window keeps strict ordering."""
    from inference_server.scheduling_policy import SchedulingPolicy
    policy = FCFSPolicy()
    for r in [_req(10, seq=i) for i in range(4)]:
        policy.on_request_arrived(r)
    head_only = SchedulingPolicy.peek_window(policy, 4)
    assert len(head_only) == 1 and head_only[0].arrival_seq == 0


class _FakeBackend(InferenceBackend):
    needs_attention_mask = False

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0
        self.waves = []

    def load_model(self, model_name): pass
    def generate(self, *a, **k): return []
    def generate_batch(self, *a, **k): return []
    def generate_step(self, *a, **k): return 0, None
    def stream(self, *a, **k):
        if False:
            yield 0

    def prefill_batch(self, prompts, session_ids=None):
        self.waves.append([len(p) for p in prompts])
        return [(len(p), 1, len(p)) for p in prompts]

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids,
                            sampling_per_row=None):
        for i in range(len(batched_kv)):
            batched_kv[i] += 1
        return current_tokens.squeeze(-1) + 1, batched_kv

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        if batched_kv is None:
            return [new_kv]
        batched_kv.append(new_kv)
        return batched_kv

    def remove_row_from_cache(self, batched_kv, row_idx):
        batched_kv.pop(row_idx)
        return batched_kv

    def kv_length(self, kv): return max(kv) if kv else 0
    def is_eos(self, token_id): return False

    @property
    def device_str(self): return "cpu"


@pytest.mark.asyncio
async def test_scheduler_serves_everyone_with_grouping_on():
    """End-to-end: grouping must not drop, duplicate or starve a request."""
    backend = _FakeBackend()
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, prefill_mode="batched",
                                     wave_window_mult=4)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        lens = [900, 12, 14, 800, 16, 850, 11, 13]
        reqs = [ScheduledRequest(token_ids=list(range(n)), max_tokens=3,
                                 session_id=f"s{i}", future=loop.create_future())
                for i, n in enumerate(lens)]
        out = await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)
        for i, r in enumerate(out):
            assert not isinstance(r, Exception), f"request {i} failed: {r!r}"
            assert len(r) == 3
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_grouping_off_matches_strict_order():
    backend = _FakeBackend()
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, prefill_mode="batched",
                                     wave_window_mult=0)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(range(10 + i)), max_tokens=3,
                                 session_id=f"s{i}", future=loop.create_future())
                for i in range(6)]
        out = await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)
        assert all(not isinstance(r, Exception) and len(r) == 3 for r in out)
    finally:
        await sched.stop()
