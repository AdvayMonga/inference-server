"""Mixed prefill: every waiting prefill advances by up to prefill_step_tokens in one forward per iteration."""

import asyncio

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, QueueFullError, ScheduledRequest
from tests.test_chunked_prefill import FakeChunkBackend, _FakeKV


class FakeMixedBackend(FakeChunkBackend):
    """Records every mixed forward as the list of chunk lengths it carried."""

    def __init__(self, fail=False):
        super().__init__()
        self.calls: list[list[int]] = []
        self.fail = fail

    def prefill_chunk_batch(self, items):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append([len(chunk) for _, chunk, _, _ in items])
        out = []
        for kv, chunk, final, _ in items:
            n = (kv.seq_len if kv is not None else 0) + len(chunk)
            self.chunk_input_tokens_total += len(chunk)
            out.append((_FakeKV(1, n), self.FIRST_TOKEN if final else None, n))
        return out


async def _run(backend, lens, step_tokens, max_tokens=3, **kw):
    sched = ContinuousBatchScheduler(backend, max_batch_size=8, prefill_mode="mixed",
                                     prefill_step_tokens=step_tokens, **kw)
    loop = asyncio.get_running_loop()
    reqs = [ScheduledRequest(token_ids=list(range(n)), max_tokens=max_tokens, session_id=f"s{i}",
                             future=loop.create_future()) for i, n in enumerate(lens)]
    for r in reqs:
        sched.enqueue(r)                  # all waiting before the loop starts: one admission wave
    sched.start()
    try:
        return await asyncio.gather(*(r.future for r in reqs), return_exceptions=True), sched
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_a_step_feeds_rows_in_order_up_to_the_token_budget():
    b = FakeMixedBackend()
    outs, sched = await _run(b, [1000, 600, 300], step_tokens=1000)
    assert outs == [[200, 201, 202]] * 3                       # the same tokens monolithic prefill gives
    assert b.calls[:2] == [[1000], [600, 300]]                 # row 1 fills the budget; rows 2+3 share the next step
    assert b.chunk_input_tokens_total == 1900 and b.store_calls == [(1000, 0), (600, 0), (300, 0)]
    assert sched.stats()["prefill_mode"] == "mixed" and sched.stats()["prefill_step_tokens"] == 1000


@pytest.mark.asyncio
async def test_a_prompt_longer_than_the_budget_takes_several_steps():
    b = FakeMixedBackend()
    outs, _ = await _run(b, [1200], step_tokens=500)
    assert outs == [[200, 201, 202]] and b.calls == [[500], [500], [200]]


@pytest.mark.asyncio
async def test_a_failed_mixed_forward_rejects_its_rows_and_frees_the_slots():
    b = FakeMixedBackend(fail=True)
    outs, sched = await _run(b, [100, 100], step_tokens=1000)
    assert all(isinstance(o, QueueFullError) or isinstance(o, RuntimeError) for o in outs)
    assert not sched._prefilling and sched._active_kv_reserved == 0


def test_mixed_needs_a_backend_with_the_batch_primitive():
    with pytest.raises(ValueError, match="prefill_chunk_batch"):
        ContinuousBatchScheduler(_NoPrimitive(), prefill_mode="mixed")


class _NoPrimitive(FakeChunkBackend):
    def __getattribute__(self, name):
        if name == "prefill_chunk_batch":
            raise AttributeError(name)
        return super().__getattribute__(name)
