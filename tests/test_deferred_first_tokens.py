"""Mixed prefill with async decode and deferred first tokens: a prefill's first tokens come back as DeviceTokens whose
host copies land late; the next decode step reads them on the device, the scheduler reads them after that step is
queued. Outputs must equal the non-deferred run's, token for token, including a first token that is EOS."""

import asyncio

import pytest
import torch

from inference_server.backends.base import DeviceToken
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from tests.test_mixed_prefill import FakeMixedBackend


class _LateEvent:
    """An event that reports done only after `polls` queries (or on synchronize), like a copy still in flight."""

    def __init__(self, polls):
        self.polls, self.synced = polls, False

    def query(self):
        self.polls -= 1
        return self.synced or self.polls < 0

    def synchronize(self):
        self.synced = True


class DeferringBackend(FakeMixedBackend):
    """First token of request i is FIRST_TOKEN + 10 * i, via DeviceTokens with late events when deferring."""

    def __init__(self, polls=3, eos=None):
        super().__init__()
        self.defers_first_tokens = True
        self.polls, self.eos, self.deferred = polls, eos, 0

    def is_eos(self, token_id):
        return token_id == self.eos

    def prefill_chunk_batch(self, items, defer_tokens=False):
        out = super().prefill_chunk_batch(items)
        finals = [k for k, (_, _, final, _) in enumerate(items) if final]
        if not finals:
            return out
        firsts = [self.FIRST_TOKEN + 10 * (len(items[k][1]) % 7) for k in finals]   # distinct per prompt length
        if not defer_tokens:
            return [(kv, firsts[finals.index(k)] if k in finals else tok, n) for k, (kv, tok, n) in enumerate(out)]
        toks = torch.tensor(firsts)
        ev = _LateEvent(self.polls)
        self.deferred += len(finals)
        return [(kv, DeviceToken(toks, toks.clone(), finals.index(k), ev) if k in finals else tok, n)
                for k, (kv, tok, n) in enumerate(out)]


async def _run(backend, lens, max_tokens, async_decode):
    sched = ContinuousBatchScheduler(backend, max_batch_size=8, prefill_mode="mixed", prefill_step_tokens=64,
                                     async_decode=async_decode)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(range(n)), max_tokens=m, session_id=f"s{i}", future=loop.create_future())
                for i, (n, m) in enumerate(zip(lens, max_tokens))]
        return await asyncio.gather(*(sched.submit(r) for r in reqs)), sched
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_deferred_first_tokens_give_the_same_outputs():
    lens, mts = [10, 11, 12, 13, 20], [4, 1, 6, 3, 5]
    ref, _ = await _run(DeferringBackend(), lens, mts, async_decode=False)          # sync decode: never deferred
    b = DeferringBackend(polls=3)
    got, sched = await _run(b, lens, mts, async_decode=True)
    assert b.deferred == len(lens)
    assert got == ref
    assert all(len(o) == m for o, m in zip(got, mts))
    assert all(o[1:] == [o[0] + j for j in range(1, len(o))] for o in got)       # decode continued from the first
    assert sched.stats()["active_size"] == 0 and sched._pending is None


@pytest.mark.asyncio
async def test_a_first_token_that_is_eos_ends_the_request_with_nothing():
    b = DeferringBackend(polls=2, eos=DeferringBackend.FIRST_TOKEN + 10 * (12 % 7))
    got, sched = await _run(b, [10, 12], [5, 5], async_decode=True)
    assert got[1] == [] and got[0] == [230 + j for j in range(5)]
    assert sched.stats()["active_size"] == 0
