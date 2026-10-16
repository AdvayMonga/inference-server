"""Async decode: step n runs on step n-1's device tokens; the host processes n-1 meanwhile. Same outputs as sync."""

import asyncio

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from tests.test_chunked_prefill import FakeChunkBackend


class EosBackend(FakeChunkBackend):
    """Decode adds one per step; token EOS_AT ends a request (so a row finishes mid-batch)."""
    EOS_AT = 203

    def is_eos(self, token_id):
        return token_id == self.EOS_AT


async def _run(backend, max_tokens_list, async_decode, cancel_first_after=None):
    sched = ContinuousBatchScheduler(backend, max_batch_size=8, async_decode=async_decode)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(range(10 + i)), max_tokens=m, session_id=f"s{i}",
                                 future=loop.create_future()) for i, m in enumerate(max_tokens_list)]
        tasks = [asyncio.ensure_future(sched.submit(r)) for r in reqs]
        if cancel_first_after is not None:
            await asyncio.sleep(cancel_first_after)
            reqs[0].cancelled = True
        return await asyncio.gather(*tasks, return_exceptions=True), sched
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_async_matches_sync_with_rows_finishing_at_different_steps():
    sync, _ = await _run(FakeChunkBackend(), [5, 3, 7, 1], async_decode=False)
    asyn, sched = await _run(FakeChunkBackend(), [5, 3, 7, 1], async_decode=True)
    assert asyn == sync == [[200, 201, 202, 203, 204], [200, 201, 202], [200 + i for i in range(7)], [200]]
    assert sched.stats()["async_decode"] is True and sched.stats()["total_completed"] == 4


@pytest.mark.asyncio
async def test_eos_found_one_step_late_drops_the_extra_token_and_frees_the_row():
    outs, sched = await _run(EosBackend(), [10, 10, 2], async_decode=True)
    assert outs == [[200, 201, 202], [200, 201, 202], [200, 201]]     # EOS (203) ends the long rows, never appended
    assert sched.stats()["active_size"] == 0 and sched._pending is None


@pytest.mark.asyncio
async def test_a_cancelled_row_leaves_while_its_token_is_still_pending():
    outs, sched = await _run(FakeChunkBackend(), [200, 4], async_decode=True, cancel_first_after=0.05)
    assert outs[1] == [200, 201, 202, 203] and sched.stats()["active_size"] == 0
