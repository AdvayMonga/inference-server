"""ignore_eos: the scheduler keeps a row past its stop token until max_tokens; without it the row ends there."""

import asyncio

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from tests.stub_backend import StubBackend


class EosAt(StubBackend):
    """The stub generates 1, 2, 3, ...; `eos` is a stop token."""

    def __init__(self, eos):
        super().__init__()
        self.eos = eos

    def is_eos(self, token_id):
        return token_id == self.eos


async def _run(ignore_eos):
    sched = ContinuousBatchScheduler(EosAt(eos=3), max_batch_size=4)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        req = ScheduledRequest(token_ids=list(range(10)), max_tokens=6, session_id="s",
                               future=loop.create_future(), ignore_eos=ignore_eos)
        return await sched.submit(req)
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_a_row_stops_at_its_stop_token_by_default():
    out = await _run(ignore_eos=False)
    assert out == [1, 2]


@pytest.mark.asyncio
async def test_ignore_eos_runs_to_max_tokens_and_keeps_the_stop_token():
    out = await _run(ignore_eos=True)
    assert out == [1, 2, 3, 4, 5, 6]
