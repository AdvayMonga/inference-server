"""ENGINE_PROCESS: the scheduler in a spawned child, driven through EngineClient as the HTTP routes drive it — streamed
and whole results, an engine-side rejection, a cancelled stream, logprob scoring over the pipe."""

import asyncio

import pytest

from inference_server.engine_process import EngineClient
from inference_server.scheduler import RequestTooLargeError, ScheduledRequest


class _Backend:
    """FakeMixedBackend plus the two calls the engine side forwards (built in the child)."""

    @staticmethod
    def make():
        from inference_server.scheduler import ContinuousBatchScheduler
        from tests.test_mixed_prefill import FakeMixedBackend

        class B(FakeMixedBackend):
            def kv_capacity_error(self, prompt_len, max_tokens):
                return "too big" if prompt_len + max_tokens > 10_000 else None

            def score_logprobs(self, token_ids, k):
                return [{t: -1.0} for t in token_ids[:k]]

        b = B()
        return b, ContinuousBatchScheduler(b, max_batch_size=8, prefill_mode="mixed", async_decode=True)


def make():
    return _Backend.make()


@pytest.mark.asyncio
async def test_engine_in_a_child_process_serves_streams_results_rejections_cancels_and_scores():
    client = EngineClient(factory="tests.test_engine_process:make")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, client.launch, 120)
    client.start()
    try:
        streamed = ScheduledRequest(token_ids=list(range(12)), max_tokens=5, session_id="a",
                                    future=loop.create_future(), token_queue=asyncio.Queue())
        client.enqueue(streamed)
        toks = []
        while (t := await asyncio.wait_for(streamed.token_queue.get(), 20)) is not None:
            toks.append(t)
        assert toks == [200, 201, 202, 203, 204] and await streamed.future == toks

        whole = ScheduledRequest(token_ids=list(range(7)), max_tokens=3, session_id="b", future=loop.create_future())
        assert await asyncio.wait_for(client.submit(whole), 20) == [200, 201, 202]

        big = ScheduledRequest(token_ids=list(range(9_000)), max_tokens=2_000, session_id="c",
                               future=loop.create_future(), token_queue=asyncio.Queue())
        client.enqueue(big)
        assert await asyncio.wait_for(big.token_queue.get(), 20) is None
        with pytest.raises(RequestTooLargeError):
            await big.future

        long = ScheduledRequest(token_ids=list(range(5)), max_tokens=100_000 // 20, session_id="d",
                                future=loop.create_future(), token_queue=asyncio.Queue())
        client.enqueue(long)
        await asyncio.wait_for(long.token_queue.get(), 20)
        long.cancelled = True                                   # the client went away
        while (t := await asyncio.wait_for(long.token_queue.get(), 20)) is not None:
            pass
        assert len(long.generated) < 5000

        scored = await loop.run_in_executor(None, client.backend.score_logprobs, [5, 6, 7], 2)
        assert scored == [{5: -1.0}, {6: -1.0}]
        await asyncio.sleep(0.7)
        assert client.stats().get("total_completed", 0) >= 2
    finally:
        await client.stop()


def make_eos():
    """The same engine, with token 202 a stop token."""
    backend, sched = _Backend.make()
    backend.is_eos = lambda t: t == 202
    return backend, sched


@pytest.mark.asyncio
async def test_ignore_eos_crosses_into_the_child():
    client = EngineClient(factory="tests.test_engine_process:make_eos")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, client.launch, 120)
    client.start()
    try:
        def req(sid, **kw):
            return ScheduledRequest(token_ids=list(range(7)), max_tokens=5, session_id=sid, future=loop.create_future(),
                                    **kw)
        assert await asyncio.wait_for(client.submit(req("stops")), 20) == [200, 201]
        assert await asyncio.wait_for(client.submit(req("runs", ignore_eos=True)), 20) == [200, 201, 202, 203, 204]
    finally:
        await client.stop()


def make_lock_probe():
    """The same engine; its stats say whether the pipe lock was held while they were read."""
    backend, sched = _Backend.make()
    stats = sched.stats
    sched.stats = lambda: {**stats(), "pipe_lock_held": sched._loop.outbox.lock.locked()}
    return backend, sched


@pytest.mark.asyncio
async def test_stats_are_read_outside_the_pipe_lock():
    """The scheduler sends (taking the pipe lock) while holding its own lock, so reading its stats under the pipe lock
    deadlocks both: seen under overload on an H200, 2026-10-28."""
    client = EngineClient(factory="tests.test_engine_process:make_lock_probe")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, client.launch, 120)
    client.start()
    try:
        await asyncio.sleep(1.2)
        assert client.stats().get("pipe_lock_held") is False
    finally:
        await client.stop()
