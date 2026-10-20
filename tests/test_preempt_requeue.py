"""KV pressure: a row whose step finds the pool empty is preempted and requeued (prefilled again from prompt +
generated), and its output is the same as without the preemption; admission reserves the 90th percentile of recent
output lengths instead of max_tokens once enough requests have finished."""

import asyncio

import pytest

from inference_server.models.paged_kv_cache import KVCacheExhausted
from inference_server.scheduler import OUT_MIN_SEEN, ContinuousBatchScheduler, ScheduledRequest
from tests.test_chunked_prefill import _FakeKV
from tests.test_mixed_prefill import FakeMixedBackend


class ContinuingBackend(FakeMixedBackend):
    """Prefill of a sequence returns the token that continues it: 200 + (tokens past the original prompt), so a resumed
    prefill (prompt + generated) picks up where decode stopped. Decode adds one. `exhaust_at` decode steps raise
    KVCacheExhausted once each; `capacity` caps the reserved tokens kv_reserve admits."""

    def __init__(self, exhaust_at=(), capacity=None):
        super().__init__()
        self.base: dict[int, int] = {}
        self.exhaust_at, self.steps, self.prefills = set(exhaust_at), 0, []
        self.capacity, self.reserved, self.reservations = capacity, 0, []

    def prefill_chunk_batch(self, items, defer_tokens=False):
        out = []
        for kv, chunk, final, _ in items:
            n = (kv.seq_len if kv is not None else 0) + len(chunk)
            tok = None
            if final:
                self.prefills.append(n)
                base = self.base.setdefault(chunk[0], n)    # prompts start with distinct ids
                tok = self.FIRST_TOKEN + (n - base)
            out.append((_FakeKV(1, n), tok, n))
        return out

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids, sampling_per_row=None):
        self.steps += 1
        if self.steps in self.exhaust_at:
            self.exhaust_at.discard(self.steps)
            raise KVCacheExhausted("pool empty")
        return super().decode_step_batched(current_tokens, batched_kv, attention_mask, position_ids, sampling_per_row)

    def kv_reserve(self, prompt_len, max_tokens):
        if self.capacity is not None and self.reserved + prompt_len + max_tokens > self.capacity:
            return False
        self.reserved += prompt_len + max_tokens
        self.reservations.append(max_tokens)
        return True

    def kv_release(self, prompt_len, max_tokens):
        self.reserved -= prompt_len + max_tokens


async def _run(backend, prompts, max_tokens, async_decode=True, batch=8):
    sched = ContinuousBatchScheduler(backend, max_batch_size=batch, prefill_mode="mixed", prefill_step_tokens=256,
                                     async_decode=async_decode)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=p, max_tokens=m, session_id=f"s{i}", future=loop.create_future())
                for i, (p, m) in enumerate(zip(prompts, max_tokens))]
        return await asyncio.gather(*(sched.submit(r) for r in reqs)), sched, reqs
    finally:
        await sched.stop()


@pytest.mark.parametrize("async_decode", [False, True])
@pytest.mark.asyncio
async def test_a_preempted_row_is_requeued_and_its_output_is_unchanged(async_decode):
    prompts = [[1000 * (i + 1) + j for j in range(10 + i)] for i in range(4)]
    mts = [12, 9, 15, 7]
    ref, _, _ = await _run(ContinuingBackend(), prompts, mts, async_decode)
    b = ContinuingBackend(exhaust_at={4, 9})
    got, sched, reqs = await _run(b, prompts, mts, async_decode)
    assert got == ref == [[200 + j for j in range(m)] for m in mts]
    assert sum(r.preemptions for r in reqs) == 2 and sched.stats()["total_completed"] == 4
    resumed = [n for n in b.prefills if n not in {len(p) for p in prompts}]
    assert len(resumed) == 2                                  # each re-prefill fed prompt + what was generated
    assert sched._active_kv_reserved == 0 and b.reserved == 0


@pytest.mark.asyncio
async def test_admission_reserves_the_recent_90th_percentile_once_enough_requests_finished():
    n = OUT_MIN_SEEN + 8
    prompts = [[10_000 * (i + 1) + j for j in range(8)] for i in range(n)]
    mts = [512] * n
    b = ContinuingBackend()
    b.is_eos = lambda t: t == 204                              # every answer is 4 tokens long (EOS on the 5th)
    got, sched, _ = await _run(b, prompts, mts, batch=4)
    assert all(o == [200, 201, 202, 203] for o in got)
    assert b.reservations[0] == 512                           # nothing finished yet: max_tokens
    assert b.reservations[-1] == 4                            # then the observed length
    assert sched._active_kv_reserved == 0 and b.reserved == 0
