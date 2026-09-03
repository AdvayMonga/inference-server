"""KV pressure: the cache must give blocks back, and a bad step must not kill the server.

Two production-breaking behaviours this pins down:

1. PrefixCache held a refcount on every prefix it ever saw, forever. Distinct prompts therefore
   leaked blocks permanently until BlockPool.alloc() raised RuntimeError on the request path —
   a crash where backpressure belongs.
2. Any exception in a scheduler iteration escaped to _fail_all, which cleared _active WITHOUT
   routing through remove_row_from_cache and then let the worker thread exit. The server would
   go on accepting requests and running none of them, with the leaked blocks unrecoverable.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
import torch

from inference_server.backends.base import InferenceBackend
from inference_server.models.paged_kv_cache import BlockPool, PagedKVCache, PrefixCache
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest

BS = 16


def _pool(n_blocks=64):
    return BlockPool(n_blocks, BS, 1, 8, torch.float32, torch.device("cpu"))


def _run_prompt(pools, pc, prompt):
    m, shared = pc.lookup(prompt)
    c = PagedKVCache(pools=pools, shared_prefix=shared, shared_prefix_tokens=m)
    n = len(prompt) - m
    if n > 0:
        c.append(0, torch.zeros(1, 1, n, 8), torch.zeros(1, 1, n, 8))
    pc.store(prompt, c.block_tables)
    c.free_all()
    return m


# ---------------------------------------------------------------- prefix cache

def test_distinct_prompts_do_not_exhaust_the_pool():
    """The regression: every distinct prompt used to leak its blocks permanently."""
    pools = [_pool(64)]
    pc = PrefixCache(pools)
    for i in range(500):                       # far more prompts than the pool can hold
        _run_prompt(pools, pc, list(range(i * 999, i * 999 + 64)))
    assert pools[0].free_count > 0
    assert pc.evictions > 0


def test_cache_stays_under_its_block_watermark():
    """Reclaim-on-demand alone would let the cache own 100% of the pool, leaving no headroom."""
    pools = [_pool(64)]
    pc = PrefixCache(pools, max_block_fraction=0.5)
    for i in range(200):
        _run_prompt(pools, pc, list(range(i * 999, i * 999 + 64)))
    st = pc.stats()
    assert st["blocks_held"] <= st["max_blocks"] == 32
    assert pools[0].free_count >= 32


def test_recent_prefixes_still_hit_after_eviction():
    pools = [_pool(64)]
    pc = PrefixCache(pools)
    prompt = list(range(500_000, 500_000 + 64))
    _run_prompt(pools, pc, prompt)
    for i in range(3):                          # a little churn, not enough to evict it
        _run_prompt(pools, pc, list(range(i * 999, i * 999 + 32)))
    assert _run_prompt(pools, pc, prompt) == 64


def test_lru_evicts_the_least_recently_used():
    pools = [_pool(64)]
    pc = PrefixCache(pools, max_entries=3)
    a, b, c = (list(range(k * 999, k * 999 + 32)) for k in (1, 2, 3))
    for p in (a, b, c):
        _run_prompt(pools, pc, p)
    _run_prompt(pools, pc, a)                   # touch `a` so `b` is now the oldest
    _run_prompt(pools, pc, list(range(9 * 999, 9 * 999 + 32)))   # forces one eviction
    assert _run_prompt(pools, pc, a) == 32      # survived
    assert _run_prompt(pools, pc, b) == 0       # evicted


def test_alloc_reclaims_instead_of_raising():
    """BlockPool asks its reclaimer before declaring exhaustion."""
    pools = [_pool(8)]
    pc = PrefixCache(pools, max_block_fraction=1.0)
    _run_prompt(pools, pc, list(range(0, 8 * BS)))       # cache now owns the whole pool
    assert pools[0].free_count == 0
    assert pools[0].alloc() >= 0                          # must reclaim, not raise
    assert pc.evictions > 0


def test_reclaim_reports_blocks_freed():
    pools = [_pool(32)]
    pc = PrefixCache(pools)
    for i in range(4):
        _run_prompt(pools, pc, list(range(i * 999, i * 999 + 64)))
    before = pools[0].free_count
    assert pc.reclaim(1) > 0
    assert pools[0].free_count > before


# ---------------------------------------------------------------- scheduler

class _FlakyBackend(InferenceBackend):
    """Fails `fail_steps` decode steps, then works. Tracks live rows to prove KV is released."""

    needs_attention_mask = False

    def __init__(self, fail_steps=1, fail_exc=None):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0
        self.fail_steps = fail_steps
        self.fail_exc = fail_exc or RuntimeError("BlockPool exhausted (test)")
        self.live_rows = 0
        self.steps = 0

    def load_model(self, model_name): pass
    def generate(self, *a, **k): return []
    def generate_batch(self, *a, **k): return []
    def generate_step(self, *a, **k): return 0, None
    def stream(self, *a, **k):
        if False:
            yield 0

    def prefill(self, token_ids, session_id="default"):
        self.last_cache_hit_tokens = 0
        return len(token_ids), 1, len(token_ids)

    def prefill_batch(self, prompts, session_ids=None):
        self.last_cache_hit_tokens = 0
        return [(len(p), 1, len(p)) for p in prompts]

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids,
                            sampling_per_row=None):
        self.steps += 1
        if self.steps <= self.fail_steps:
            raise self.fail_exc
        for i in range(len(batched_kv)):
            batched_kv[i] += 1
        return current_tokens.squeeze(-1) + 1, batched_kv

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        self.live_rows += 1
        if batched_kv is None:
            return [new_kv]
        batched_kv.append(new_kv)
        return batched_kv

    def remove_row_from_cache(self, batched_kv, row_idx):
        self.live_rows -= 1
        batched_kv.pop(row_idx)
        return batched_kv

    def kv_length(self, kv): return max(kv) if kv else 0
    def is_eos(self, token_id): return False

    @property
    def device_str(self): return "cpu"


async def _submit(sched, n, max_tokens=3, plen=8):
    loop = asyncio.get_running_loop()
    reqs = [ScheduledRequest(token_ids=list(range(plen)), max_tokens=max_tokens,
                             session_id=f"s{i}", future=loop.create_future())
            for i in range(n)]
    return await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)


@pytest.mark.asyncio
async def test_worker_survives_a_failing_step_and_keeps_serving():
    backend = _FlakyBackend(fail_steps=1)
    sched = ContinuousBatchScheduler(backend, max_batch_size=2, prefill_mode="batched")
    sched.start()
    try:
        first = await _submit(sched, 1)
        assert isinstance(first[0], Exception)          # the failing step drops it
        later = await _submit(sched, 3)                 # worker must still be alive
        assert all(not isinstance(r, Exception) and len(r) == 3 for r in later), later
    finally:
        await sched.stop()
    assert sched._total_iteration_errors >= 1


@pytest.mark.asyncio
async def test_dropped_rows_release_their_kv():
    """_fail_all used to clear _active without freeing blocks — the pool never recovered."""
    backend = _FlakyBackend(fail_steps=1)
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, prefill_mode="batched")
    sched.start()
    try:
        await _submit(sched, 2)
        await _submit(sched, 2)
    finally:
        await sched.stop()
    assert backend.live_rows == 0, f"{backend.live_rows} rows leaked their KV"


@pytest.mark.asyncio
async def test_pressure_preempts_the_newest_row_not_the_batch():
    """With several rows in flight, one bad step should cost ONE request, not all of them."""
    backend = _FlakyBackend(fail_steps=1)
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, prefill_mode="batched")
    sched.start()
    try:
        out = await _submit(sched, 4, max_tokens=4)
    finally:
        await sched.stop()
    failed = [r for r in out if isinstance(r, Exception)]
    assert len(failed) == 1, f"expected exactly one casualty, got {len(failed)}"
    assert sched._total_preempted == 1
    assert backend.live_rows == 0


@pytest.mark.asyncio
async def test_stale_queued_requests_are_shed_not_served():
    """Overload must become rejection, not unbounded latency. Without a deadline the queue
    reached 1001 deep under stress and end-to-end p95 hit 105s while almost nothing was
    rejected — the server looked healthy while being useless."""
    import time as _time

    backend = _FlakyBackend(fail_steps=0)
    sched = ContinuousBatchScheduler(backend, max_batch_size=1, prefill_mode="batched",
                                     max_queue_wait_s=0.05)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        # Backdate arrivals so they are already past the deadline when admission looks at them.
        reqs = []
        for i in range(6):
            r = ScheduledRequest(token_ids=list(range(8)), max_tokens=2,
                                 session_id=f"s{i}", future=loop.create_future())
            sched.enqueue(r)
            r.enqueue_ts = _time.perf_counter() - 10.0
            reqs.append(r)
        out = await asyncio.gather(*(r.future for r in reqs), return_exceptions=True)
    finally:
        await sched.stop()
    from inference_server.scheduler import QueueFullError
    shed = [r for r in out if isinstance(r, QueueFullError)]
    assert shed, "nothing was shed despite every request being past the deadline"
    assert sched._total_expired == len(shed)
    assert all("deadline" in str(r) for r in shed)


@pytest.mark.asyncio
async def test_deadline_off_admits_everything():
    backend = _FlakyBackend(fail_steps=0)
    sched = ContinuousBatchScheduler(backend, max_batch_size=2, prefill_mode="batched",
                                     max_queue_wait_s=0)
    sched.start()
    try:
        out = await _submit(sched, 4, max_tokens=2)
    finally:
        await sched.stop()
    assert all(not isinstance(r, Exception) for r in out)
    assert sched._total_expired == 0
