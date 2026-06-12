"""Regression: scheduler mask bookkeeping must not assume mask width == kv_length.

Paged backends report kv_length = max(per-row seq_len), which DROPS when the longest row
is evicted. The attention-mask width doesn't shrink, so a later admit used to cat
mismatched widths and crash the worker (mixed-workload crash, 2026-06-12). CPU-only —
pure scheduler logic, no model.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from inference_server.backends.base import InferenceBackend
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest


class FakePagedBackend(InferenceBackend):
    """batched_kv = list of per-row lengths; kv_length = max (drops on eviction)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0

    def load_model(self, model_name): pass
    def generate(self, *a, **k): return []
    def generate_batch(self, *a, **k): return []
    def generate_step(self, *a, **k): return 0, None
    def stream(self, *a, **k):
        if False:
            yield 0

    def prefill(self, token_ids, session_id="default"):
        self.last_cache_hit_tokens = 0
        return len(token_ids), 1, len(token_ids)   # (per-row kv == its length, first_token, kv_len)

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids,
                            sampling_per_row=None):
        for i in range(len(batched_kv)):
            batched_kv[i] += 1
        next_tokens = current_tokens.squeeze(-1) + 1
        return next_tokens, batched_kv

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        if batched_kv is None:
            return [new_kv]
        batched_kv.append(new_kv)
        return batched_kv

    def remove_row_from_cache(self, batched_kv, row_idx):
        batched_kv.pop(row_idx)
        return batched_kv

    def kv_length(self, kv):
        return max(kv) if kv else 0          # paged semantic: drops when longest row leaves

    def is_eos(self, token_id): return False

    @property
    def device_str(self): return "cpu"


@pytest.mark.asyncio
async def test_admit_after_longest_row_evicted_does_not_crash():
    """Long row finishes (kv_length drops), then a new row is admitted — must not crash."""
    sched = ContinuousBatchScheduler(FakePagedBackend(), max_batch_size=2)
    sched.start()
    try:
        loop = asyncio.get_running_loop()

        def mk(plen, mx):
            return ScheduledRequest(token_ids=list(range(plen)), max_tokens=mx,
                                    session_id="s", future=loop.create_future())

        # Batch fills with A (long, finishes after 1 token) + B (short, keeps going).
        # When A evicts, kv_length drops 50→~6 while the mask stays wide; admitting C
        # then exercises the splice path that used to crash.
        a, b, c = mk(50, 1), mk(5, 12), mk(5, 12)
        results = await asyncio.gather(
            sched.submit(a), sched.submit(b), sched.submit(c),
            return_exceptions=True,
        )
        for r in results:
            assert not isinstance(r, Exception), f"scheduler crashed: {r!r}"
        assert len(results[0]) == 1 and len(results[1]) == 12 and len(results[2]) == 12
    finally:
        await sched.stop()
