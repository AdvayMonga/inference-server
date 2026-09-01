"""De-Pythoned decode step: batched sampling + skipping the unused attention mask.

The per-step cost used to scale with batch size for reasons unrelated to the model: a
Python loop calling the sampler once per row over a [1, 262144] logit slice, N separate
.item() device syncs, and a growing [B, S] mask cat that paged backends never read.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
import torch

from inference_server.backends.base import InferenceBackend
from inference_server.sampling import SamplingParams, sample, sample_batched
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest


def _per_row(logits, params):
    """The old implementation, kept here as the reference to match."""
    return torch.stack([sample(logits[i], params[i]) for i in range(len(params))])


def test_all_greedy_matches_per_row_loop():
    torch.manual_seed(0)
    logits = torch.randn(8, 512)
    params = [SamplingParams.greedy()] * 8
    assert torch.equal(sample_batched(logits, params), _per_row(logits, params))


def test_mixed_groups_match_per_row_loop():
    """top_k=1 makes the sampling path deterministic (multinomial over one candidate),
    so a mixed batch can be compared exactly against the per-row reference."""
    torch.manual_seed(0)
    logits = torch.randn(6, 512)
    a = SamplingParams.greedy()
    b = SamplingParams(temperature=0.7, top_k=1)
    c = SamplingParams(temperature=1.5, top_k=1)
    params = [a, b, c, b, a, c]
    assert torch.equal(sample_batched(logits, params), _per_row(logits, params))


def test_grouping_preserves_row_order():
    """Rows must come back in input order, not grouped order."""
    logits = torch.full((4, 10), -1e9)
    for i in range(4):
        logits[i, i * 2] = 0.0          # row i's argmax is 2i
    params = [SamplingParams.greedy(), SamplingParams(temperature=1.0, top_k=1)] * 2
    assert sample_batched(logits, params).tolist() == [0, 2, 4, 6]


def test_empty_batch():
    out = sample_batched(torch.randn(0, 10), [])
    assert out.shape == (0,) and out.dtype == torch.long


class _FakeBackend(InferenceBackend):
    """Minimal paged-style backend; `needs_attention_mask` is set per-subclass."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0
        self.seen_masks = []

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

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids,
                            sampling_per_row=None):
        self.seen_masks.append(attention_mask)
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


class _MaskFreeBackend(_FakeBackend):
    needs_attention_mask = False


async def _run(backend, n_rows=3, max_tokens=6):
    sched = ContinuousBatchScheduler(backend, max_batch_size=n_rows)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(range(4 + i)), max_tokens=max_tokens,
                                 session_id=f"s{i}", future=loop.create_future())
                for i in range(n_rows)]
        out = await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)
        for r in out:
            assert not isinstance(r, Exception), f"scheduler crashed: {r!r}"
        return sched, out
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_mask_free_backend_never_builds_a_mask():
    backend = _MaskFreeBackend()
    sched, out = await _run(backend)
    assert all(len(o) == 6 for o in out)              # generation still correct
    assert sched._attention_mask is None
    assert all(m is None for m in backend.seen_masks)


@pytest.mark.asyncio
async def test_default_backend_still_gets_a_mask():
    """Guard: the HF backend consumes the mask — it must not be silently disabled for it."""
    backend = _FakeBackend()
    sched, out = await _run(backend)
    assert all(len(o) == 6 for o in out)
    assert any(m is not None for m in backend.seen_masks)
    widest = max(m.shape for m in backend.seen_masks if m is not None)
    assert widest[0] == 3                              # one mask row per active row
