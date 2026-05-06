"""Tests that chunked prefill (Version A) produces identical output to monolithic prefill."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Generator

import pytest
import torch

from inference_server.backends.base import InferenceBackend
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest


@dataclass
class _FakeKV:
    """Stand-in KV; tracks (batch, seq) shape and how many input tokens have been fed."""
    batch_size: int
    seq_len: int


class FakeChunkBackend(InferenceBackend):
    """Backend that implements both monolithic and chunked prefill primitives.

    First sampled token is always 200; decode advances tokens by +1 each step.
    Tracks total chunk-input tokens so tests can assert all chunks were fed.
    """

    FIRST_TOKEN = 200

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0
        self.chunk_input_tokens_total = 0  # cumulative input tokens through prefill_chunk
        self.store_calls: list[tuple[int, int]] = []  # (len(token_ids), matched)

    def load_model(self, model_name): pass

    def generate(self, token_ids, max_tokens, template_prefix_len=0, session_id="default"):
        return [self.FIRST_TOKEN + i for i in range(max_tokens)]

    def generate_batch(self, batch_token_ids, max_tokens, session_ids=None):
        return [[self.FIRST_TOKEN + i for i in range(mt)] for mt in max_tokens]

    def generate_step(self, token_ids, kv_cache=None):
        return self.FIRST_TOKEN, None

    def stream(self, token_ids, max_tokens, template_prefix_len=0, session_id="default"):
        for i in range(max_tokens):
            yield self.FIRST_TOKEN + i

    # --- Monolithic prefill ---
    def prefill(self, token_ids, session_id="default"):
        self.last_cache_hit_tokens = 0
        return _FakeKV(batch_size=1, seq_len=len(token_ids)), self.FIRST_TOKEN, len(token_ids)

    # --- Chunked prefill primitives ---
    def prefill_lookup(self, token_ids, session_id="default"):
        self.last_cache_hit_tokens = 0
        return None, 0

    def prefill_chunk(self, chunk_token_ids, partial_kv):
        prev_len = partial_kv.seq_len if partial_kv is not None else 0
        new_len = prev_len + len(chunk_token_ids)
        self.chunk_input_tokens_total += len(chunk_token_ids)
        return _FakeKV(batch_size=1, seq_len=new_len), self.FIRST_TOKEN, new_len

    def prefill_store(self, token_ids, full_kv, matched, session_id="default"):
        self.store_calls.append((len(token_ids), matched))

    # --- Batched decode ---
    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids):
        next_tokens = current_tokens.squeeze(-1) + 1
        batched_kv.seq_len += 1
        return next_tokens, batched_kv

    def stack_caches_left_padded(self, per_row_caches, max_kv_len):
        return _FakeKV(batch_size=len(per_row_caches), seq_len=max_kv_len)

    def remove_row_from_cache(self, batched_kv, row_idx):
        return _FakeKV(batch_size=batched_kv.batch_size - 1, seq_len=batched_kv.seq_len)

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        if batched_kv is None:
            return _FakeKV(batch_size=1, seq_len=new_kv_len)
        return _FakeKV(
            batch_size=batched_kv.batch_size + 1,
            seq_len=max(batched_kv.seq_len, new_kv_len),
        )

    def kv_length(self, kv): return kv.seq_len
    def is_eos(self, token_id): return False

    @property
    def device_str(self): return "cpu"


async def _run_one(backend, chunk_size, prompt_len, max_tokens):
    sched = ContinuousBatchScheduler(
        backend, max_batch_size=8, prefill_chunk_size=chunk_size,
    )
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        req = ScheduledRequest(
            token_ids=list(range(prompt_len)), max_tokens=max_tokens,
            session_id="t", future=loop.create_future(),
        )
        out = await sched.submit(req)
        return out, sched.stats()
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_chunked_matches_monolithic_output():
    """Chunked and monolithic must produce byte-identical generated tokens."""
    prompt_len, max_tokens = 25, 6

    out_mono, _ = await _run_one(FakeChunkBackend(), chunk_size=0,
                                  prompt_len=prompt_len, max_tokens=max_tokens)
    out_chunked, stats = await _run_one(FakeChunkBackend(), chunk_size=8,
                                         prompt_len=prompt_len, max_tokens=max_tokens)

    assert out_mono == out_chunked
    assert len(out_mono) == max_tokens
    assert stats["prefill_chunks_processed"] >= 1


@pytest.mark.asyncio
async def test_all_prompt_tokens_fed_through_chunks():
    """Every prompt token must pass through prefill_chunk exactly once."""
    backend = FakeChunkBackend()
    prompt_len = 25
    await _run_one(backend, chunk_size=8, prompt_len=prompt_len, max_tokens=3)
    assert backend.chunk_input_tokens_total == prompt_len


@pytest.mark.asyncio
async def test_chunk_count_matches_ceil_division():
    """ceil(prompt_len / chunk_size) chunks for a cold prefill."""
    import math
    backend = FakeChunkBackend()
    prompt_len, chunk_size = 25, 8
    _, stats = await _run_one(backend, chunk_size=chunk_size,
                               prompt_len=prompt_len, max_tokens=3)
    assert stats["prefill_chunks_processed"] == math.ceil(prompt_len / chunk_size)


@pytest.mark.asyncio
async def test_short_prompt_one_chunk():
    """Prompt shorter than chunk_size finishes prefill in a single chunk."""
    backend = FakeChunkBackend()
    _, stats = await _run_one(backend, chunk_size=64, prompt_len=10, max_tokens=3)
    assert stats["prefill_chunks_processed"] == 1


@pytest.mark.asyncio
async def test_prefill_store_called_once_on_completion():
    """Cache store fires exactly once per request, after the final chunk."""
    backend = FakeChunkBackend()
    await _run_one(backend, chunk_size=8, prompt_len=25, max_tokens=3)
    assert len(backend.store_calls) == 1
    assert backend.store_calls[0] == (25, 0)
