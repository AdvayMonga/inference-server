"""Tests for /generate using a fake backend that satisfies the scheduler's primitives."""

import asyncio
import threading
from dataclasses import dataclass
from typing import Generator

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from inference_server.backends.base import InferenceBackend
from inference_server.scheduler import ContinuousBatchScheduler
from inference_server.server import app


@dataclass
class _FakeKV:
    """Stand-in for a per-row or batched KV cache. Just tracks shape state."""
    batch_size: int
    seq_len: int


class FakeBackend(InferenceBackend):
    """Minimal backend: produces deterministic tokens, no real model."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0

    def load_model(self, model_name: str) -> None:
        pass

    # Legacy methods (unused now that server routes through scheduler, but required by ABC)
    def generate(self, token_ids, max_tokens, template_prefix_len=0, session_id="default"):
        return list(range(100, 100 + max_tokens))

    def generate_batch(self, batch_token_ids, max_tokens, session_ids=None):
        return [list(range(100, 100 + mt)) for mt in max_tokens]

    def generate_step(self, token_ids, kv_cache=None):
        return 100, None

    def stream(self, token_ids, max_tokens, template_prefix_len=0, session_id="default"):
        for i in range(max_tokens):
            yield 100 + i

    # Continuous-batching primitives
    def prefill(self, token_ids, session_id="default"):
        self.last_cache_hit_tokens = 0
        kv = _FakeKV(batch_size=1, seq_len=len(token_ids))
        return kv, 100, len(token_ids)

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids):
        # Each row produces (current + 1) as next token, deterministic
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

    def kv_length(self, kv):
        return kv.seq_len

    def is_eos(self, token_id):
        return False  # rely on max_tokens to terminate

    @property
    def device_str(self):
        return "cpu"


@pytest.fixture(autouse=True)
async def inject_fake_backend():
    from inference_server.kv_cache.cache_manager import CacheManager
    from inference_server.tokenizer import Tokenizer

    backend = FakeBackend()
    cache_adapter = CacheManager(num_blocks=20, block_size=4)
    backend.set_cache_adapter(cache_adapter)

    scheduler = ContinuousBatchScheduler(backend, max_batch_size=8)
    scheduler.start()

    app.state.backend = backend
    app.state.tokenizer = Tokenizer("gpt2", 100)
    app.state.scheduler = scheduler
    app.state.cache_adapter = cache_adapter

    yield

    await scheduler.stop()
    del app.state.backend
    del app.state.tokenizer
    del app.state.scheduler
    del app.state.cache_adapter


@pytest.fixture
async def client():
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Non-streaming tests ---

@pytest.mark.asyncio
async def test_generate_success(client):
    response = await client.post("/generate", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["tokens_generated"] > 0


@pytest.mark.asyncio
async def test_generate_custom_max_tokens(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 5}
    )
    assert response.status_code == 200
    assert response.json()["tokens_generated"] == 5


@pytest.mark.asyncio
async def test_generate_missing_text(client):
    response = await client.post("/generate", json={})
    assert response.status_code == 422


# --- Streaming tests ---

@pytest.mark.asyncio
async def test_stream_returns_sse(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 3, "stream": True}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_contains_done(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 3, "stream": True}
    )
    assert "data: [DONE]" in response.text


@pytest.mark.asyncio
async def test_stream_contains_ttft(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 3, "stream": True}
    )
    assert "ttft_ms" in response.text


# --- Concurrency tests ---

@pytest.mark.asyncio
async def test_concurrent_requests_return_correct_results(client):
    async def send_request(text, max_tokens):
        resp = await client.post(
            "/generate", json={"text": text, "max_tokens": max_tokens}
        )
        return resp.json()

    results = await asyncio.gather(
        send_request("Hello", 3),
        send_request("Hello world", 5),
        send_request("Hi", 2),
    )

    assert results[0]["tokens_generated"] == 3
    assert results[1]["tokens_generated"] == 5
    assert results[2]["tokens_generated"] == 2


@pytest.mark.asyncio
async def test_concurrent_requests_no_cross_contamination(client):
    results = await asyncio.gather(
        client.post("/generate", json={"text": "A", "max_tokens": 3}),
        client.post("/generate", json={"text": "B", "max_tokens": 3}),
    )
    assert results[0].status_code == 200
    assert results[1].status_code == 200
    assert len(results[0].json()["text"]) > 0
    assert len(results[1].json()["text"]) > 0
