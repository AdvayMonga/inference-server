"""Tests for the /generate endpoint using a fake backend."""

import asyncio
from typing import Generator

import pytest
from httpx import ASGITransport, AsyncClient

from inference_server.backends.base import InferenceBackend
from inference_server.batcher import BatchProcessor
from inference_server.server import app


class FakeBackend(InferenceBackend):
    """Returns predictable tokens for testing server logic."""

    def load_model(self, model_name: str) -> None:
        pass

    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        return list(range(100, 100 + max_tokens))

    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        return [
            list(range(100 + len(ids), 100 + len(ids) + mt))
            for ids, mt in zip(batch_token_ids, max_tokens)
        ]

    def generate_step(
        self, token_ids: list[int], kv_cache: object | None = None
    ) -> tuple[int, object]:
        return 100, None

    def stream(self, token_ids: list[int], max_tokens: int) -> Generator[int, None, None]:
        for i in range(max_tokens):
            yield 100 + i


@pytest.fixture(autouse=True)
async def inject_fake_backend():
    """Inject fake backend, tokenizer, batcher, and cache manager into app.state."""
    from inference_server.kv_cache.cache_manager import CacheManager
    from inference_server.tokenizer import Tokenizer

    backend = FakeBackend()
    cache_manager = CacheManager(num_blocks=20, block_size=4, eviction_policy="lru")
    backend.set_cache_manager(cache_manager)

    batcher = BatchProcessor(backend)
    batcher.start()

    app.state.backend = backend
    app.state.tokenizer = Tokenizer("gpt2", 100)
    app.state.batcher = batcher
    app.state.cache_manager = cache_manager

    yield

    await batcher.stop()
    del app.state.backend
    del app.state.tokenizer
    del app.state.batcher
    del app.state.cache_manager


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
    assert "tokens_generated" in data
    assert data["tokens_generated"] > 0
    assert "total_ms" in data


@pytest.mark.asyncio
async def test_generate_custom_max_tokens(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tokens_generated"] == 5


@pytest.mark.asyncio
async def test_generate_empty_input(client):
    response = await client.post("/generate", json={"text": ""})
    assert response.status_code == 500


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


@pytest.mark.asyncio
async def test_stream_false_returns_json(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 3, "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data


# --- Batching tests ---

@pytest.mark.asyncio
async def test_concurrent_requests_return_correct_results(client):
    """Multiple concurrent requests should each get their own result."""
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
    """Different requests must never receive each other's tokens."""
    results = await asyncio.gather(
        client.post("/generate", json={"text": "A", "max_tokens": 3}),
        client.post("/generate", json={"text": "B", "max_tokens": 3}),
    )

    assert results[0].status_code == 200
    assert results[1].status_code == 200
    assert len(results[0].json()["text"]) > 0
    assert len(results[1].json()["text"]) > 0
