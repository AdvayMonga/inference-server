"""Tests for the /generate endpoint using a fake backend."""

from typing import Generator

import pytest
from httpx import ASGITransport, AsyncClient

from inference_server.backends.base import InferenceBackend
from inference_server.server import app


class FakeBackend(InferenceBackend):
    """Returns predictable tokens for testing server logic."""

    def load_model(self, model_name: str) -> None:
        pass

    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        return list(range(100, 100 + max_tokens))

    def generate_step(
        self, token_ids: list[int], kv_cache: object | None = None
    ) -> tuple[int, object]:
        return 100, None

    def stream(self, token_ids: list[int], max_tokens: int) -> Generator[int, None, None]:
        for i in range(max_tokens):
            yield 100 + i


@pytest.fixture(autouse=True)
def inject_fake_backend():
    """Inject fake backend and tokenizer into app.state before each test."""
    from inference_server.tokenizer import Tokenizer

    app.state.backend = FakeBackend()
    app.state.tokenizer = Tokenizer("gpt2", 100)
    yield
    del app.state.backend
    del app.state.tokenizer


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
    assert "ttft_ms" in data
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
    body = response.text
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_stream_contains_ttft(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 3, "stream": True}
    )
    body = response.text
    assert "ttft_ms" in body


@pytest.mark.asyncio
async def test_stream_false_returns_json(client):
    response = await client.post(
        "/generate", json={"text": "Hello world", "max_tokens": 3, "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
