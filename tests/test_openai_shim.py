"""OpenAI shim — request/response translation for both surfaces, no model load.

Stubs the scheduler + tokenizer so we exercise the real router paths (non-stream, stream,
usage, finish_reason, 429) without a GPU. End-to-end vs the real engine is P3 (guidellm)."""

import asyncio
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_server.openai_shim import router
from inference_server.scheduler import QueueFullError


class FakeTokenizer:
    """1 char = 1 token; decode_token echoes the char. Deterministic, no HF load."""
    def encode(self, text):
        if not text:
            raise ValueError("empty")
        return [ord(c) for c in text]

    def encode_messages(self, messages, thinking=True):
        if not messages:
            raise ValueError("No messages provided")
        return self.encode("".join(m["content"] for m in messages))

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def decode_token(self, tid):
        return chr(tid)


class FakeScheduler:
    """Returns a fixed 3-token completion; streams it through the request's token_queue."""
    def __init__(self, full=False):
        self.full = full
        self.out = [ord("a"), ord("b"), ord("c")]

    def submit(self, req):
        if self.full:
            raise QueueFullError("queue full")
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(self.out)
        return fut

    def enqueue(self, req):
        if self.full:
            raise QueueFullError("queue full")
        for t in self.out:
            req.token_queue.put_nowait(t)
        req.token_queue.put_nowait(None)


def _app(scheduler):
    app = FastAPI()
    app.include_router(router)
    app.state.scheduler = scheduler
    app.state.tokenizer = FakeTokenizer()
    app.state.model_name = "test-model"
    return app


def test_non_streaming_completion():
    client = TestClient(_app(FakeScheduler()))
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 10})
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "text_completion"
    assert d["choices"][0]["text"] == "abc"
    assert d["choices"][0]["finish_reason"] == "stop"       # 3 < 10 max
    assert d["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_finish_reason_length():
    client = TestClient(_app(FakeScheduler()))
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
    assert r.json()["choices"][0]["finish_reason"] == "length"  # 3 == 3 max


def test_streaming_completion():
    client = TestClient(_app(FakeScheduler()))
    with client.stream("POST", "/v1/completions",
                       json={"prompt": "hello", "max_tokens": 10, "stream": True}) as r:
        assert r.status_code == 200
        chunks = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    tokens = [json.loads(c)["choices"][0]["text"] for c in chunks[:3]]
    assert tokens == ["a", "b", "c"]                         # one chunk per token
    finish = json.loads(chunks[3])["choices"][0]["finish_reason"]
    assert finish == "stop"
    usage = json.loads(chunks[4])["usage"]                   # penultimate = usage chunk
    assert usage["completion_tokens"] == 3


def test_queue_full_429():
    client = TestClient(_app(FakeScheduler(full=True)))
    r = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 5})
    assert r.status_code == 429


def test_models_endpoint():
    client = TestClient(_app(FakeScheduler()))
    d = client.get("/v1/models").json()
    assert d["data"][0]["id"] == "test-model"


# ------------------------------------------------------------------ chat completions

def test_non_streaming_chat_completion():
    client = TestClient(_app(FakeScheduler()))
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 10})
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["id"].startswith("chatcmpl-")
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "abc"}
    assert d["choices"][0]["finish_reason"] == "stop"
    assert d["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_chat_uses_every_turn_not_just_the_last():
    """A frontend sends the whole history each turn; dropping it silently loses context."""
    client = TestClient(_app(FakeScheduler()))
    d = client.post("/v1/chat/completions", json={"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ], "max_tokens": 10}).json()
    assert d["usage"]["prompt_tokens"] == len("sys") + len("one") + len("two") + len("three")


def test_max_completion_tokens_wins_over_max_tokens():
    """max_tokens is the deprecated spelling; newer clients send both."""
    client = TestClient(_app(FakeScheduler()))
    d = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100, "max_completion_tokens": 3}).json()
    assert d["choices"][0]["finish_reason"] == "length"      # 3 generated == budget of 3


def test_streaming_chat_completion():
    client = TestClient(_app(FakeScheduler()))
    with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 10, "stream": True}) as r:
        assert r.status_code == 200
        chunks = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    parsed = [json.loads(c) for c in chunks[:-1]]
    assert all(p["object"] == "chat.completion.chunk" for p in parsed)
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"   # role chunk first
    assert [p["choices"][0]["delta"]["content"] for p in parsed[1:4]] == ["a", "b", "c"]
    assert parsed[4]["choices"][0]["finish_reason"] == "stop"
    assert parsed[5]["usage"]["completion_tokens"] == 3


def test_chat_empty_messages_400():
    client = TestClient(_app(FakeScheduler()))
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_chat_queue_full_429():
    client = TestClient(_app(FakeScheduler(full=True)))
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert r.status_code == 429
