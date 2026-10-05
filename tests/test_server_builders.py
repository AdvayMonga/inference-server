"""The engine builders in server.py: plain functions the lab shares with lifespan, which stays a context manager."""

from __future__ import annotations

import inspect

from inference_server.config import Settings
from inference_server.server import build_scheduler, lifespan
from tests.stub_backend import StubBackend


def test_build_scheduler_applies_settings():
    sched = build_scheduler(StubBackend(), Settings(max_batch_size=3, max_queue_size=7))
    assert sched.max_batch_size == 3 and sched.max_queue_size == 7
    assert sched.stats()["timeline"]["enabled"] is False


def test_lifespan_is_an_async_context_manager():
    cm = lifespan(object())
    assert hasattr(cm, "__aenter__") and not inspect.iscoroutine(cm)


def test_unset_prefill_mode_is_batched_where_the_backend_supports_it():
    class Batching(StubBackend):
        def prefill_batch(self, *a, **k):
            raise NotImplementedError

    assert build_scheduler(Batching(), Settings()).prefill_mode == "batched"
    assert build_scheduler(Batching(), Settings(prefill_chunk_size=256)).prefill_mode == "chunked"
    assert build_scheduler(Batching(), Settings(prefill_mode="monolithic")).prefill_mode == "monolithic"
    assert build_scheduler(StubBackend(), Settings()).prefill_mode == "monolithic"
