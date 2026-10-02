"""Canaries are off by default, refuse unknown names, and each one changes the behaviour it names."""

import time
from types import SimpleNamespace

import pytest

from inference_server import canary


@pytest.fixture
def set_canary(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("INFERENCE_SERVER_CANARY", raising=False)
        else:
            monkeypatch.setenv("INFERENCE_SERVER_CANARY", value)
        return canary.reload()
    yield _set
    monkeypatch.delenv("INFERENCE_SERVER_CANARY", raising=False)
    canary.reload()


def test_off_by_default_and_unknown_refused(set_canary):
    assert set_canary(None) == frozenset()
    assert set_canary("kv_leak, admit_fewer") == {"kv_leak", "admit_fewer"}
    with pytest.raises(ValueError):
        set_canary("make_it_fast")


def test_admit_fewer_shrinks_the_batch(set_canary):
    from inference_server.scheduler import ContinuousBatchScheduler
    backend = SimpleNamespace(device="cpu")
    set_canary(None)
    assert ContinuousBatchScheduler(backend, max_batch_size=8).max_batch_size == 8
    set_canary("admit_fewer")
    assert ContinuousBatchScheduler(backend, max_batch_size=8).max_batch_size == 7


def test_kv_leak_never_frees_reservations(set_canary):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    fake = SimpleNamespace(pools=[object()], _reserved=[10], _kv_footprints=lambda p, m: [4])
    set_canary(None)
    CustomTorchBackend.kv_release(fake, 8, 8)
    assert fake._reserved == [6]
    set_canary("kv_leak")
    CustomTorchBackend.kv_release(fake, 8, 8)
    assert fake._reserved == [6]


def test_slow_decode_adds_time_per_step(set_canary, monkeypatch):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    fake = SimpleNamespace(device=SimpleNamespace(type="none"))
    set_canary("slow_decode")
    with pytest.raises(Exception):      # the fake stops the step right after the canary
        CustomTorchBackend.decode_step_batched(fake, None, None, None, None)
    assert slept == [0.003]
