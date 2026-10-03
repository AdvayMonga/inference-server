"""lab.canary patches the engine from outside: unknown names refused, each canary changes the behaviour it names."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from lab import canary


@pytest.fixture
def engine(monkeypatch):
    """Fresh class attributes per test, since apply() patches them in place."""
    from inference_server.backends.custom_torch_backend import CustomTorchBackend as B
    from inference_server.scheduler import ContinuousBatchScheduler as S
    for cls, names in ((B, ("load_model", "decode_step_batched", "kv_release")), (S, ("_admit_pending",))):
        for n in names:
            monkeypatch.setattr(cls, n, getattr(cls, n))
    return B, S


def test_parse_refuses_unknown_names():
    assert canary.parse("kv_leak, admit_fewer") == {"kv_leak", "admit_fewer"} and canary.parse("") == frozenset()
    with pytest.raises(ValueError):
        canary.parse("make_it_fast")


def test_kv_leak_never_frees_reservations(engine):
    B, _ = engine
    fake = SimpleNamespace(pools=[object()], _reserved=[10], _kv_footprints=lambda p, m: [4])
    B.kv_release(fake, 8, 8)
    assert fake._reserved == [6]
    canary.apply(frozenset({"kv_leak"}))
    B.kv_release(fake, 8, 8)
    assert fake._reserved == [6]


def test_slow_decode_and_slow_start_add_time(engine, monkeypatch):
    B, _ = engine
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(B, "decode_step_batched", lambda self, *a, **k: "stepped")
    monkeypatch.setattr(B, "load_model", lambda self, name: None)
    canary.apply(frozenset({"slow_decode", "slow_start"}))
    assert B.decode_step_batched(object(), None, None, None, None) == "stepped"
    B.load_model(object(), "m")
    assert slept == [canary.SLOW_DECODE_S, canary.SLOW_START_S]


def test_admit_fewer_admits_one_less_but_reports_the_config(engine, monkeypatch):
    _, S = engine
    seen = []
    monkeypatch.setattr(S, "_admit_pending", lambda self, device: seen.append(self.max_batch_size))
    canary.apply(frozenset({"admit_fewer"}))
    sched = SimpleNamespace(max_batch_size=8)
    S._admit_pending(sched, "cpu")
    assert seen == [7] and sched.max_batch_size == 8
    one = SimpleNamespace(max_batch_size=1)
    S._admit_pending(one, "cpu")
    assert seen[-1] == 0          # not inert at 1: nothing admits, which a harness must notice
