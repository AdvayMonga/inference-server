"""The instrument-side contract: panels carry provenance, and the regime is named not implied."""

from __future__ import annotations

import pytest

from inference_server.research import harness as H
from inference_server.research.compare import comparable
from inference_server.research.schemas import SchemaError, Vitals

CFG = {"model": "E4B", "gpu": "A100", "max_batch_size": 256, "pool_size": 4000,
       "rates": "2", "compile": "0"}


def test_regime_is_named_from_the_measured_hit_rate():
    """POOL_SIZE=64 against a warm cache is a cache-HIT benchmark; the panel has to say so."""
    assert H.infer_regime(0.85) == "cache_hit_heavy"
    assert H.infer_regime(0.02) == "cache_miss_heavy"
    assert H.infer_regime(0.30) == "mixed"
    assert H.infer_regime(None) == "synthetic"


def test_panel_from_engine_stats_carries_pressure_and_cache_signals():
    sched = {"wave_sizes": {1: 174, 2: 16}, "kv_admit_blocked": 5, "total_expired": 2,
             "total_preempted": 0, "total_iteration_errors": 0, "pending_high_water": 12}
    cache = {"hit_rate": 0.02, "lookups": 400, "entries": 25, "evictions": 375,
             "blocks_held": 100, "max_blocks": 100, "pool_free_blocks": 100,
             "pool_total_blocks": 200, "pool_utilization": 0.5}
    v = H.build_validity("bench_serving", CFG, n_samples=50,
                         workload_regime=H.infer_regime(cache["hit_rate"]), stderr_value=1.2)
    panel = H.panel_from_stats(v, scheduler_stats=sched, cache_stats=cache, ttft_p95=204.0)
    panel.validate()
    assert panel.wave_sizes == {"1": 174, "2": 16}       # keys normalised to str for JSON
    assert panel.total_expired == 2
    assert panel.cache_hit_rate == 0.02
    assert panel.validity.workload_regime == "cache_miss_heavy"


def test_arms_from_one_process_are_comparable_and_others_are_not(monkeypatch):
    a = H.panel_from_stats(H.build_validity("bench_serving", CFG, n_samples=5,
                                            workload_regime="mixed", stderr_value=1.0))
    b = H.panel_from_stats(H.build_validity("bench_serving", CFG, n_samples=5,
                                            workload_regime="mixed", stderr_value=1.0))
    assert comparable(a, b), "arms measured in one process must be comparable"

    monkeypatch.setattr(H, "_RUN_GROUP", "a-different-session")
    c = H.panel_from_stats(H.build_validity("bench_serving", CFG, n_samples=5,
                                            workload_regime="mixed", stderr_value=1.0))
    assert not comparable(a, c), "panels from another session must not be comparable"


def test_emit_refuses_a_panel_without_provenance(tmp_path):
    v = H.build_validity("bench_serving", {}, n_samples=1, workload_regime="mixed")
    with pytest.raises(SchemaError, match="harness_config"):
        H.emit(Vitals(validity=v), runs_dir=tmp_path)
    assert not list(tmp_path.glob("*.json")), "an invalid panel must not reach disk"


def test_emit_writes_a_loadable_record(tmp_path):
    v = H.build_validity("bench_serving", CFG, n_samples=3, workload_regime="mixed",
                         stderr_value=0.5)
    path = H.emit(H.panel_from_stats(v, ttft_p95=204.0, tok_s_within_slo=121.0),
                  runs_dir=tmp_path)
    back = Vitals.load(path)
    assert back.ttft_p95 == 204.0 and back.validity.harness == "bench_serving"


def test_stderr_needs_more_than_one_sample():
    assert H.stderr([1.0]) is None
    assert H.stderr([1.0, 2.0, 3.0]) is not None
