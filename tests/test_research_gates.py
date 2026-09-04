"""The eval gates. Each test is a real failure mode this project hit — the gates exist so the
loop cannot repeat them unattended.
"""

from __future__ import annotations

import pytest

from inference_server.research.gates import (
    correctness_gate,
    cost_gate,
    judge,
    significance_gate,
    validity_gate,
)
from inference_server.research.schemas import Hypothesis, Validity, Vitals


def _v(**over):
    base = dict(engine_sha="abc", dirty=False, harness="bench_serving",
                harness_config={"model": "E4B", "max_batch_size": 256, "pool_size": 4000},
                workload_regime="cache_miss_heavy", n_samples=5, run_group="g1", stderr=1.0)
    base.update(over)
    return Validity(**base)


def _panel(**over):
    v = over.pop("validity", None) or _v()
    d = dict(wave_sizes={"1": 100, "4": 20}, ttft_prefill_p95=200.0, ttft_queue_p95=20.0,
             cache_lookups=500, total_iteration_errors=0)
    d.update(over)
    return Vitals(validity=v, **d)


def _hyp(**over):
    base = dict(statement="s", gap_id="prefill-tail", predicted_metric="ttft_p95",
                predicted_direction="decrease", predicted_magnitude=">20%",
                falsification_tier=3, falsification_test="probe")
    base.update(over)
    return Hypothesis(**base)


# ---------------------------------------------------------------- validity (judged FIRST)

def test_validity_rejects_a_harness_that_never_exercised_the_change():
    """Length-grouped waves looked obviously right until wave_sizes showed 83-91% K=1."""
    h = _hyp(predicted_metric="wave_sizes", predicted_direction="increase")
    only_k1 = _panel(wave_sizes={"1": 300})
    g = validity_gate(h, only_k1, only_k1)
    assert not g.passed
    assert "K=1" in g.reason


def test_validity_rejects_a_cache_change_judged_on_a_hit_heavy_run():
    h = _hyp(gap_id="cache-miss-path", predicted_metric="cache_hit_rate",
             predicted_direction="increase")
    hit = _panel(validity=_v(workload_regime="cache_hit_heavy"))
    g = validity_gate(h, hit, hit)
    assert not g.passed and "cache_hit_heavy" in g.reason


def test_validity_rejects_incomparable_arms():
    a, b = _panel(), _panel(validity=_v(run_group="other-session"))
    g = validity_gate(_hyp(), a, b)
    assert not g.passed and "not comparable" in g.reason


def test_validity_passes_a_well_formed_experiment():
    assert validity_gate(_hyp(), _panel(), _panel()).passed


# ---------------------------------------------------------------- significance

def test_noise_does_not_pass():
    a = _panel(ttft_p95=200.0, validity=_v(stderr=20.0))
    b = _panel(ttft_p95=192.0, validity=_v(stderr=20.0))
    g = significance_gate(_hyp(), a, b)
    assert not g.passed and "noise" in g.reason


def test_a_regression_is_not_reported_as_a_win():
    """Moving significantly the WRONG way must fail, not pass on magnitude alone."""
    a = _panel(ttft_p95=200.0, validity=_v(stderr=1.0))
    b = _panel(ttft_p95=400.0, validity=_v(stderr=1.0))
    g = significance_gate(_hyp(), a, b)          # predicted decrease
    assert not g.passed and "against the prediction" in g.reason.lower()


def test_a_real_improvement_passes():
    a = _panel(ttft_p95=200.0, validity=_v(stderr=1.0))
    b = _panel(ttft_p95=100.0, validity=_v(stderr=1.0))
    assert significance_gate(_hyp(), a, b).passed


# ---------------------------------------------------------------- correctness

def test_correctness_fails_on_new_iteration_errors():
    a, b = _panel(total_iteration_errors=0), _panel(total_iteration_errors=3)
    g = correctness_gate(a, b, run_tests=False)
    assert not g.passed and "iteration errors" in g.reason


def test_correctness_passes_when_errors_flat():
    a, b = _panel(total_iteration_errors=2), _panel(total_iteration_errors=2)
    assert correctness_gate(a, b, run_tests=False).passed


# ---------------------------------------------------------------- cost

def test_cost_flags_a_latency_win_that_costs_startup():
    """Bucketed decode graphs: a real 1.8x win that quietly added ~21 min of startup."""
    a = _panel(graph_capture_s=6.0)
    b = _panel(graph_capture_s=574.0)
    g = cost_gate(a, b)
    assert not g.passed and "startup" in g.reason


def test_cost_flags_memory_regression():
    g = cost_gate(_panel(peak_gpu_mem_gb=40.0), _panel(peak_gpu_mem_gb=52.0))
    assert not g.passed and "memory" in g.reason


def test_cost_passes_when_nothing_regressed():
    assert cost_gate(_panel(graph_capture_s=6.0, peak_gpu_mem_gb=40.0),
                     _panel(graph_capture_s=7.0, peak_gpu_mem_gb=40.5)).passed


# ---------------------------------------------------------------- judge()

def test_judge_verdict_is_invalid_when_validity_fails_even_if_the_effect_is_huge():
    """The ordering that matters: a big number from a harness that never exercised the change
    is a confident wrong answer, not a win."""
    h = _hyp(predicted_metric="wave_sizes", predicted_direction="increase")
    a = _panel(wave_sizes={"1": 300}, validity=_v(stderr=1.0))
    b = _panel(wave_sizes={"1": 300}, validity=_v(stderr=1.0))
    j = judge(h, a, b, run_tests=False)
    assert not j.passed and j.verdict == "invalid"


def test_judge_verdict_noise_when_effect_within_variance():
    a = _panel(ttft_p95=200.0, validity=_v(stderr=25.0))
    b = _panel(ttft_p95=195.0, validity=_v(stderr=25.0))
    j = judge(_hyp(), a, b, run_tests=False)
    assert j.verdict == "noise"


def test_judge_confirms_a_clean_win_and_records_all_four_gates():
    a = _panel(ttft_p95=200.0, validity=_v(stderr=1.0))
    b = _panel(ttft_p95=100.0, validity=_v(stderr=1.0))
    j = judge(_hyp(), a, b, run_tests=False)
    assert j.passed and j.verdict == "confirmed"
    assert set(j.to_dict()) == {"validity", "significance", "correctness", "cost"}


# ---------------------------------------------------------------- pre-merge enforcement

def test_premerge_classifies_engine_vs_exempt_changes():
    """The gate must fire on engine behaviour and stay out of the way otherwise — a gate that
    fires on docs and tests just teaches people to bypass it."""
    import importlib.util
    from pathlib import Path

    from inference_server.research.schemas import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "premerge", Path(REPO_ROOT) / "scripts" / "premerge_check.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)

    engine = ["src/inference_server/scheduler.py",
              "src/inference_server/backends/custom_torch_backend.py"]
    exempt = ["docs/architecture.html", "tests/test_x.py", "scripts/bench_y.py",
              "knowledge/kb-1.json", "src/inference_server/research/gates.py"]

    assert pm.behavioural(engine) == engine
    assert pm.behavioural(exempt) == []
    assert pm.behavioural(engine + exempt) == engine
