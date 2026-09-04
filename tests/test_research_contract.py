"""The result contract: a panel must carry its provenance, and unsafe comparisons must fail.

Every case here is a real failure this project hit. They are tests, not documentation, because
the loop will make these comparisons unattended.
"""

from __future__ import annotations

import json

import pytest

from inference_server.research.compare import comparable, significance
from inference_server.research.schemas import (
    PANEL_VERSION,
    Experiment,
    Hypothesis,
    KnowledgeEntry,
    SchemaError,
    Validity,
    Vitals,
)


def _validity(**over):
    base = dict(
        engine_sha="abc1234", dirty=False, harness="bench_serving",
        harness_config={"model": "E4B", "gpu": "A100", "max_batch_size": 256, "pool_size": 64,
                        "rates": "1,2", "compile": 0},
        workload_regime="cache_hit_heavy", n_samples=5, run_group="grp-1", stderr=1.0,
    )
    base.update(over)
    return Validity(**base)


def _panel(tok_s=100.0, **over):
    v = over.pop("validity", None) or _validity(**over.pop("validity_over", {}))
    return Vitals(validity=v, tok_s_within_slo=tok_s, **over)


# ---------------------------------------------------------------- provenance is mandatory

def test_panel_without_harness_config_is_rejected():
    """The POOL_SIZE=64 / MAX_BATCH_SIZE class of error: a number with no knobs attached."""
    p = _panel(validity=_validity(harness_config={}))
    with pytest.raises(SchemaError, match="harness_config"):
        p.validate()


def test_panel_without_run_group_is_rejected():
    p = _panel(validity=_validity(run_group=""))
    with pytest.raises(SchemaError, match="run_group"):
        p.validate()


def test_unknown_workload_regime_rejected():
    p = _panel(validity=_validity(workload_regime="whatever"))
    with pytest.raises(SchemaError, match="workload_regime"):
        p.validate()


def test_roundtrip_preserves_everything(tmp_path):
    p = _panel(ttft_p95=204.0, wave_sizes={"1": 174, "2": 16})
    path = tmp_path / "run.json"
    p.to_json(path)
    back = Vitals.load(path)
    assert back.ttft_p95 == 204.0
    assert back.wave_sizes == {"1": 174, "2": 16}
    assert back.validity.run_group == "grp-1"


def test_future_panel_version_is_refused(tmp_path):
    """A record written by a different definition of the panel must not be silently read."""
    p = _panel()
    d = json.loads(p.to_json())
    d["panel_version"] = PANEL_VERSION + 1
    with pytest.raises(SchemaError, match="panel_version"):
        Vitals.from_dict(d).validate()


# ---------------------------------------------------------------- refusing bad comparisons

def test_cross_session_comparison_refused():
    """957 vs 1151 tok/s for identical config, different sessions. Never again."""
    a = _panel(validity=_validity(run_group="session-A"))
    b = _panel(validity=_validity(run_group="session-B"))
    c = comparable(a, b)
    assert not c
    assert any("run_group" in r for r in c.reasons)


def test_differing_harness_config_refused():
    a = _panel()
    b = _panel(validity=_validity(harness_config={**_validity().harness_config,
                                                  "max_batch_size": 32}))
    c = comparable(a, b)
    assert not c
    assert any("max_batch_size" in r for r in c.reasons)


def test_differing_workload_regime_refused():
    """A cache-hit run flatters prefill work a cache-miss run does not."""
    a = _panel()
    b = _panel(validity=_validity(workload_regime="cache_miss_heavy"))
    assert not comparable(a, b)


def test_identical_setup_is_comparable():
    assert comparable(_panel(), _panel(tok_s=110.0))


# ---------------------------------------------------------------- significance

def test_single_sample_is_never_significant():
    a = _panel(100.0, validity=_validity(n_samples=1, stderr=None))
    b = _panel(140.0, validity=_validity(n_samples=1, stderr=None))
    s = significance(a, b, "tok_s_within_slo", direction="increase")
    assert s.verdict == "insufficient_samples"


def test_effect_inside_variance_is_noise():
    a = _panel(100.0, validity=_validity(stderr=12.0))
    b = _panel(104.0, validity=_validity(stderr=12.0))
    s = significance(a, b, "tok_s_within_slo", direction="increase")
    assert s.verdict == "noise", s.detail


def test_clear_effect_is_significant():
    a = _panel(100.0, validity=_validity(stderr=1.0))
    b = _panel(140.0, validity=_validity(stderr=1.0))
    s = significance(a, b, "tok_s_within_slo", direction="increase")
    assert s.verdict == "significant" and s.improved
    assert s.pct == pytest.approx(40.0)


def test_regression_against_prediction_is_flagged_not_celebrated():
    a = _panel(140.0, validity=_validity(stderr=1.0))
    b = _panel(100.0, validity=_validity(stderr=1.0))
    s = significance(a, b, "tok_s_within_slo", direction="increase")
    assert s.verdict == "significant"
    assert "AGAINST the prediction" in s.detail


def test_significance_refuses_incomparable_arms():
    a = _panel(100.0, validity=_validity(run_group="A", stderr=1.0))
    b = _panel(140.0, validity=_validity(run_group="B", stderr=1.0))
    assert significance(a, b, "tok_s_within_slo", direction="increase").verdict == "not_comparable"


# ---------------------------------------------------------------- hypotheses

def _hyp(**over):
    base = dict(statement="s", gap_id="g1", predicted_metric="ttft_p95",
                predicted_direction="decrease", predicted_magnitude=">20%",
                falsification_tier=2, falsification_test="cpu repro")
    base.update(over)
    return Hypothesis(**base)


def test_hypothesis_requires_a_magnitude_before_measuring():
    with pytest.raises(SchemaError, match="predicted_magnitude"):
        _hyp(predicted_magnitude="").validate()


def test_hypothesis_metric_must_be_a_real_panel_field():
    with pytest.raises(SchemaError, match="not a panel field"):
        _hyp(predicted_metric="vibes").validate()


def test_hypothesis_can_target_the_loop_itself():
    """'This harness cannot test what I am about to change' is a first-class hypothesis —
    it is how the K=1 wave discovery would be produced deliberately rather than by luck."""
    h = _hyp(kind="harness_change", predicted_metric="wave_sizes",
             predicted_direction="increase", falsification_tier=1)
    h.validate()
    assert h.kind == "harness_change"


def test_experiment_requires_all_four_gates_green():
    e = Experiment(hypothesis_id="h", engine_sha_base="abc")
    assert not e.all_gates_green()
    e.gates = {g: {"passed": True} for g in ("validity", "significance", "correctness")}
    assert not e.all_gates_green(), "three of four is not enough"
    e.gates["cost"] = {"passed": True}
    assert e.all_gates_green()
    e.gates["correctness"] = {"passed": False}
    assert not e.all_gates_green()


def test_knowledge_entry_validates():
    k = KnowledgeEntry(title="t", summary="s", status="rejected")
    k.validate()
    with pytest.raises(SchemaError):
        KnowledgeEntry(title="t", summary="s", status="maybe").validate()
