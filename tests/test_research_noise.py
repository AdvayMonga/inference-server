"""The noise band: computing it from a null experiment, and the behaviour it changes.

Synthetic panels throughout — the real runs are the instrument's job. What is tested here is the
arithmetic and, more importantly, the wiring: a delta the t-test calls significant must come back
`inconclusive` once a band says the harness moves that far on its own.
"""

from __future__ import annotations

import pytest

from inference_server.research import harness as H
from inference_server.research.compare import significance_replicated
from inference_server.research.gates import judge, significance_gate
from inference_server.research.noise import (
    NoiseBand,
    band_for,
    band_from_arms,
    find_band,
    format_band,
    metric_band,
)
from inference_server.research.schemas import Hypothesis, Vitals

MODEL = "google/gemma-4-E2B-it"
HARDWARE = "Apple M4 Pro (MPS)"
CORPUS = "c0ffee" * 10


def _panel(ttft_p95: float, *, tpot_p50: float = 100.0, arm: str = "baseline",
           workload_class: str = "cold_start", model: str = MODEL,
           hardware: str = HARDWARE, group: str = "grp-null") -> Vitals:
    v = H.build_validity(
        "replay_trace",
        {"workload_class": workload_class, "split": "seen", "rate_scale": 1.0, "model": model,
         "max_batch_size": "8"},
        n_samples=6, workload_regime="cache_miss_heavy", notes=f"arm={arm}")
    v.run_group = group
    v.corpus_version = CORPUS
    v.workload_class = workload_class
    v.device_state = {"gpu_name": hardware, "clocks_locked": False}
    v.clocks_locked = False
    return H.panel_from_stats(v, ttft_p95=ttft_p95, ttft_p50=ttft_p95 / 2, tpot_p50=tpot_p50,
                              wall_s=30.0)


def _arms(baseline: list[float], treatment: list[float], **kw) -> dict[str, list[Vitals]]:
    return {"baseline": [_panel(x, arm="baseline", **kw) for x in baseline],
            "treatment": [_panel(x, arm="treatment", **kw) for x in treatment]}


# --------------------------------------------------------------------------- the arithmetic

def test_metric_band_reports_spread_both_ways():
    b = metric_band([100.0, 120.0, 140.0])
    assert b.n == 3 and b.mean == pytest.approx(120.0)
    assert b.sd == pytest.approx(20.0)
    assert b.cv == pytest.approx(20.0 / 120.0, abs=1e-4)
    assert (b.min, b.max) == (100.0, 140.0)
    assert b.min_max_ratio == pytest.approx(1.4)


def test_one_run_is_not_a_band():
    assert metric_band([42.0]) is None


def test_band_drops_each_arms_warmup_run_like_the_significance_test_does():
    """The band has to describe the spread the gate faces, and the gate never sees run 1.
    Three identical back-to-back runs once measured 1494/401/229 ms — including the first would
    report a band nearly twice as wide as anything an A/B can be judged against."""
    arms = _arms([1000.0, 100.0, 110.0, 120.0], [1000.0, 105.0, 115.0, 125.0])
    band = band_from_arms(arms)
    assert band.n_runs == 6
    assert band.metrics["ttft_p95"].max == 125.0            # the two 1000s are gone
    raw = band_from_arms(arms, drop_first_per_arm=False)
    assert raw.metrics["ttft_p95"].max == 1000.0 and raw.n_runs == 8


def test_band_skips_a_metric_that_is_missing_from_any_run():
    arms = _arms([100.0, 110.0, 120.0], [105.0, 115.0, 125.0])
    arms["treatment"][-1].tpot_p50 = None
    band = band_from_arms(arms)
    assert "ttft_p95" in band.metrics
    assert "tpot_p50" not in band.metrics, "a band from a partial series is worse than none"


def test_band_records_where_it_applies():
    band = band_from_arms(_arms([100.0, 110.0, 120.0], [105.0, 115.0, 125.0]))
    assert (band.harness, band.workload_class, band.model, band.hardware) == \
        ("replay_trace", "cold_start", MODEL, HARDWARE)
    assert band.corpus_version == CORPUS and band.arms == ["baseline", "treatment"]
    assert len(band.run_ids) == band.n_runs


def test_band_needs_two_runs_after_warmup():
    with pytest.raises(ValueError, match=">= 2 runs"):
        band_from_arms({"baseline": [_panel(100.0)], "treatment": [_panel(101.0)]})


def test_band_round_trips_through_json(tmp_path):
    band = band_from_arms(_arms([100.0, 110.0, 120.0], [105.0, 115.0, 125.0]))
    p = tmp_path / "b.json"
    band.to_json(p)
    back = NoiseBand.load(p)
    assert back.to_dict() == band.to_dict()
    assert "ttft_p95" in format_band(back)


# --------------------------------------------------------------------------- lookup

def _write(tmp_path, band: NoiseBand) -> None:
    name = "".join(c if c.isalnum() else "-" for c in band.key())
    band.to_json(tmp_path / f"{name}.json")


def test_find_band_matches_the_whole_situation_and_never_a_near_miss(tmp_path):
    _write(tmp_path, band_from_arms(_arms([100.0, 110.0, 120.0], [105.0, 115.0, 125.0])))
    kw = dict(harness="replay_trace", workload_class="cold_start", model=MODEL,
              hardware=HARDWARE, directory=tmp_path)
    assert find_band(**kw) is not None
    # A band for another class / box / model is not an approximation of this one.
    assert find_band(**{**kw, "workload_class": "steady_interactive"}) is None
    assert find_band(**{**kw, "hardware": "NVIDIA A100 80GB PCIe"}) is None
    assert find_band(**{**kw, "model": "google/gemma-4-E4B-it"}) is None
    assert find_band(**{**kw, "corpus_version": "different"}) is None


def test_band_for_reads_the_situation_off_the_panel(tmp_path):
    _write(tmp_path, band_from_arms(_arms([100.0, 110.0, 120.0], [105.0, 115.0, 125.0])))
    assert band_for(_panel(111.0), directory=tmp_path) is not None
    assert band_for(_panel(111.0, workload_class="long_context"), directory=tmp_path) is None


def test_inside_says_nothing_about_a_metric_it_never_measured():
    band = band_from_arms(_arms([100.0, 110.0, 120.0], [105.0, 115.0, 125.0]))
    assert band.inside("pct_of_memory_roof", 99.0) is None, \
        "an unmeasured metric must leave the caller behaving as if there were no band"


# --------------------------------------------------------------------------- the wiring

HYP = Hypothesis(statement="x", gap_id="g", predicted_metric="ttft_p95",
                 predicted_direction="decrease", predicted_magnitude="-20%",
                 falsification_tier=3, falsification_test="replay")


def _tight_arms():
    """Arms that separate cleanly on a t-test: 100±2 vs 85±2, a -15% effect."""
    return ([_panel(x, arm="baseline") for x in (999.0, 100.0, 102.0, 98.0)],
            [_panel(x, arm="treatment") for x in (999.0, 85.0, 87.0, 83.0)])


def test_without_a_band_a_clean_separation_is_significant():
    base, treat = _tight_arms()
    s = significance_replicated(base, treat, "ttft_p95", direction="decrease")
    assert s.verdict == "significant" and s.pct == pytest.approx(-15.0, abs=0.5)


def test_a_delta_inside_the_recorded_band_is_inconclusive_not_a_win():
    """The whole point. Three runs an arm can separate 15% cleanly; if the same harness moves
    20% with nothing changed, that separation is not evidence."""
    band = band_from_arms(_arms([100.0, 80.0, 120.0, 90.0], [100.0, 125.0, 85.0, 115.0]))
    assert band.width_pct("ttft_p95") > 15.0
    base, treat = _tight_arms()
    s = significance_replicated(base, treat, "ttft_p95", direction="decrease", band=band)
    assert s.verdict == "inconclusive"
    assert "inside the measured noise band" in s.detail
    assert not s.improved


def test_an_effect_larger_than_the_band_still_passes():
    band = band_from_arms(_arms([100.0, 98.0, 102.0, 99.0], [100.0, 101.0, 97.0, 103.0]))
    assert band.width_pct("ttft_p95") < 15.0
    base, treat = _tight_arms()
    s = significance_replicated(base, treat, "ttft_p95", direction="decrease", band=band)
    assert s.verdict == "significant"


def test_the_significance_gate_fails_on_an_inconclusive_delta():
    band = band_from_arms(_arms([100.0, 80.0, 120.0, 90.0], [100.0, 125.0, 85.0, 115.0]))
    base, treat = _tight_arms()
    g = significance_gate(HYP, base, treat, band=band)
    assert not g.passed and g.evidence["verdict"] == "inconclusive"
    assert significance_gate(HYP, base, treat).passed, "no band: the gate is what it always was"


def test_judgement_names_inconclusive_separately_from_noise():
    """`noise` means the arms did not separate; `inconclusive` means they did but by less than
    the harness's own spread. More replicates fix the first and cannot fix the second."""
    band = band_from_arms(_arms([100.0, 80.0, 120.0, 90.0], [100.0, 125.0, 85.0, 115.0]))
    base, treat = _tight_arms()
    j = judge(HYP, base, treat, run_tests=False, band=band)
    assert not j.passed and j.verdict == "inconclusive"
    assert judge(HYP, base, treat, run_tests=False).verdict == "confirmed"
