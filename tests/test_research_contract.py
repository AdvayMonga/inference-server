"""The result contract: a panel must carry its provenance, and unsafe comparisons must fail.

Every case here is a real failure this project hit. They are tests, not documentation, because
the loop will make these comparisons unattended.
"""

from __future__ import annotations

import json

import pytest

from inference_server.research.compare import (
    comparable,
    significance,
    significance_replicated,
)
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


def _runs(values):
    return [_panel(v) for v in values]


def test_effect_inside_variance_is_noise():
    s = significance_replicated(_runs([1, 100, 104, 98]), _runs([1, 104, 99, 103]),
                                "tok_s_within_slo", direction="increase")
    assert s.verdict == "noise", s.detail


def test_clear_effect_is_significant():
    s = significance_replicated(_runs([1, 100, 101, 99]), _runs([1, 140, 141, 139]),
                                "tok_s_within_slo", direction="increase")
    assert s.verdict == "significant" and s.improved
    assert s.pct == pytest.approx(40.0, abs=1.0)


def test_regression_against_prediction_is_flagged_not_celebrated():
    s = significance_replicated(_runs([1, 140, 141, 139]), _runs([1, 100, 101, 99]),
                                "tok_s_within_slo", direction="increase")
    assert s.verdict == "significant"
    assert "AGAINST the prediction" in s.detail


def test_too_few_replicates_is_refused():
    s = significance_replicated(_runs([1, 100]), _runs([1, 140]),
                                "tok_s_within_slo", direction="increase")
    assert s.verdict == "insufficient_samples"


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
    e.gates = {g: {"passed": True}
               for g in ("validity", "sanity", "significance", "correctness")}
    assert not e.all_gates_green(), "four of five is not enough"
    e.gates["cost"] = {"passed": True}
    assert e.all_gates_green()
    e.gates["correctness"] = {"passed": False}
    assert not e.all_gates_green()


def test_knowledge_entry_validates():
    k = KnowledgeEntry(title="t", summary="s", status="rejected")
    k.validate()
    with pytest.raises(SchemaError):
        KnowledgeEntry(title="t", summary="s", status="maybe").validate()


# ---------------------------------------------------------------- knowledge base

def test_kb_roundtrip_and_index(tmp_path):
    from inference_server.research.kb import (
        generate_index,
        load_entries,
        query,
        save_entry,
    )
    d = tmp_path / "kb"
    save_entry(KnowledgeEntry(title="Mixed-batch prefill", status="rejected",
                              tags=["prefill", "scheduler"],
                              summary="Eager forward is dispatch-bound; 3-5x tax.",
                              triggers=["a cheap variable-shape forward exists"]), d)
    save_entry(KnowledgeEntry(title="Radix prefix cache", status="resolved", tags=["cache"],
                              summary="Shares at every block boundary."), d)

    entries = load_entries(d)
    assert len(entries) == 2
    assert len(query(entries, tags=["cache"])) == 1
    assert len(query(entries, status="rejected")) == 1
    assert len(query(entries, text="dispatch")) == 1

    idx = generate_index(entries)
    assert "Do not retry blind" in idx          # the rejected section carries its warning
    assert "Mixed-batch prefill" in idx
    assert "**Revisit when:**" in idx


def test_kb_flags_a_rehash_of_a_rejected_idea(tmp_path):
    """The guard that stops the loop re-proposing a measured dead end."""
    from inference_server.research.kb import already_rejected, save_entry
    d = tmp_path / "kb"
    save_entry(KnowledgeEntry(
        title="Mixed-batch chunked prefill rejected",
        summary="A mixed step cannot replay the decode graph; eager forward is dispatch bound.",
        status="rejected", tags=["prefill"]), d)
    from inference_server.research.kb import load_entries
    entries = load_entries(d)
    hits = already_rejected("try mixed-batch chunked prefill to overlap decode", entries)
    assert hits, "should flag a restatement of a rejected idea"
    assert not already_rejected("add int8 weight quantisation to the decode path", entries)


def test_experiment_ledger_roundtrip(tmp_path):
    from inference_server.research.kb import (
        experiment_for_sha,
        load_experiments,
        save_experiment,
    )
    from inference_server.research.schemas import Arm
    d = tmp_path / "exp"
    e = Experiment(hypothesis_id="hyp-1", engine_sha_base="aaa1111", branch="perf/x",
                   arms=[Arm(name="baseline", sha="aaa1111"),
                         Arm(name="treatment", sha="bbb2222")], verdict="confirmed")
    save_experiment(e, d)
    back = load_experiments(d)
    assert len(back) == 1 and back[0].arms[1].sha == "bbb2222"
    assert experiment_for_sha("bbb2222", d) is not None
    assert experiment_for_sha("ccc3333", d) is None


def test_related_finds_settled_entries_by_tag_not_wording(tmp_path):
    """The first real iteration's false negative: 'chunk long prefills across ticks' shares
    almost no words with the entry that rules it out, so free-text matching missed it. Tags are
    declared rather than inferred, so they do not miss."""
    from inference_server.research.kb import related, save_entry
    d = tmp_path / "kb"
    save_entry(KnowledgeEntry(
        "TTFT p95 is prefill compute",
        summary="Queue p95 21ms vs prefill p95 203ms; scheduling levers are ruled out.",
        status="resolved", tags=["prefill", "scheduler"]), d)
    save_entry(KnowledgeEntry(title="Unrelated quantisation note", summary="int8 weights",
                              status="open", tags=["quantization"]), d)
    from inference_server.research.kb import load_entries
    entries = load_entries(d)

    hits = related("chunk long prefills across scheduler ticks", entries,
                   tags=["prefill", "scheduler"])
    assert hits, "tagged lookup must surface the entry that rules this out"
    assert hits[0][1].title.startswith("TTFT p95")

    # and it must not drag in everything
    assert all(e.title != "Unrelated quantisation note" for _, e in hits)
