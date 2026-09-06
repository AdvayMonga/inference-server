"""The procedure guards. Each one is a mistake this project actually made and paid for."""

from __future__ import annotations

import json

import pytest

from inference_server.research.schemas import Validity, Vitals
from inference_server.research.session import (
    NotMeasurable,
    _alternating,
    arm_label,
    arms_for,
)


def _panel(tmp_path, arm, *, started_at, run_group="g1", sha="abc", dirty=False, rid=None):
    v = Validity(engine_sha=sha, dirty=dirty, harness="bench_serving",
                 harness_config={"pool_size": 4000}, workload_regime="cache_miss_heavy",
                 n_samples=5, run_group=run_group, started_at=started_at,
                 notes=f"arm={arm}" if arm else "", stderr=1.0)
    if rid:
        v.run_id = rid
    p = Vitals(validity=v, ttft_p95=100.0)
    (tmp_path / f"run-{v.run_id}.json").write_text(json.dumps(p.to_dict()))
    return p


def _group(tmp_path, order, **kw):
    for i, arm in enumerate(order):
        _panel(tmp_path, arm, started_at=1000.0 + i, **kw)
    return tmp_path


def test_arm_label_reads_the_notes_token():
    v = Validity(engine_sha="a", dirty=False, harness="h", harness_config={},
                 workload_regime="mixed", n_samples=1, run_group="g",
                 notes="cold cache, arm=treatment")
    assert arm_label(Vitals(validity=v)) == "treatment"


def test_untagged_panel_is_refused_rather_than_guessed():
    """An untagged panel silently becoming its own arm is worse than a hard stop."""
    v = Validity(engine_sha="a", dirty=False, harness="h", harness_config={},
                 workload_regime="mixed", n_samples=1, run_group="g", notes="")
    assert arm_label(Vitals(validity=v)) == ""


def test_arms_split_and_order_is_accepted_when_alternating(tmp_path):
    _group(tmp_path, ["baseline", "treatment", "treatment", "baseline",
                      "baseline", "treatment", "treatment", "baseline"])
    arms = arms_for("g1", runs_dir=tmp_path)
    assert {k: len(v) for k, v in arms.items()} == {"baseline": 4, "treatment": 4}


def test_blocked_arm_order_is_refused(tmp_path):
    """Three identical back-to-back runs measured 1494 / 401 / 229ms. Run every baseline first
    and the treatment wins on warmup drift alone."""
    _group(tmp_path, ["baseline"] * 3 + ["treatment"] * 3)
    with pytest.raises(NotMeasurable, match="blocks, not alternating"):
        arms_for("g1", runs_dir=tmp_path)


def test_thin_arms_are_refused(tmp_path):
    """<3 runs leaves <2 after discarding warmup, which cannot estimate a spread."""
    _group(tmp_path, ["baseline", "treatment", "baseline", "treatment"])
    with pytest.raises(NotMeasurable, match="fewer than 3 runs"):
        arms_for("g1", runs_dir=tmp_path)


def test_three_arms_are_refused(tmp_path):
    """One variable per experiment, or the result attributes to nothing in particular."""
    _group(tmp_path, ["baseline", "treatment", "other",
                      "baseline", "treatment", "other",
                      "baseline", "treatment", "other"])
    with pytest.raises(NotMeasurable, match="expected exactly 2 arms"):
        arms_for("g1", runs_dir=tmp_path)


def test_untagged_panel_in_a_group_is_refused(tmp_path):
    _group(tmp_path, ["baseline", "treatment", "baseline", ""])
    with pytest.raises(NotMeasurable, match="no arm="):
        arms_for("g1", runs_dir=tmp_path)


def test_empty_group_is_refused(tmp_path):
    with pytest.raises(NotMeasurable, match="no panels"):
        arms_for("nope", runs_dir=tmp_path)


def test_panels_from_other_run_groups_are_ignored(tmp_path):
    """The rule that makes cross-session comparison impossible rather than discouraged."""
    _group(tmp_path, ["baseline", "treatment", "baseline", "treatment", "baseline", "treatment"])
    for i, arm in enumerate(["baseline"] * 3):
        _panel(tmp_path, arm, started_at=2000.0 + i, run_group="other-session")
    arms = arms_for("g1", runs_dir=tmp_path)
    assert {k: len(v) for k, v in arms.items()} == {"baseline": 3, "treatment": 3}


@pytest.mark.parametrize("order,ok", [
    (["a", "b", "b", "a"], True),
    (["a", "b", "a", "b"], True),
    (["a", "a", "b", "b"], False),
    (["a", "a", "a", "b", "b", "b"], False),
])
def test_alternating_accepts_abba_and_abab_but_not_blocks(order, ok):
    assert _alternating(order) is ok


def test_dirty_tree_panels_cannot_be_judged(tmp_path, monkeypatch):
    """A sha names code that was not what ran."""
    from inference_server.research import session

    _group(tmp_path, ["baseline", "treatment", "baseline",
                      "treatment", "baseline", "treatment"], dirty=True)
    monkeypatch.setattr(session, "_changed_files", lambda a, b: [])
    with pytest.raises(NotMeasurable, match="dirty tree"):
        session.check_still_measurable(arms_for("g1", runs_dir=tmp_path))


def test_engine_drift_since_measurement_is_refused(tmp_path, monkeypatch):
    """The rebase-after-measuring rule, enforced rather than remembered. Cost one A100 run."""
    from inference_server.research import session

    _group(tmp_path, ["baseline", "treatment", "baseline",
                      "treatment", "baseline", "treatment"])
    monkeypatch.setattr(session, "_changed_files",
                        lambda a, b: ["src/inference_server/scheduler.py"])
    with pytest.raises(NotMeasurable, match="do not rebase after measuring"):
        session.check_still_measurable(arms_for("g1", runs_dir=tmp_path))


def test_doc_and_test_churn_since_measurement_is_fine(tmp_path, monkeypatch):
    """A gate that fires on documentation only teaches people to bypass it."""
    from inference_server.research import session

    _group(tmp_path, ["baseline", "treatment", "baseline",
                      "treatment", "baseline", "treatment"])
    monkeypatch.setattr(session, "_changed_files",
                        lambda a, b: ["docs/architecture.html", "tests/test_x.py",
                                      "src/inference_server/research/gates.py"])
    session.check_still_measurable(arms_for("g1", runs_dir=tmp_path))


def test_drift_prefixes_match_the_merge_gate():
    """Two definitions of 'engine behaviour' that disagree would let a change pass one and fail
    the other, which is worse than having only one."""
    import importlib.util
    from pathlib import Path

    from inference_server.research import session
    from inference_server.research.schemas import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "premerge", Path(REPO_ROOT) / "scripts" / "premerge_check.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)

    assert session.BEHAVIOURAL_PREFIXES == pm.BEHAVIOURAL_PREFIXES
    assert session.EXEMPT_PREFIXES == pm.EXEMPT_PREFIXES
