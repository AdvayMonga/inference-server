"""The procedure guards. Each one is a mistake this project actually made and paid for."""

from __future__ import annotations

import json
import subprocess

import pytest

from inference_server.research.schemas import Hypothesis, Validity, Vitals
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


# ------------------------------------------- the staleness rule, against a real git history
#
# These build an actual repo because the rule is now about ancestry, not just a file list: the
# baseline arm of a two-commit A/B sits one commit behind the treatment arm on purpose.


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _commit(repo, path, text):
    f = repo / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"touch {path}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


def _history(tmp_path):
    """A repo whose commits are, in order: an engine baseline, the change under test, and two
    follow-ons (one engine, one doc) the tests check out as HEAD."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    base = _commit(repo, "src/inference_server/backend.py", "v1\n")
    treat = _commit(repo, "src/inference_server/backend.py", "v2\n")
    later = _commit(repo, "src/inference_server/scheduler.py", "v1\n")
    _git(repo, "checkout", "-q", treat)
    docs = _commit(repo, "docs/architecture.html", "<p>x</p>\n")
    _git(repo, "checkout", "-q", treat)
    return repo, base, treat, later, docs


def _sibling(repo, of, path, text):
    """A commit on a branch that forks from `of`, so it and `of`'s other children have no tip."""
    _git(repo, "checkout", "-q", of)
    sha = _commit(repo, path, text)
    _git(repo, "checkout", "-q", of)
    return sha


def _ab(runs_dir, base_sha, treat_sha):
    """Six alternating panels, baseline measured at one sha and treatment at another."""
    runs_dir.mkdir(exist_ok=True)
    for i, arm in enumerate(["baseline", "treatment"] * 3):
        _panel(runs_dir, arm, started_at=1000.0 + i,
               sha=base_sha if arm == "baseline" else treat_sha)
    return arms_for("g1", runs_dir=runs_dir)


def test_same_sha_arms_still_fail_when_an_engine_file_moved(tmp_path, monkeypatch):
    """The case the rule was written for, unchanged: one sha, one engine commit after it."""
    from inference_server.research import session

    repo, base, treat, later, _ = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    arms = _ab(tmp_path / "runs", base, base)
    with pytest.raises(NotMeasurable, match="do not rebase after measuring"):
        session.check_still_measurable(arms, ref=treat)


def test_same_sha_arms_pass_when_nothing_engine_moved(tmp_path, monkeypatch):
    from inference_server.research import session

    repo, base, treat, _, docs = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    session.check_still_measurable(_ab(tmp_path / "runs", treat, treat), ref=docs)


def test_two_commit_ab_needs_no_escape_hatch(tmp_path, monkeypatch):
    """baseline@parent, treatment@child, HEAD==child. The drift between the baseline arm's sha
    and HEAD is the change under test; only a blanket --no-drift-check used to get through."""
    from inference_server.research import session

    repo, base, treat, _, _ = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    session.check_still_measurable(_ab(tmp_path / "runs", base, treat), ref=treat)


def test_a_commit_after_a_two_commit_ab_fails_both_arms(tmp_path, monkeypatch):
    """Drift is what landed after EVERY arm, so it invalidates every arm — including the one
    measured at the tip."""
    from inference_server.research import session

    repo, base, treat, later, _ = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    arms = _ab(tmp_path / "runs", base, treat)
    for first in ("baseline", "treatment"):
        ordered = {first: arms[first], **arms}
        with pytest.raises(NotMeasurable, match=f"arm '{first}' measured"):
            session.check_still_measurable(ordered, ref=later)


def test_non_engine_commit_after_a_two_commit_ab_fails_neither_arm(tmp_path, monkeypatch):
    from inference_server.research import session

    repo, base, treat, _, docs = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    session.check_still_measurable(_ab(tmp_path / "runs", base, treat), ref=docs)


def test_escape_hatch_still_bypasses_the_check(tmp_path, monkeypatch):
    """check_drift=False remains, for the case the rule cannot know about."""
    from inference_server.research import session

    called = []
    monkeypatch.setattr(session, "check_still_measurable",
                        lambda *a, **k: called.append(a))
    monkeypatch.setattr(session, "judge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    _group(tmp_path, ["baseline", "treatment", "baseline",
                      "treatment", "baseline", "treatment"])
    hyp = Hypothesis(statement="s", gap_id="g1", predicted_metric="ttft_p95",
                     predicted_direction="decrease", predicted_magnitude=">10%",
                     falsification_tier=2, falsification_test="cpu repro")
    for check_drift, expected in ((True, 1), (False, 0)):
        called.clear()
        with pytest.raises(RuntimeError):
            session.judge_group(hyp, "g1", check_drift=check_drift, record=False,
                                runs_dir=tmp_path)
        assert len(called) == expected


def test_arms_on_divergent_branches_fall_back_to_their_own_shas(tmp_path, monkeypatch):
    """No arm sha is an ancestor of the other, so there is no tip and no relief: each arm is
    checked against its own sha, which is the stricter rule."""
    from inference_server.research import session

    repo, base, treat, _, _ = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    fork = _sibling(repo, base, "src/inference_server/backend.py", "v3\n")
    assert session._experiment_tip([treat, fork]) == ""
    # `fork` is not in treat's history, so treat's own panels are stale against it and vice versa.
    with pytest.raises(NotMeasurable, match="do not rebase after measuring"):
        session.check_still_measurable(_ab(tmp_path / "runs", treat, fork), ref=fork)


def test_the_tip_of_a_three_commit_chain_is_the_last_one_whatever_the_order(tmp_path, monkeypatch):
    """The tip is a property of the history, not of the order the arms happen to be listed in."""
    import itertools

    from inference_server.research import session

    repo, base, treat, later, _ = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    for order in itertools.permutations([base, treat, later]):
        assert session._experiment_tip(list(order)) == later


def test_a_sha_git_cannot_resolve_yields_no_tip(tmp_path, monkeypatch):
    """An unanswerable ancestry question must not read as 'not an ancestor': it drops the arms
    back to the per-arm check rather than silently picking a tip from the shas that did resolve."""
    from inference_server.research import session

    repo, base, treat, _, _ = _history(tmp_path)
    monkeypatch.setattr(session, "REPO_ROOT", repo)
    assert session._is_ancestor(base, treat) is True
    assert session._is_ancestor(treat, base) is False
    assert session._is_ancestor("deadbee", treat) is None
    assert session._experiment_tip([base, treat, "deadbee"]) == ""


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


# ---------------------------------------------------------------- derived headline metrics

def _point(arm, rate, *, ttft_p95, tpot_p95, tok_s, trial="t1", started_at=1000.0,
           slo=(200.0, 50.0)):
    v = Validity(engine_sha="abc", dirty=False, harness="bench_serving",
                 harness_config={"rates": str(rate)}, workload_regime="cache_miss_heavy",
                 n_samples=5, run_group="g1", started_at=started_at,
                 notes=f"open-loop Poisson, rate={rate}, arm={arm}, trial={trial}")
    return Vitals(validity=v, ttft_p95=ttft_p95, tpot_p95=tpot_p95,
                  tok_s_within_slo=tok_s, slo_ttft_ms=slo[0], slo_tpot_ms=slo[1])


def test_headline_finds_the_knee_and_best_throughput():
    """The project's primary metric — throughput at a fixed tail-latency budget — existed only in
    the instrument's printed table. A sweep of rate-points is enough to derive it."""
    from inference_server.research.session import sweep_headline

    sweep = [_point("baseline", 1, ttft_p95=90, tpot_p95=30, tok_s=50),
             _point("baseline", 2, ttft_p95=120, tpot_p95=40, tok_s=95),
             _point("baseline", 4, ttft_p95=180, tpot_p95=48, tok_s=140),
             _point("baseline", 6, ttft_p95=400, tpot_p95=80, tok_s=150),
             _point("baseline", 8, ttft_p95=900, tpot_p95=120, tok_s=155)]
    h = sweep_headline(sweep)
    assert h["max_tok_s_within_slo"] == 140      # NOT 155: 6 and 8 req/s broke the budget
    assert h["best_rate_within_slo"] == 4
    assert h["saturation_knee_rate"] == 6
    assert (h["rates_measured"], h["rates_within_slo"]) == (5, 3)


def test_knee_is_the_lowest_breaking_rate_not_the_first_seen():
    """Rate-points can arrive in any order; the knee is where capacity ends."""
    from inference_server.research.session import sweep_headline

    h = sweep_headline([_point("b", 8, ttft_p95=900, tpot_p95=120, tok_s=1),
                        _point("b", 4, ttft_p95=400, tpot_p95=80, tok_s=1),
                        _point("b", 1, ttft_p95=90, tpot_p95=30, tok_s=50)])
    assert h["saturation_knee_rate"] == 4


def test_a_sweep_that_never_breaks_has_no_knee():
    from inference_server.research.session import sweep_headline

    h = sweep_headline([_point("b", 1, ttft_p95=90, tpot_p95=30, tok_s=50),
                        _point("b", 2, ttft_p95=95, tpot_p95=31, tok_s=99)])
    assert h["saturation_knee_rate"] is None and h["max_tok_s_within_slo"] == 99


def test_a_sweep_that_breaks_everywhere_has_no_throughput():
    """iter10 measured exactly this: every run failed the budget at rate 2."""
    from inference_server.research.session import sweep_headline

    h = sweep_headline([_point("b", 2, ttft_p95=400, tpot_p95=120, tok_s=None)])
    assert h["max_tok_s_within_slo"] is None and h["saturation_knee_rate"] == 2


def test_replicate_sweeps_stay_separate():
    """Averaging replicates here would hide the run-to-run spread that decides significance."""
    from inference_server.research.session import headline

    ps = [_point("baseline", 1, ttft_p95=90, tpot_p95=30, tok_s=50, trial="t1"),
          _point("baseline", 2, ttft_p95=400, tpot_p95=90, tok_s=None, trial="t1"),
          _point("baseline", 1, ttft_p95=95, tpot_p95=31, tok_s=60, trial="t2"),
          _point("baseline", 2, ttft_p95=410, tpot_p95=95, tok_s=None, trial="t2")]
    hs = headline(ps)
    assert len(hs) == 2
    assert sorted(h["max_tok_s_within_slo"] for h in hs) == [50, 60]


def test_untagged_panels_become_sweeps_of_one():
    """An untagged panel yields no knee rather than a wrong one shared with a stranger."""
    from inference_server.research.session import sweeps

    a = _point("baseline", 1, ttft_p95=90, tpot_p95=30, tok_s=50)
    a.validity.notes = "no trial tag here"
    b = _point("baseline", 2, ttft_p95=95, tpot_p95=31, tok_s=60)
    b.validity.notes = "nor here"
    assert len(sweeps([a, b])) == 2


def test_within_slo_is_none_when_the_panel_carries_no_budget():
    """Absent budgets must not read as 'passed'."""
    from inference_server.research.session import within_slo

    p = _point("b", 1, ttft_p95=90, tpot_p95=30, tok_s=50)
    p.slo_ttft_ms = None
    assert within_slo(p) is None
