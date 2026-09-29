"""The runner walks a whole turn on stub nodes; real nodes replace the stubs one at a time."""

from __future__ import annotations

import pytest

from inference_server.research.method import graph
from inference_server.research.method.runner import Result, run_turn
from inference_server.research.method.state import TurnState


def follows_check(inputs):
    """A review_1 stand-in that never approves a failing check."""
    return Result("changes" if inputs["CheckReport"].outcome == "fail" else "ok", {})


def stubs(**overrides):
    """Every node returns its first declared outcome; `overrides` replace chosen nodes."""
    impls = {name: (lambda _inputs, o=node.outcomes[0]: Result(o, {}))
             for name, node in graph.NODES.items()}
    return {**impls, **overrides}


@pytest.fixture
def state(tmp_path):
    return TurnState(tmp_path / "run1" / "turn-001")


def test_a_turn_runs_end_to_end_on_stubs(state):
    last = run_turn(state, stubs())
    assert [r.node for r in state.records] == [
        "question", "measure", "hypothesize", "build", "check", "review_1",
        "experiment", "review_2", "publish", "record"]
    assert last.node == "record" and state.expected_node() is None


def test_a_node_sees_only_the_latest_records_it_takes(state):
    seen = {}

    def review_2(inputs):
        seen.update({t: r.node for t, r in inputs.items()})
        return Result("reject", {})

    run_turn(state, stubs(review_2=review_2))
    assert seen == {"Hypothesis": "hypothesize", "Change": "build", "Results": "experiment"}
    judgement = state.latest("Judgement")
    assert judgement.reads == tuple(state.latest(t).id for t in ("Hypothesis", "Change", "Results"))


def test_an_objection_stops_the_turn_at_ask_human(state):
    run_turn(state, stubs(build=lambda _i: Result("objection", {}, objection="wrong target")))
    assert [r.node for r in state.records][-2:] == ["build", "ask_human"]
    assert state.expected_node() is None


def test_supervised_mode_pauses_and_resumes_from_disk(state, tmp_path):
    run_turn(state, stubs(), pause=lambda rec: rec.node != "hypothesize")
    assert state.records[-1].node == "hypothesize"
    again = TurnState(tmp_path / "run1" / "turn-001")
    run_turn(again, stubs())
    assert again.records[-1].node == "record"
    assert [r.node for r in again.records].count("hypothesize") == 1


def test_endless_check_failures_give_up_at_the_build_cap(state):
    run_turn(state, stubs(check=lambda _i: Result("fail", {}), review_1=follows_check))
    nodes = [r.node for r in state.records]
    assert nodes.count("build") == graph.NODES["build"].max_visits
    assert nodes[-1] == graph.GIVE_UP


def test_every_node_needs_an_implementation(state):
    impls = stubs()
    del impls["review_1"]
    with pytest.raises(ValueError, match="review_1"):
        run_turn(state, impls)


def test_review_1_sees_the_passing_change_after_a_failed_build(state):
    outcomes = iter(["fail", "pass"])
    seen = {}

    def review_1(inputs):
        seen.update({t: r.seq for t, r in inputs.items()})
        return follows_check(inputs)

    run_turn(state, stubs(check=lambda _i: Result(next(outcomes), {}), review_1=review_1))
    builds = [r.seq for r in state.records if r.node == "build"]
    checks = [r for r in state.records if r.node == "check"]
    assert seen["Change"] == builds[-1]
    assert seen["CheckReport"] == checks[-1].seq and checks[-1].outcome == "pass"


def test_an_interrupted_node_is_never_re_run_blind(state, tmp_path):
    def crash(_inputs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_turn(state, stubs(experiment=crash))
    again = TurnState(tmp_path / "run1" / "turn-001")
    with pytest.raises(RuntimeError, match="interrupted"):
        run_turn(again, stubs())
