"""The loop graph is well-formed, and a turn's saved state admits records only in the order it allows."""

from __future__ import annotations

import json

import pytest

from inference_server.research.method import graph
from inference_server.research.method.state import StateError, TurnState


def test_every_edge_joins_declared_nodes_and_outcomes():
    for (src, outcome), dst in graph.EDGES.items():
        assert src in graph.NODES and dst in graph.NODES
        assert outcome in graph.NODES[src].outcomes


def test_every_outcome_of_a_live_node_has_somewhere_to_go():
    for name, node in graph.NODES.items():
        if name in graph.TERMINAL:
            continue
        for outcome in node.outcomes:
            assert (name, outcome) in graph.EDGES, (name, outcome)


def test_only_a_violation_or_an_anomaly_stops_for_a_human():
    to_human = {src_outcome for src_outcome, dst in graph.EDGES.items() if dst == "ask_human"}
    assert to_human == {("check", "violation"), ("experiment", "anomaly")}


def test_every_node_is_reachable_from_the_entry():
    seen, frontier = set(), [graph.ENTRY]
    while frontier:
        n = frontier.pop()
        if n not in seen:
            seen.add(n)
            frontier += [dst for (src, _), dst in graph.EDGES.items() if src == n]
    assert seen == set(graph.NODES)


def _walk_to_hypothesis(led: TurnState):
    q = led.append("question", "ok", {"class": "cold_start"})
    p = led.append("measure", "ok", {"ttft_p95": 1.0}, reads=(q.id,))
    return q, p


def _walk_to_change(led: TurnState):
    q, p = _walk_to_hypothesis(led)
    h = led.append("hypothesize", "ok", {"claim": "x"}, reads=(q.id, p.id))
    c = led.append("build", "ok", {}, reads=(h.id, p.id))
    return h, c


def _walk_to_results(led: TurnState, h, c):
    k = led.append("check", "pass", {}, reads=(c.id,))
    led.append("review_1", "ok", {}, reads=(h.id, c.id, k.id))
    res = led.append("experiment", "done", {}, reads=(h.id, c.id))
    prof = led.append("profile", "ok", {}, reads=(led.latest("Panel").id, res.id))
    return res, prof


def test_an_accepted_change_is_published_then_recorded(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    h, c = _walk_to_change(led)
    res, prof = _walk_to_results(led, h, c)
    j = led.append("review_2", "accept", {}, reads=(h.id, c.id, res.id, prof.id))
    led.append("publish", "ok", {}, reads=(c.id, j.id))
    led.append("record", "ok", {}, reads=(h.id,))
    assert led.expected_node() is None
    with pytest.raises(StateError, match="ended"):
        led.append("measure", "ok", {})


def test_a_rejected_change_skips_publish(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    h, c = _walk_to_change(led)
    res, prof = _walk_to_results(led, h, c)
    led.append("review_2", "reject", {}, reads=(h.id, c.id, res.id, prof.id))
    assert led.expected_node() == "record"


def _send_back(led, h, c, outcome):
    k = led.append("check", "fail" if outcome == "fix" else "pass", {}, reads=(c.id,))
    led.append("review_1", outcome, {}, reads=(h.id, c.id, k.id))


def test_rounds_are_capped_but_check_failures_are_free(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    h, c = _walk_to_change(led)
    for _ in range(3):   # failed checks spend no round
        _send_back(led, h, c, "fix")
        c = led.append("build", "ok", {}, reads=(h.id, led.latest("Panel").id))
    for _ in range(graph.MAX_ROUNDS):
        _send_back(led, h, c, "changes")
        c = led.append("build", "ok", {}, reads=(h.id, led.latest("Panel").id))
    _send_back(led, h, c, "changes")   # one round past the cap
    assert led.expected_node() == graph.GIVE_UP


def test_a_revision_is_refused_once_the_experiments_are_spent(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    h, c = _walk_to_change(led)
    for _ in range(graph.MAX_EXPERIMENTS):
        res, prof = _walk_to_results(led, h, c)
        led.append("review_2", "revise", {}, reads=(h.id, c.id, res.id, prof.id))
        if led.expected_node() == "build":
            c = led.append("build", "ok", {}, reads=(h.id, led.latest("Panel").id))
    assert led.expected_node() == graph.GIVE_UP   # a revision it could never measure


def test_a_crashed_agent_runs_again(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    h = led.append("hypothesize", "ok", {}, reads=(q.id, p.id))
    led.append("build", "crashed", {}, reads=(h.id, p.id))
    assert led.expected_node() == "build"


def test_a_violation_stops_the_turn_for_a_human(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    h, c = _walk_to_change(led)
    led.append("check", "violation", {}, reads=(c.id,))
    assert led.expected_node() == "ask_human"


def test_the_state_refuses_a_skipped_step(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    with pytest.raises(StateError, match="expects 'hypothesize'"):
        led.append("build", "ok", {}, reads=(p.id,))


def test_the_state_refuses_an_undeclared_outcome(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    with pytest.raises(StateError, match="cannot end with"):
        led.append("question", "confirmed", {})


def test_a_node_must_read_the_records_it_takes(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    with pytest.raises(StateError, match="Question"):
        led.append("hypothesize", "ok", {}, reads=(p.id,))


def test_a_note_is_recorded_and_never_changes_the_path(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    h = led.append("hypothesize", "ok", {}, reads=(q.id, p.id), note="I needed the KV profile")
    assert h.note == "I needed the KV profile" and led.expected_node() == "build"
    assert TurnState(tmp_path / "run1" / "turn-001").records[-1].note == h.note


def test_records_survive_a_reload_and_are_never_overwritten(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, _ = _walk_to_hypothesis(led)
    again = TurnState(tmp_path / "run1" / "turn-001")
    assert [r.id for r in again.records] == [r.id for r in led.records]
    assert again.expected_node() == "hypothesize"
    stale = TurnState(tmp_path / "run1" / "turn-001")
    again.append("hypothesize", "ok", {}, reads=(q.id, again.records[1].id))
    with pytest.raises(FileExistsError):   # a second writer cannot clobber a record
        stale.append("hypothesize", "ok", {}, reads=(q.id, stale.records[1].id))
    assert json.loads((tmp_path / "run1" / "turn-001" / "002-hypothesize.json").read_text())["node"] == "hypothesize"
