"""The loop graph is well-formed, and a turn's saved state admits records only in the order it allows."""

from __future__ import annotations

import json

import pytest

from inference_server.research import graph
from inference_server.research.state import StateError, TurnState


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


def test_every_agent_node_can_object_to_a_human():
    for name, node in graph.NODES.items():
        if node.kind == "agent":
            assert graph.EDGES.get((name, "objection")) == "ask_human", name


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


def test_a_turn_ends_at_record(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    h = led.append("hypothesize", "ok", {"claim": "x"}, reads=(q.id, p.id))
    v = led.append("experiment", "confirmed", {}, reads=(h.id, p.id))
    led.append("record", "confirmed", {}, reads=(h.id, v.id))
    assert led.expected_node() is None
    with pytest.raises(StateError, match="ended"):
        led.append("measure", "ok", {})


def test_the_state_refuses_a_skipped_step(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    with pytest.raises(StateError, match="expects 'hypothesize'"):
        led.append("experiment", "confirmed", {}, reads=(p.id,))


def test_the_state_refuses_an_undeclared_outcome(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    with pytest.raises(StateError, match="cannot end with"):
        led.append("question", "confirmed", {})


def test_a_node_must_read_the_records_it_takes(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    with pytest.raises(StateError, match="Question"):
        led.append("hypothesize", "ok", {}, reads=(p.id,))


def test_an_objection_stops_the_run_and_needs_a_reason(tmp_path):
    led = TurnState(tmp_path / "run1" / "turn-001")
    q, p = _walk_to_hypothesis(led)
    with pytest.raises(StateError, match="reason"):
        led.append("hypothesize", "objection", {}, reads=(q.id, p.id))
    led.append("hypothesize", "objection", {}, reads=(q.id, p.id), objection="measure first")
    led.append("ask_human", "stopped", {})
    assert led.expected_node() is None
    with pytest.raises(StateError, match="ended"):
        led.append("measure", "ok", {})


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
