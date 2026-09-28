"""Walks the graph for one turn: calls each node, saves its record, stops where the graph ends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import graph
from .state import Record, TurnState


@dataclass(frozen=True)
class Result:
    outcome: str
    body: dict[str, Any]
    objection: str | None = None


# A node sees only the latest record of each type it `takes`; the runner picks them, not the node.
NodeFn = Callable[[dict[str, Record]], Result]


def run_turn(state: TurnState, impls: dict[str, NodeFn], *,
             pause: Callable[[Record], bool] | None = None) -> Record:
    """Run until the turn ends, or until `pause` (supervised mode) returns False; resumable."""
    missing = set(graph.NODES) - set(impls)
    if missing:
        raise ValueError(f"no implementation for nodes {sorted(missing)}")
    while (node := state.expected_node()) is not None:
        inputs = {t: state.latest(t) for t in graph.NODES[node].takes}
        absent = [t for t, r in inputs.items() if r is None]
        if absent:
            raise ValueError(f"{node!r} needs {absent} but the turn has none")
        marker = state.turn_dir / f"{len(state.records):03d}-{node}.started"
        if marker.exists():   # a node may have side effects (GPU spend, a PR); never re-run blind
            raise RuntimeError(f"{node!r} was interrupted mid-run; check its effects, then delete {marker}")
        state.turn_dir.mkdir(parents=True, exist_ok=True)
        marker.touch()
        res = impls[node](inputs)
        rec = state.append(node, res.outcome, res.body,
                           reads=tuple(r.id for r in inputs.values()), objection=res.objection)
        marker.unlink()
        if pause is not None and not pause(rec):
            return rec
    return state.records[-1]
