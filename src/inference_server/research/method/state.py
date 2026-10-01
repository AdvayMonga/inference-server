"""Saved state of one loop turn: append-only JSON records, admitted only in the order graph.py allows."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import graph
from ..schemas import git_sha


class StateError(ValueError):
    """A record the graph does not allow here. Always fatal."""


@dataclass(frozen=True)
class Record:
    id: str
    run_id: str
    turn: str
    seq: int
    node: str
    type: str | None
    outcome: str
    reads: tuple[str, ...]
    body: dict[str, Any]
    note: str | None             # an agent's suggestion to the human; never changes the path
    graph_version: int
    engine_sha: str
    created_at: str


class TurnState:
    """Records of one turn, stored as `runs/<run_id>/<turn>/<seq>-<node>.json`."""

    def __init__(self, turn_dir: Path):
        self.turn_dir = Path(turn_dir)
        self.run_id, self.turn = self.turn_dir.parent.name, self.turn_dir.name
        self.records: list[Record] = []
        for path in sorted(self.turn_dir.glob("*.json")):
            raw = json.loads(path.read_text())
            self.records.append(Record(**{**raw, "reads": tuple(raw["reads"])}))

    def get(self, record_id: str) -> Record:
        for r in self.records:
            if r.id == record_id:
                return r
        raise StateError(f"no record {record_id!r} in {self.run_id}/{self.turn}")

    def latest(self, type_: str) -> Record | None:
        return next((r for r in reversed(self.records) if r.type == type_), None)

    def expected_node(self) -> str | None:
        """The only node the graph admits next; None once the turn has ended."""
        if not self.records:
            return graph.ENTRY
        last = self.records[-1]
        nxt = graph.next_node(last.node, last.outcome)
        if nxt is None:
            return None
        rounds = sum((r.node, r.outcome) in graph.ROUNDS for r in self.records)
        experiments = sum(r.node == "experiment" for r in self.records)
        if nxt == "build" and rounds > graph.MAX_ROUNDS:
            return graph.GIVE_UP
        if experiments >= graph.MAX_EXPERIMENTS and (
                nxt == "experiment" or (nxt == "build" and last.node == "review_2")):
            return graph.GIVE_UP   # a revision it could never measure
        if sum(r.node == nxt for r in self.records) >= graph.MAX_NODE_VISITS:
            return graph.GIVE_UP
        return nxt

    def append(self, node: str, outcome: str, body: dict[str, Any], *,
               reads: tuple[str, ...] = (), note: str | None = None) -> Record:
        expected = self.expected_node()
        if expected is None:
            raise StateError(f"turn {self.run_id}/{self.turn} has ended; nothing may follow")
        if node != expected:
            raise StateError(f"graph expects {expected!r} next, got {node!r}")
        spec = graph.NODES[node]
        if outcome not in spec.outcomes:
            raise StateError(f"{node!r} cannot end with {outcome!r}; allowed {spec.outcomes}")
        read_types = {self.get(i).type for i in reads}
        missing = set(spec.takes) - read_types
        if missing:
            raise StateError(f"{node!r} must read a {sorted(missing)} record")

        record = Record(
            id=f"rec-{uuid.uuid4().hex[:8]}", run_id=self.run_id, turn=self.turn,
            seq=len(self.records),
            node=node, type=spec.gives, outcome=outcome, reads=tuple(reads), body=body,
            note=note, graph_version=graph.GRAPH_VERSION, engine_sha=git_sha(),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self.turn_dir.mkdir(parents=True, exist_ok=True)
        path = self.turn_dir / f"{record.seq:03d}-{node}.json"
        with path.open("x") as f:   # exclusive create: an existing record is never overwritten
            json.dump(asdict(record), f, indent=2)
        self.records.append(record)
        return record
