"""Append-only run state: one JSON file per record, admitted only in the order graph.py allows."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import graph
from .schemas import git_sha


class LedgerError(ValueError):
    """A record the graph does not allow here. Always fatal."""


@dataclass(frozen=True)
class Record:
    id: str
    run_id: str
    seq: int
    node: str
    type: str | None
    outcome: str
    reads: tuple[str, ...]
    body: dict[str, Any]
    objection: str | None
    graph_version: int
    engine_sha: str
    created_at: str


class Ledger:
    """Records of one run, stored as `<run_dir>/<seq>-<node>.json`."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.records: list[Record] = []
        for path in sorted(self.run_dir.glob("*.json")):
            raw = json.loads(path.read_text())
            self.records.append(Record(**{**raw, "reads": tuple(raw["reads"])}))

    def get(self, record_id: str) -> Record:
        for r in self.records:
            if r.id == record_id:
                return r
        raise LedgerError(f"no record {record_id!r} in run {self.run_id}")

    def latest(self, type_: str) -> Record | None:
        return next((r for r in reversed(self.records) if r.type == type_), None)

    def expected_node(self) -> str | None:
        """The only node the graph admits next; None once the run has stopped."""
        if not self.records:
            return graph.ENTRY
        last = self.records[-1]
        return graph.next_node(last.node, last.outcome)

    def append(self, node: str, outcome: str, body: dict[str, Any], *,
               reads: tuple[str, ...] = (), objection: str | None = None) -> Record:
        expected = self.expected_node()
        if expected is None:
            raise LedgerError(f"run {self.run_id} has stopped; nothing may follow")
        if node != expected:
            raise LedgerError(f"graph expects {expected!r} next, got {node!r}")
        spec = graph.NODES[node]
        if outcome not in spec.outcomes:
            raise LedgerError(f"{node!r} cannot end with {outcome!r}; allowed {spec.outcomes}")
        if (outcome == "objection") != bool(objection):
            raise LedgerError("an objection outcome needs its reason, and only it may carry one")
        read_types = {self.get(i).type for i in reads}
        missing = set(spec.takes) - read_types
        if missing:
            raise LedgerError(f"{node!r} must read a {sorted(missing)} record")

        record = Record(
            id=f"rec-{uuid.uuid4().hex[:8]}", run_id=self.run_id, seq=len(self.records),
            node=node, type=spec.gives, outcome=outcome, reads=tuple(reads), body=body,
            objection=objection, graph_version=graph.GRAPH_VERSION, engine_sha=git_sha(),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / f"{record.seq:03d}-{node}.json"
        with path.open("x") as f:   # exclusive create: an existing record is never overwritten
            json.dump(asdict(record), f, indent=2)
        self.records.append(record)
        return record
