"""Prefix-aware session-affine router: locality (session affinity, then prefix) versus load.

    score = w*(aw*holds_session + matched_blocks/prompt_blocks)/(1+aw) + (1-w)*(1 - load/max_load)

`w=0` is exactly shortest-queue, `w=1` exactly best-locality, `aw=0` ablates affinity. `route()`
is pure, the `scheduling_policy.py` pattern. Why this shape: `kb-20260917-0c9ba6de`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from control_plane.prefix_index import PrefixIndex, Prompt


class ReplicaView(NamedTuple):
    """Immutable snapshot of one replica, as the router last heard it."""

    replica_id: str
    queue_depth: int = 0
    active: int = 0
    # Only ever tested > 0: capacity is a gate, not a score term, so the magnitude is not
    # consulted. A replica reporting 0 is skipped while any other has room.
    free_blocks: int = 1
    sessions: frozenset[str] = frozenset()     # sessions whose KV this replica holds

    @property
    def load(self) -> int:
        return self.queue_depth + self.active


class RequestView(NamedTuple):
    """What routing reads off a request. `prompt` is whatever the index was built to accept."""

    session_id: str
    prompt: Prompt


def replica_score(replica: ReplicaView, session_id: str, matched_blocks: int, prompt_blocks: int,
                  max_load: int, weight: float, affinity_weight: float = 1.0) -> float:
    """One replica's score under the module formula. Pure; higher wins."""
    affinity = 1.0 if session_id in replica.sessions else 0.0
    prefix = matched_blocks / prompt_blocks if prompt_blocks else 0.0
    locality = (affinity_weight * affinity + prefix) / (1.0 + affinity_weight)
    load = 1.0 - (replica.load / max_load if max_load else 0.0)
    return weight * locality + (1.0 - weight) * load


def route(request: RequestView, replicas: Sequence[ReplicaView], index: PrefixIndex,
          weight: float, affinity_weight: float = 1.0) -> str | None:
    """Pick a replica for one request. Pure, deterministic; None only if there are none."""
    # Full replicas are skipped, but with nowhere else to go a full one is still a decision.
    candidates = [r for r in replicas if r.free_blocks > 0] or list(replicas)
    if not candidates:
        return None
    matched = {m.replica_id: m.matched_blocks for m in index.lookup(request.prompt)}
    prompt_blocks = index.n_blocks(request.prompt)
    max_load = max(r.load for r in candidates)

    def key(r: ReplicaView) -> tuple[float, str]:
        score = replica_score(r, request.session_id, matched.get(r.replica_id, 0), prompt_blocks,
                              max_load, weight, affinity_weight)
        # Rounded: a tie must break on replica_id, never on float noise, or every sweep that
        # replays this decision gets a different answer.
        return (-round(score, 9), r.replica_id)

    return min(candidates, key=key).replica_id


class Router:
    """Stateful shell: the replica table and the index. Every decision goes through `route`."""

    def __init__(self, block_size: int, weight: float = 0.5, affinity_weight: float = 1.0,
                 tokenize: Callable[[str], Sequence[int]] | None = None):
        self.index = PrefixIndex(block_size, tokenize)
        self.replicas: dict[str, ReplicaView] = {}
        self.weight = weight
        self.affinity_weight = affinity_weight

    def on_request(self, request: RequestView) -> str | None:
        """Route one request. Reads the index; never writes it — only replicas confirm state."""
        return route(request, list(self.replicas.values()), self.index, self.weight,
                     self.affinity_weight)

    def on_replica_state(self, view: ReplicaView) -> None:
        """Replace a replica's snapshot. Feeding these late is how staleness is simulated."""
        self.replicas[view.replica_id] = view

    def on_replica_gone(self, replica_id: str) -> None:
        """Drop a replica from the table and from the index."""
        self.replicas.pop(replica_id, None)
        self.index.drop_replica(replica_id)
