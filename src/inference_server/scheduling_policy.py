"""Scheduling policies — decide which pending request to admit next."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference_server.scheduler import ScheduledRequest


class SchedulingPolicy(ABC):
    """Interface for admission ordering. Swap implementations via config."""

    @abstractmethod
    def pick_next(self) -> "ScheduledRequest | None":
        """Return the next request to admit, or None if nothing is ready."""
        ...

    @abstractmethod
    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        """Index a newly enqueued request into the policy's internal store."""
        ...

    def on_tokens_processed(self, request: "ScheduledRequest", n_tokens: int) -> None:
        """Called per active row per decode step. No-op for arrival-order policies."""
        pass

    def on_request_finished(self, request: "ScheduledRequest") -> None:
        """Called when a request leaves the running batch."""
        pass


class FCFSPolicy(SchedulingPolicy):
    """First-come-first-served, with priority as the dominant key.

    Sort key: (-priority, arrival_seq). Higher priority drains first;
    within a priority tier, oldest arrival wins.
    """

    def __init__(self) -> None:
        self._pending: list["ScheduledRequest"] = []

    def pick_next(self) -> "ScheduledRequest | None":
        if not self._pending:
            return None
        chosen = min(self._pending, key=lambda r: (-r.priority, r.arrival_seq))
        self._pending.remove(chosen)
        return chosen

    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        self._pending.append(request)


class FairPolicy(SchedulingPolicy):
    """Virtual Token Counter fairness over session_id, gated by priority.

    Sort key: (-priority, virtual_counter[session_id], arrival_seq).
    Priority dominates fairness; within a priority tier, the session that
    has been served least wins; arrival_seq breaks final ties.

    New sessions inherit the current min counter across active sessions
    so they don't starve behind old high-spenders, and can't catch-up-burst.
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._pending: list["ScheduledRequest"] = []

    def pick_next(self) -> "ScheduledRequest | None":
        if not self._pending:
            return None
        chosen = min(
            self._pending,
            key=lambda r: (-r.priority, self._counters.get(r.session_id, 0.0), r.arrival_seq),
        )
        self._pending.remove(chosen)
        return chosen

    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        sid = request.session_id
        if sid not in self._counters:
            pending_sids = {r.session_id for r in self._pending}
            active = [c for s, c in self._counters.items() if s in pending_sids]
            self._counters[sid] = min(active) if active else 0.0
        self._pending.append(request)

    def on_tokens_processed(self, request: "ScheduledRequest", n_tokens: int) -> None:
        self._counters[request.session_id] = (
            self._counters.get(request.session_id, 0.0) + n_tokens
        )


def create_scheduling_policy(policy_name: str) -> SchedulingPolicy:
    """Factory — create scheduling policy by config name."""
    if policy_name == "fcfs":
        return FCFSPolicy()
    if policy_name == "fair":
        return FairPolicy()
    raise ValueError(
        f"Unknown scheduling policy: {policy_name}. Available: fcfs, fair"
    )
