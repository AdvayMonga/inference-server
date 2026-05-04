"""Scheduling policies — decide which pending request to admit next."""

from abc import ABC, abstractmethod
from collections import deque
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
    """First-come-first-served: admit in arrival order."""

    def __init__(self) -> None:
        self._queue: deque["ScheduledRequest"] = deque()

    def pick_next(self) -> "ScheduledRequest | None":
        if not self._queue:
            return None
        return self._queue.popleft()

    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        self._queue.append(request)


class FairPolicy(SchedulingPolicy):
    """Virtual Token Counter fairness over session_id.

    Picks the session with the fewest tokens served so far. New sessions
    inherit the current min counter so they don't get an unfair burst.
    """

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._pending: dict[str, deque["ScheduledRequest"]] = {}

    def pick_next(self) -> "ScheduledRequest | None":
        sessions_with_pending = [s for s, q in self._pending.items() if q]
        if not sessions_with_pending:
            return None
        # Smallest counter wins; tiebreak on the head request's arrival_seq.
        chosen = min(
            sessions_with_pending,
            key=lambda s: (self._counters.get(s, 0.0), self._pending[s][0].arrival_seq),
        )
        return self._pending[chosen].popleft()

    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        sid = request.session_id
        if sid not in self._counters:
            active = [c for s, c in self._counters.items() if self._pending.get(s)]
            self._counters[sid] = min(active) if active else 0.0
        self._pending.setdefault(sid, deque()).append(request)

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
