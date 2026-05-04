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


def create_scheduling_policy(policy_name: str) -> SchedulingPolicy:
    """Factory — create scheduling policy by config name."""
    if policy_name == "fcfs":
        return FCFSPolicy()
    raise ValueError(
        f"Unknown scheduling policy: {policy_name}. Available: fcfs"
    )
