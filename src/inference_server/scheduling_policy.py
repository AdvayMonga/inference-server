"""Scheduling policies — pure ordering functions, wrapped in thin stateful shells.

The decision lives in the module-level functions so the scheduler and a simulator share it.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from inference_server.scheduler import ScheduledRequest


class Candidate(Protocol):
    """What ordering reads off a waiting request — a ScheduledRequest or a simulator's record."""
    session_id: str
    priority: int
    arrival_seq: int


C = TypeVar("C", bound=Candidate)


def fcfs_order(candidates: Iterable[C]) -> list[C]:
    """Admission order under FCFS: (-priority, arrival_seq). Pure."""
    return sorted(candidates, key=lambda r: (-r.priority, r.arrival_seq))


def fair_order(candidates: Iterable[C], counters: Mapping[str, float]) -> list[C]:
    """Admission order under VTC: (-priority, counter[session_id], arrival_seq). Pure."""
    return sorted(
        candidates,
        key=lambda r: (-r.priority, counters.get(r.session_id, 0.0), r.arrival_seq),
    )


def fair_initial_counter(
    session_id: str, counters: Mapping[str, float], pending_sessions: Iterable[str]
) -> float:
    """Counter on arrival: own if known, else min over sessions with pending work, else 0. Pure."""
    if session_id in counters:
        return counters[session_id]
    active = [counters[s] for s in set(pending_sessions) if s in counters]
    return min(active) if active else 0.0


def fair_charge(counters: dict[str, float], session_id: str, n_tokens: int) -> None:
    """Charge n_tokens to a session, in place (runs per row per decode step; copying is waste)."""
    counters[session_id] = counters.get(session_id, 0.0) + n_tokens


class SchedulingPolicy(ABC):
    """Interface for admission ordering. Swap implementations via config."""

    @abstractmethod
    def pick_next(self) -> "ScheduledRequest | None":
        """Return and remove the next request to admit, or None if nothing is ready."""
        ...

    @abstractmethod
    def peek_next(self) -> "ScheduledRequest | None":
        """Return the next request to admit without removing it, or None."""
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

    def peek_window(self, n: int) -> list["ScheduledRequest"]:
        """Up to `n` next requests in policy order, without removing them.

        Lets the scheduler form a prefill wave out of similarly-sized prompts (a ragged wave is
        right-padded to Smax, so mixed lengths waste most of the prefill). Default returns just
        the head, i.e. no reordering — a custom policy keeps strict ordering unless it opts in.
        """
        head = self.peek_next()
        return [head] if head is not None else []

    def pick(self, request: "ScheduledRequest") -> None:
        """Remove a SPECIFIC request that peek_window returned. Default: unsupported."""
        raise NotImplementedError(f"{type(self).__name__} does not support pick()")

    def pending(self) -> list["ScheduledRequest"]:
        """Every waiting request, in no particular order.

        The admission deadline needs this because no policy here orders by age — FCFS keys on
        (-priority, arrival_seq) and VTC on (-priority, counter, arrival_seq) — so checking only
        peek_next() leaves an overtaken request ageing forever without ever being examined.
        Expiry is a property of a request, not of its position in the queue.
        """
        return []


class _ListPolicy(SchedulingPolicy):
    """Shell shared by the built-in policies: a pending list, ordered by a pure function."""

    def __init__(self) -> None:
        self._pending: list["ScheduledRequest"] = []

    @abstractmethod
    def _order(self) -> list["ScheduledRequest"]:
        """Every pending request in admission order."""
        ...

    def peek_next(self) -> "ScheduledRequest | None":
        ordered = self._order()
        return ordered[0] if ordered else None

    def pick_next(self) -> "ScheduledRequest | None":
        head = self.peek_next()
        if head is not None:
            self._pending.remove(head)
        return head

    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        self._pending.append(request)

    def peek_window(self, n: int) -> list["ScheduledRequest"]:
        return self._order()[:n]

    def pick(self, request: "ScheduledRequest") -> None:
        self._pending.remove(request)

    def pending(self) -> list["ScheduledRequest"]:
        return list(self._pending)


class FCFSPolicy(_ListPolicy):
    """First-come-first-served, with priority as the dominant key (see fcfs_order)."""

    def _order(self) -> list["ScheduledRequest"]:
        return fcfs_order(self._pending)


class FairPolicy(_ListPolicy):
    """Virtual Token Counter fairness over session_id, gated by priority (see fair_order).

    Priority dominates fairness; within a tier the least-served session wins; arrival_seq
    breaks final ties. New sessions inherit the min counter across sessions with pending work.
    """

    def __init__(self) -> None:
        super().__init__()
        self._counters: dict[str, float] = {}

    def _order(self) -> list["ScheduledRequest"]:
        return fair_order(self._pending, self._counters)

    def on_request_arrived(self, request: "ScheduledRequest") -> None:
        sid = request.session_id
        self._counters[sid] = fair_initial_counter(
            sid, self._counters, (r.session_id for r in self._pending)
        )
        super().on_request_arrived(request)

    def on_tokens_processed(self, request: "ScheduledRequest", n_tokens: int) -> None:
        fair_charge(self._counters, request.session_id, n_tokens)


def create_scheduling_policy(policy_name: str) -> SchedulingPolicy:
    """Factory — create scheduling policy by config name."""
    if policy_name == "fcfs":
        return FCFSPolicy()
    if policy_name == "fair":
        return FairPolicy()
    raise ValueError(
        f"Unknown scheduling policy: {policy_name}. Available: fcfs, fair"
    )
