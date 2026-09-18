"""Global prefix index: which replicas are believed to hold which block-aligned prefixes.

Centralised on purpose (`notes/infserv/10-control-plane.md`): one process, easy to reason
about, and the interesting result is the staleness-tolerance curve rather than the topology.

It mirrors the engine's `PrefixCache.lookup` notion of a match — the longest prefix that is a
whole number of blocks, because a trailing partial block is not safe to share — but imports
nothing from the engine, which needs torch.

Tokenisation is the caller's job, since the engine's tokenizer is not importable here. Supply
`block_size` and then either pass pre-tokenised ids to every call, or construct with
`tokenize` (a `str -> Sequence[int]`) and pass prompts; a text-only simulator can pass a
stand-in that chunks characters, as long as the same one is used for insert and lookup.

Staleness is first class: each entry records the tick it was last confirmed at, and the index
never updates itself. Updates arrive only when a replica reports, so a simulation can feed them
late (or not at all) and measure how wrong the routing decision was.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

Prompt = Sequence[int] | str


class Match(NamedTuple):
    """One replica's longest block-aligned hit, and the tick that hit was last confirmed at."""

    replica_id: str
    matched_blocks: int
    confirmed_at: int


class PrefixIndex:
    """Aligned prompt prefix -> replicas believed to hold it, and how many blocks each holds."""

    def __init__(self, block_size: int, tokenize: Callable[[str], Sequence[int]] | None = None):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size
        self.tokenize = tokenize
        self.tick = 0
        # aligned prefix (tuple of token ids) -> replica_id -> tick last confirmed
        self._holders: dict[tuple[int, ...], dict[str, int]] = {}
        self._by_replica: dict[str, set[tuple[int, ...]]] = {}

    def advance(self, n: int = 1) -> int:
        """Move the index's clock on. Entry age is `tick - confirmed_at`."""
        self.tick += n
        return self.tick

    def n_blocks(self, prompt: Prompt) -> int:
        """Full blocks in a prompt. The trailing partial block is never counted."""
        return len(self._ids(prompt)) // self.block_size

    def insert(self, replica_id: str, prompt: Prompt, n_blocks: int | None = None,
               as_of: int | None = None) -> None:
        """Record that `replica_id` holds this prompt's first `n_blocks` blocks (default: all).

        `as_of` is the tick the report describes; it defaults to now, and a late report should
        pass the tick it was actually true at.
        """
        key = self._key(prompt, n_blocks)
        if key is None:
            return
        self._holders.setdefault(key, {})[replica_id] = self.tick if as_of is None else as_of
        self._by_replica.setdefault(replica_id, set()).add(key)

    def lookup(self, prompt: Prompt) -> list[Match]:
        """Each replica's longest block-aligned match, longest first, ties by replica_id.

        Like `PrefixCache.lookup`, a stored entry is only hit by a prompt that extends it: an
        entry is keyed by the whole aligned prefix it covers, so a shorter prompt misses.
        """
        ids = self._ids(prompt)
        best: dict[str, Match] = {}
        for n in range(len(ids) // self.block_size, 0, -1):
            holders = self._holders.get(tuple(ids[: n * self.block_size]), {})
            for replica_id, tick in holders.items():
                if replica_id not in best:
                    best[replica_id] = Match(replica_id, n, tick)
        return sorted(best.values(), key=lambda m: (-m.matched_blocks, m.replica_id))

    def evict(self, replica_id: str, prompt: Prompt, n_blocks: int | None = None) -> None:
        """Forget one entry — the same prefix granularity `insert` stored it under."""
        key = self._key(prompt, n_blocks)
        holders = self._holders.get(key) if key is not None else None
        if holders is None or replica_id not in holders:
            return
        del holders[replica_id]
        if not holders:
            del self._holders[key]
        self._by_replica.get(replica_id, set()).discard(key)

    def drop_replica(self, replica_id: str) -> None:
        """Forget everything a replica held. It can no longer win any routing decision."""
        for key in self._by_replica.pop(replica_id, set()):
            holders = self._holders.get(key, {})
            holders.pop(replica_id, None)
            if not holders:
                self._holders.pop(key, None)

    def _ids(self, prompt: Prompt) -> list[int]:
        if isinstance(prompt, str):
            if self.tokenize is None:
                raise TypeError("PrefixIndex was given a prompt string but built without a "
                                "tokenize function; pass token ids or supply tokenize=")
            return list(self.tokenize(prompt))
        return list(prompt)

    def _key(self, prompt: Prompt, n_blocks: int | None) -> tuple[int, ...] | None:
        ids = self._ids(prompt)
        full = len(ids) // self.block_size
        n = full if n_blocks is None else min(n_blocks, full)
        return tuple(ids[: n * self.block_size]) if n > 0 else None
