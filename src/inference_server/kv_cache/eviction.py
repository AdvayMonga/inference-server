"""Eviction policies — decide which KV cache block to evict when memory is full."""

import time
from abc import ABC, abstractmethod
from collections import defaultdict

from inference_server.kv_cache.block import Block


class EvictionPolicy(ABC):
    """Interface for eviction policies. Swap implementations via config."""

    @abstractmethod
    def select_victim(self, candidates: list[Block]) -> Block | None:
        """Pick which block to evict from eviction-eligible candidates (ref_count == 0)."""
        ...

    @abstractmethod
    def on_access(self, block: Block) -> None:
        """Called when a block is accessed — update tracking state."""
        ...

    def record_attention(self, block: Block, score: float) -> None:
        """Record attention score for a block. Only used by score-based policies."""
        pass


class LRUPolicy(EvictionPolicy):
    """Evict the least recently used block."""

    def select_victim(self, candidates: list[Block]) -> Block | None:
        if not candidates:
            return None
        return min(candidates, key=lambda b: b.last_accessed)

    def on_access(self, block: Block) -> None:
        block.last_accessed = time.time()


class AttentionSinkLRUPolicy(EvictionPolicy):
    """LRU but first block of every sequence is permanently protected."""

    def select_victim(self, candidates: list[Block]) -> Block | None:
        evictable = [b for b in candidates if not b.is_first_block]
        if not evictable:
            return None
        return min(evictable, key=lambda b: b.last_accessed)

    def on_access(self, block: Block) -> None:
        block.last_accessed = time.time()


class H2OPolicy(EvictionPolicy):
    """Heavy Hitter Oracle — evict by lowest cumulative attention score."""

    def __init__(self):
        self._scores: dict[int, float] = defaultdict(float)

    def select_victim(self, candidates: list[Block]) -> Block | None:
        evictable = [b for b in candidates if not b.is_first_block]
        if not evictable:
            return None
        return min(evictable, key=lambda b: self._scores.get(b.block_id, 0.0))

    def on_access(self, block: Block) -> None:
        block.last_accessed = time.time()

    def record_attention(self, block: Block, score: float) -> None:
        """Accumulate attention score for this block."""
        self._scores[block.block_id] += score

    def get_score(self, block: Block) -> float:
        """Get cumulative attention score for a block."""
        return self._scores.get(block.block_id, 0.0)

    def clear_scores(self, block: Block) -> None:
        """Remove scores for a freed block."""
        self._scores.pop(block.block_id, None)


def create_eviction_policy(policy_name: str) -> EvictionPolicy:
    """Factory — create eviction policy by config name."""
    if policy_name == "lru":
        return LRUPolicy()
    elif policy_name == "attention_sink_lru":
        return AttentionSinkLRUPolicy()
    elif policy_name == "h2o":
        return H2OPolicy()
    else:
        raise ValueError(
            f"Unknown eviction policy: {policy_name}. Available: lru, attention_sink_lru, h2o"
        )
