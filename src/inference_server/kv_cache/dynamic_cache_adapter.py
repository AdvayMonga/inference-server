"""Simple prompt-level KV cache — stores full DynamicCache per unique prompt."""

import logging
from collections import OrderedDict

from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)


class DynamicCacheAdapter:
    """Stores and restores full DynamicCache objects keyed by prompt token tuples."""

    def __init__(self, max_entries: int = 64):
        self._cache: OrderedDict[tuple, DynamicCache] = OrderedDict()
        self._max_entries = max_entries
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def store(self, token_ids: list[int], dynamic_cache: DynamicCache,
              session_id: str = "default") -> None:
        """Store a prompt's KV cache."""
        key = tuple(token_ids)
        if key in self._cache:
            return  # already cached

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
            self.eviction_count += 1

        # Deep copy the cache so it's not affected by ongoing generation
        stored = DynamicCache()
        for layer in dynamic_cache.layers:
            stored.update(
                layer.keys.clone(),
                layer.values.clone(),
                len(stored.layers),
            )

        self._cache[key] = stored
        logger.debug(f"[{session_id}] Stored cache for {len(token_ids)} tokens")

    def restore(self, token_ids: list[int],
                session_id: str = "default") -> tuple[int, DynamicCache | None]:
        """Look up exact match for prompt. Returns (tokens_matched, cache)."""
        key = tuple(token_ids)
        if key in self._cache:
            self._cache.move_to_end(key)  # LRU: mark as recently used
            self.hit_count += 1
            dc = self._cache[key]
            logger.debug(f"[{session_id}] Cache hit: {len(token_ids)} tokens")
            return len(token_ids), dc
        self.miss_count += 1
        return 0, None

    def release(self, token_ids: list[int], session_id: str = "default") -> None:
        """No-op for simple cache (no ref counting needed)."""
        pass

    @property
    def hit_rate_info(self) -> dict:
        total = self.hit_count + self.miss_count
        return {
            "total_blocks": self._max_entries,
            "used_blocks": len(self._cache),
            "free_blocks": self._max_entries - len(self._cache),
            "utilization": len(self._cache) / self._max_entries if self._max_entries > 0 else 0,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total > 0 else 0,
            "eviction_count": self.eviction_count,
        }
