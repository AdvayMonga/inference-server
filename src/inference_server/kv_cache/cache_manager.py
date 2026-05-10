"""Cache manager — ties block manager, radix tree, and eviction policy into one interface."""

import logging

import torch

from inference_server.kv_cache.block import Block
from inference_server.kv_cache.block_manager import BlockManager
from inference_server.kv_cache.eviction import EvictionPolicy, create_eviction_policy
from inference_server.kv_cache.radix_tree import RadixTree

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified interface for KV cache prefix lookup, storage, eviction, and release."""

    def __init__(
        self, num_blocks: int, block_size: int, eviction_policy: str = "lru",
        layer_shapes: list[tuple[int, int]] | None = None,
        device: str = "cpu",
        dtype=None,
    ):
        import torch
        if dtype is None:
            dtype = torch.bfloat16
        self.block_manager = BlockManager(
            num_blocks, block_size,
            layer_shapes=layer_shapes, device=device, dtype=dtype,
        )
        self.radix_tree = RadixTree()
        # Exact-match index: full prompt tokens -> all blocks for that prompt.
        # Catches the case where radix insert bails on mid-block divergence.
        self._exact_index: dict[tuple[int, ...], list[Block]] = {}
        self.eviction_policy: EvictionPolicy = create_eviction_policy(eviction_policy)
        self.block_size = block_size
        self._eviction_count = 0
        self._hit_count = 0
        self._miss_count = 0

    def lookup(self, token_ids: list[int], session_id: str = "default") -> tuple[int, list[Block]]:
        """Find the longest cached prefix. Returns (tokens_matched, blocks).

        Guarantees matched == sum of returned blocks' token counts (block-aligned).
        Tries the exact-match index first, then falls back to radix prefix matching.
        """
        # Layer 1: exact full-prompt match. Always block-aligned (covers all stored tokens).
        exact_blocks = self._exact_index.get(tuple(token_ids))
        if exact_blocks is not None and all(b.k_tensor is not None for b in exact_blocks):
            matched = sum(b.num_tokens_stored for b in exact_blocks)
            for block in exact_blocks:
                block.acquire()
                self.eviction_policy.on_access(block)
            self._hit_count += 1
            logger.debug(f"[{session_id}] Exact cache hit: {matched} tokens from {len(exact_blocks)} blocks")
            return matched, list(exact_blocks)

        matched, blocks = self.radix_tree.find_prefix(token_ids)

        aligned_blocks: list[Block] = []
        running = 0
        for b in blocks:
            n = b.num_tokens_stored
            if running + n > matched:
                break
            aligned_blocks.append(b)
            running += n

        matched = running
        blocks = aligned_blocks

        if matched > 0:
            for block in blocks:
                block.acquire()
                self.eviction_policy.on_access(block)
            self._hit_count += 1
            logger.debug(f"[{session_id}] Cache hit: {matched} tokens from {len(blocks)} blocks")
        else:
            self._miss_count += 1
        return matched, blocks

    def store(self, token_ids: list[int], kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
              skip_tokens: int = 0, session_id: str = "default") -> list[Block]:
        """Store KV state for a sequence. Evicts if necessary. Returns allocated blocks."""
        new_token_ids = token_ids[skip_tokens:]
        if not new_token_ids:
            return []

        num_blocks_needed = self.block_manager.blocks_needed(len(new_token_ids))

        # Try to evict if not enough space (can_allocate takes a token count)
        while not self.block_manager.can_allocate(len(new_token_ids)):
            evicted = self._evict_one()
            if not evicted:
                logger.debug(f"[{session_id}] Cannot evict — all blocks in use, skipping store")
                return []

        blocks = self.block_manager.allocate(num_blocks_needed)

        # Mark the first block of a new sequence for attention sink protection
        if skip_tokens == 0 and blocks:
            blocks[0].is_first_block = True

        # Distribute tokens across blocks
        token_offset = 0
        for block in blocks:
            end = min(token_offset + self.block_size, len(new_token_ids))
            block.token_ids = new_token_ids[token_offset:end]
            self.eviction_policy.on_access(block)
            token_offset = end

        if kv_tensors:
            self._store_kv_in_blocks(blocks, kv_tensors, skip_tokens)

        _, prefix_blocks = self.radix_tree.find_prefix(token_ids[:skip_tokens])
        all_blocks = list(prefix_blocks) + blocks
        self.radix_tree.insert(token_ids, all_blocks)

        # Layer 1 index: exact full-prompt entry (always succeeds, even when radix bails).
        self._exact_index[tuple(token_ids)] = all_blocks

        # Release the allocation ref — blocks are now in cache, not in active use
        for block in blocks:
            block.release()

        logger.debug(f"[{session_id}] Stored {len(new_token_ids)} tokens in {len(blocks)} blocks")
        return blocks

    def release(self, token_ids: list[int], session_id: str = "default") -> None:
        """Release blocks for a completed request (decrement ref counts)."""
        exact_blocks = self._exact_index.get(tuple(token_ids))
        if exact_blocks is not None:
            for block in exact_blocks:
                block.release()
            return
        _, blocks = self.radix_tree.find_prefix(token_ids)
        for block in blocks:
            block.release()

    def _evict_one(self) -> bool:
        """Evict a single block using the eviction policy. Returns True if successful."""
        # Gather eviction candidates: blocks with ref_count == 0 and data stored
        candidates = [
            b for b in self.block_manager.blocks.values()
            if b.ref_count == 0 and not b.is_free
        ]

        victim = self.eviction_policy.select_victim(candidates)
        if victim is None:
            return False

        # Remove from radix tree
        if victim.token_ids:
            self.radix_tree.remove(victim.token_ids)

        # Drop any exact-match entries that referenced this block
        stale = [k for k, blocks in self._exact_index.items() if victim in blocks]
        for k in stale:
            del self._exact_index[k]

        # Free the block
        victim.clear()
        self.block_manager._free_ids.add(victim.block_id)
        self._eviction_count += 1
        logger.debug(f"Evicted block {victim.block_id}")
        return True

    def build_kv_from_blocks(self, blocks: list[Block]) -> object | None:
        """Reconstruct per-layer (K, V) tuples from cached blocks."""
        valid_blocks = [b for b in blocks if b.k_tensor is not None]
        if not valid_blocks:
            return None

        num_layers = len(valid_blocks[0].k_tensor)
        layer_kv = []
        for layer in range(num_layers):
            k_parts = [b.k_tensor[layer] for b in valid_blocks]
            v_parts = [b.v_tensor[layer] for b in valid_blocks]
            k = torch.cat(k_parts, dim=1)  # concat along seq_len dim
            v = torch.cat(v_parts, dim=1)
            layer_kv.append((k, v))

        return tuple(layer_kv) if layer_kv else None

    def _store_kv_in_blocks(
        self, blocks: list[Block],
        kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
        skip_tokens: int,
    ) -> None:
        """Copy KV tensors into the pre-allocated per-layer pools."""
        num_layers = len(kv_tensors)
        bm = self.block_manager
        token_offset = skip_tokens

        if bm.pool_view is None:
            # Pool-less mode kept for tests that don't run a model
            for block in blocks:
                num_tokens = block.num_tokens_stored
                end = token_offset + num_tokens
                block.token_ids  # noop; legacy path retained only to satisfy code paths
                token_offset = end
            return

        for block in blocks:
            num_tokens = block.num_tokens_stored
            end = token_offset + num_tokens
            for l in range(num_layers):
                # Source: per-layer 3D (n_heads, seq, head_dim) — already batch-stripped upstream
                src_k = kv_tensors[l][0][:, token_offset:end, :]
                src_v = kv_tensors[l][1][:, token_offset:end, :]
                # Dest: pool[l][block_id, :, :num_tokens, :] — shape (n_heads, num_tokens, head_dim)
                bm.k_pools[l][block.block_id, :, :num_tokens, :].copy_(src_k)
                bm.v_pools[l][block.block_id, :, :num_tokens, :].copy_(src_v)
            token_offset = end

    @property
    def free_blocks(self) -> int:
        return self.block_manager.free_blocks

    @property
    def total_blocks(self) -> int:
        return self.block_manager.total_blocks

    @property
    def pressure(self) -> float:
        return self.block_manager.utilization

    def blocks_needed(self, num_tokens: int) -> int:
        return self.block_manager.blocks_needed(num_tokens)

    @property
    def hit_rate_info(self) -> dict:
        """Return cache stats including hit rate and eviction count."""
        total_lookups = self._hit_count + self._miss_count
        return {
            "total_blocks": self.block_manager.total_blocks,
            "used_blocks": self.block_manager.used_blocks,
            "free_blocks": self.block_manager.free_blocks,
            "utilization": self.block_manager.utilization,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / total_lookups if total_lookups > 0 else 0.0,
            "eviction_count": self._eviction_count,
        }
