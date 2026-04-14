"""Cache manager — ties block manager and radix tree into one interface."""

import logging

import torch

from inference_server.kv_cache.block import Block
from inference_server.kv_cache.block_manager import BlockManager
from inference_server.kv_cache.radix_tree import RadixTree

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified interface for KV cache prefix lookup, storage, and release."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_manager = BlockManager(num_blocks, block_size)
        self.radix_tree = RadixTree()
        self.block_size = block_size

    def lookup(self, token_ids: list[int]) -> tuple[int, list[Block]]:
        """Find the longest cached prefix. Returns (tokens_matched, blocks)."""
        matched, blocks = self.radix_tree.find_prefix(token_ids)
        if matched > 0:
            # Increment ref counts so these blocks aren't evicted while in use
            for block in blocks:
                block.acquire()
            logger.debug(f"Cache hit: {matched} tokens matched from {len(blocks)} blocks")
        return matched, blocks

    def store(self, token_ids: list[int], kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
              skip_tokens: int = 0) -> list[Block]:
        """Store KV state for a sequence in the cache. Returns allocated blocks."""
        # Only store the tokens we don't already have cached
        new_token_ids = token_ids[skip_tokens:]
        if not new_token_ids:
            return []

        num_blocks_needed = self.block_manager.blocks_needed(len(new_token_ids))

        if not self.block_manager.can_allocate(num_blocks_needed):
            logger.debug(f"Cannot allocate {num_blocks_needed} blocks, skipping cache store")
            return []

        blocks = self.block_manager.allocate(num_blocks_needed)

        # Distribute tokens across blocks
        token_offset = 0
        for block in blocks:
            end = min(token_offset + self.block_size, len(new_token_ids))
            block.token_ids = new_token_ids[token_offset:end]
            token_offset = end

        # Store KV tensors in blocks if provided
        if kv_tensors:
            self._store_kv_in_blocks(blocks, kv_tensors, skip_tokens)

        # Insert the full sequence into the radix tree
        # Get existing blocks for the prefix portion
        _, prefix_blocks = self.radix_tree.find_prefix(token_ids[:skip_tokens])
        all_blocks = list(prefix_blocks) + blocks
        self.radix_tree.insert(token_ids, all_blocks)

        logger.debug(f"Stored {len(new_token_ids)} tokens in {len(blocks)} new blocks")
        return blocks

    def release(self, token_ids: list[int]) -> None:
        """Release blocks for a completed request (decrement ref counts)."""
        _, blocks = self.radix_tree.find_prefix(token_ids)
        for block in blocks:
            block.release()

    def build_kv_from_blocks(self, blocks: list[Block]) -> object | None:
        """Reconstruct a KV cache tuple from cached blocks."""
        if not blocks or blocks[0].k_tensor is None:
            return None

        # Each block stores per-layer K and V slices
        # Concatenate across blocks to rebuild the full KV cache
        num_layers = blocks[0].k_tensor.shape[0]
        layer_kv = []
        for layer in range(num_layers):
            k_parts = [b.k_tensor[layer] for b in blocks if b.k_tensor is not None]
            v_parts = [b.v_tensor[layer] for b in blocks if b.v_tensor is not None]
            if k_parts and v_parts:
                k = torch.cat(k_parts, dim=1)  # concat along sequence dim
                v = torch.cat(v_parts, dim=1)
                layer_kv.append((k, v))

        return tuple(layer_kv) if layer_kv else None

    def _store_kv_in_blocks(
        self, blocks: list[Block],
        kv_tensors: list[tuple[torch.Tensor, torch.Tensor]],
        skip_tokens: int,
    ) -> None:
        """Slice KV tensors and store them in blocks."""
        # kv_tensors is a tuple of (K, V) per layer
        # K shape: [num_heads, seq_len, head_dim]
        num_layers = len(kv_tensors)
        token_offset = skip_tokens

        for block in blocks:
            num_tokens = block.num_tokens_stored
            end = token_offset + num_tokens

            # Stack all layers' K and V slices for this block
            k_slices = []
            v_slices = []
            for layer_idx in range(num_layers):
                k, v = kv_tensors[layer_idx]
                k_slices.append(k[:, token_offset:end, :])
                v_slices.append(v[:, token_offset:end, :])

            block.k_tensor = torch.stack(k_slices)  # [num_layers, num_heads, block_tokens, head_dim]
            block.v_tensor = torch.stack(v_slices)
            token_offset = end

    @property
    def hit_rate_info(self) -> dict:
        """Return cache utilization stats."""
        return {
            "total_blocks": self.block_manager.total_blocks,
            "used_blocks": self.block_manager.used_blocks,
            "free_blocks": self.block_manager.free_blocks,
            "utilization": self.block_manager.utilization,
        }
