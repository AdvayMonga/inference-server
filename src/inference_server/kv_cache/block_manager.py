"""Block manager — allocates and frees fixed-size KV cache blocks from a pool."""

import logging
import time

import torch

from inference_server.kv_cache.block import Block, BlockPoolView

logger = logging.getLogger(__name__)


class BlockManager:
    """Manages a pre-allocated pool of KV cache blocks.

    Each layer gets its own (num_blocks, n_kv_heads, block_size, head_dim) tensor.
    Per-layer pools (rather than one stacked tensor) let us support hybrid
    architectures where different layers have different KV shapes
    (e.g. Gemma 4: most layers head_dim=256, every 5th layer head_dim=512).
    """

    def __init__(
        self, num_blocks: int, block_size: int,
        layer_shapes: list[tuple[int, int]] | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.block_size = block_size
        self.layer_shapes = layer_shapes  # list of (n_kv_heads, head_dim) per layer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        if layer_shapes is not None:
            self.k_pools: list[torch.Tensor] = [
                torch.zeros(num_blocks, n_heads, block_size, head_dim,
                            device=self.device, dtype=self.dtype)
                for (n_heads, head_dim) in layer_shapes
            ]
            self.v_pools: list[torch.Tensor] = [
                torch.zeros(num_blocks, n_heads, block_size, head_dim,
                            device=self.device, dtype=self.dtype)
                for (n_heads, head_dim) in layer_shapes
            ]
            self.pool_view = BlockPoolView(self.k_pools, self.v_pools)
            total_bytes = sum(p.numel() * p.element_size() for p in self.k_pools + self.v_pools)
            logger.info(
                f"Pre-allocated KV pool: {num_blocks} blocks × {len(layer_shapes)} layers, "
                f"{total_bytes / 1e9:.2f} GB on {self.device}"
            )
        else:
            # Pool-less mode (legacy / tests that don't run a model)
            self.k_pools = []
            self.v_pools = []
            self.pool_view = None

        self.blocks: dict[int, Block] = {
            i: Block(block_id=i, block_size=block_size, pool_view=self.pool_view)
            for i in range(num_blocks)
        }
        self._free_ids: set[int] = set(range(num_blocks))

    @property
    def total_blocks(self) -> int:
        return len(self.blocks)

    @property
    def free_blocks(self) -> int:
        return len(self._free_ids)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    @property
    def utilization(self) -> float:
        """Fraction of blocks currently in use (0.0 to 1.0)."""
        if self.total_blocks == 0:
            return 0.0
        return self.used_blocks / self.total_blocks

    def allocate(self, count: int = 1) -> list[Block]:
        """Allocate `count` free blocks. Raises if not enough available."""
        if count > self.free_blocks:
            raise MemoryError(
                f"Requested {count} blocks but only {self.free_blocks} available"
            )

        allocated = []
        for _ in range(count):
            block_id = self._free_ids.pop()
            block = self.blocks[block_id]
            block.acquire()
            block.last_accessed = time.time()
            allocated.append(block)

        logger.debug(f"Allocated {count} blocks, {self.free_blocks} remaining")
        return allocated

    def free(self, blocks: list[Block]):
        """Release blocks back to the pool. Only frees blocks with ref_count 0."""
        for block in blocks:
            block.release()
            if block.ref_count == 0:
                block.clear()
                self._free_ids.add(block.block_id)

        logger.debug(f"Freed blocks, {self.free_blocks} now available")

    def get_block(self, block_id: int) -> Block:
        """Get a block by ID."""
        return self.blocks[block_id]

    def blocks_needed(self, num_tokens: int) -> int:
        """Calculate how many blocks are needed for a given number of tokens."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if enough free blocks exist for the given token count."""
        return self.blocks_needed(num_tokens) <= self.free_blocks
