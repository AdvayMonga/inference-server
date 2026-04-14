"""Block manager — allocates and frees fixed-size KV cache blocks from a pool."""

import logging
import time

from inference_server.kv_cache.block import Block

logger = logging.getLogger(__name__)


class BlockManager:
    """Manages a pre-allocated pool of KV cache blocks."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: dict[int, Block] = {
            i: Block(block_id=i, block_size=block_size) for i in range(num_blocks)
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
