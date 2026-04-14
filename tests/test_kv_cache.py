"""Tests for KV cache block manager."""

import pytest

from inference_server.kv_cache.block import Block
from inference_server.kv_cache.block_manager import BlockManager


# --- Block tests ---

def test_block_starts_free():
    block = Block(block_id=0, block_size=16)
    assert block.is_free
    assert block.ref_count == 0
    assert block.num_tokens_stored == 0


def test_block_acquire_release():
    block = Block(block_id=0, block_size=16)
    block.acquire()
    assert block.ref_count == 1
    assert not block.is_free
    block.release()
    assert block.ref_count == 0


def test_block_clear_resets_state():
    block = Block(block_id=0, block_size=16)
    block.acquire()
    block.token_ids = [1, 2, 3]
    block.clear()
    assert block.is_free
    assert block.k_tensor is None
    assert block.token_ids == []


# --- BlockManager tests ---

def test_manager_initial_state():
    manager = BlockManager(num_blocks=10, block_size=16)
    assert manager.total_blocks == 10
    assert manager.free_blocks == 10
    assert manager.used_blocks == 0
    assert manager.utilization == 0.0


def test_allocate_single_block():
    manager = BlockManager(num_blocks=10, block_size=16)
    blocks = manager.allocate(1)
    assert len(blocks) == 1
    assert manager.free_blocks == 9
    assert blocks[0].ref_count == 1


def test_allocate_multiple_blocks():
    manager = BlockManager(num_blocks=10, block_size=16)
    blocks = manager.allocate(5)
    assert len(blocks) == 5
    assert manager.free_blocks == 5
    assert manager.utilization == 0.5


def test_allocate_all_blocks():
    manager = BlockManager(num_blocks=4, block_size=16)
    blocks = manager.allocate(4)
    assert manager.free_blocks == 0
    assert manager.utilization == 1.0


def test_allocate_too_many_raises():
    manager = BlockManager(num_blocks=3, block_size=16)
    with pytest.raises(MemoryError, match="only 3 available"):
        manager.allocate(5)


def test_free_returns_blocks_to_pool():
    manager = BlockManager(num_blocks=5, block_size=16)
    blocks = manager.allocate(3)
    assert manager.free_blocks == 2
    manager.free(blocks)
    assert manager.free_blocks == 5


def test_free_respects_ref_count():
    manager = BlockManager(num_blocks=5, block_size=16)
    blocks = manager.allocate(1)
    block = blocks[0]
    # Simulate a second request sharing this block
    block.acquire()
    assert block.ref_count == 2
    # First release — ref_count drops to 1, block stays allocated
    manager.free([block])
    assert block.ref_count == 1
    assert manager.free_blocks == 4  # not freed yet
    # Second release — ref_count drops to 0, block is freed
    manager.free([block])
    assert block.ref_count == 0
    assert manager.free_blocks == 5


def test_blocks_needed_calculation():
    manager = BlockManager(num_blocks=10, block_size=16)
    assert manager.blocks_needed(1) == 1
    assert manager.blocks_needed(16) == 1
    assert manager.blocks_needed(17) == 2
    assert manager.blocks_needed(32) == 2
    assert manager.blocks_needed(100) == 7  # ceil(100/16)


def test_can_allocate():
    manager = BlockManager(num_blocks=5, block_size=16)
    assert manager.can_allocate(80)  # needs 5 blocks, has 5
    assert not manager.can_allocate(81)  # needs 6 blocks, has 5


def test_allocate_free_reuse():
    """Freed blocks should be reusable."""
    manager = BlockManager(num_blocks=2, block_size=16)
    blocks1 = manager.allocate(2)
    assert manager.free_blocks == 0
    manager.free(blocks1)
    assert manager.free_blocks == 2
    blocks2 = manager.allocate(2)
    assert manager.free_blocks == 0
    assert len(blocks2) == 2
