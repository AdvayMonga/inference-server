"""Tests for eviction policies and eviction integration with cache manager."""

import time

import pytest

from inference_server.kv_cache.block import Block
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.kv_cache.eviction import (
    AttentionSinkLRUPolicy,
    H2OPolicy,
    LRUPolicy,
    create_eviction_policy,
)


# --- LRU Policy unit tests ---

def test_lru_selects_oldest():
    policy = LRUPolicy()
    b1 = Block(block_id=0, block_size=16)
    b2 = Block(block_id=1, block_size=16)
    b1.last_accessed = 100.0
    b2.last_accessed = 200.0

    victim = policy.select_victim([b1, b2])
    assert victim.block_id == 0  # b1 is older


def test_lru_returns_none_for_empty():
    policy = LRUPolicy()
    assert policy.select_victim([]) is None


def test_lru_on_access_updates_timestamp():
    policy = LRUPolicy()
    block = Block(block_id=0, block_size=16)
    block.last_accessed = 0.0
    policy.on_access(block)
    assert block.last_accessed > 0.0


def test_create_eviction_policy_lru():
    policy = create_eviction_policy("lru")
    assert isinstance(policy, LRUPolicy)


def test_create_eviction_policy_unknown():
    with pytest.raises(ValueError, match="Unknown"):
        create_eviction_policy("nonexistent")


# --- Eviction integration with CacheManager ---

def test_eviction_triggers_when_full():
    """When cache is full, storing new data should evict old blocks."""
    cache = CacheManager(num_blocks=2, block_size=4, eviction_policy="lru")

    # Fill the cache (2 blocks = 8 tokens)
    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)
    time.sleep(0.01)  # ensure different timestamps
    cache.store([5, 6, 7, 8], kv_tensors=[], skip_tokens=0)

    assert cache.block_manager.free_blocks == 0

    # Storing more should trigger eviction and succeed
    cache.store([9, 10, 11, 12], kv_tensors=[], skip_tokens=0)
    assert cache._eviction_count >= 1


def test_eviction_frees_least_recently_used():
    """LRU should evict the block that was accessed longest ago."""
    cache = CacheManager(num_blocks=2, block_size=4, eviction_policy="lru")

    # Store two sequences with different access times
    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)
    time.sleep(0.01)
    cache.store([5, 6, 7, 8], kv_tensors=[], skip_tokens=0)

    # Access the first one again to make it more recent
    cache.lookup([1, 2, 3, 4])
    # Release so it's evictable
    cache.release([1, 2, 3, 4])

    # Now store new data — should evict [5,6,7,8] (older access)
    cache.store([9, 10, 11, 12], kv_tensors=[], skip_tokens=0)

    # The recently accessed [1,2,3,4] should still be findable
    matched, _ = cache.lookup([1, 2, 3, 4])
    assert matched == 4


def test_eviction_skips_blocks_in_use():
    """Blocks with ref_count > 0 must not be evicted."""
    cache = CacheManager(num_blocks=2, block_size=4, eviction_policy="lru")

    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)
    cache.store([5, 6, 7, 8], kv_tensors=[], skip_tokens=0)

    # Hold a reference to both blocks (simulate in-flight requests)
    cache.lookup([1, 2, 3, 4])  # increments ref_count
    cache.lookup([5, 6, 7, 8])  # increments ref_count

    # Try to store — should fail because all blocks are in use
    result = cache.store([9, 10, 11, 12], kv_tensors=[], skip_tokens=0)
    assert result == []


def test_hit_rate_tracking():
    cache = CacheManager(num_blocks=10, block_size=4, eviction_policy="lru")

    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)

    cache.lookup([1, 2, 3, 4])  # hit
    cache.lookup([9, 9, 9, 9])  # miss
    cache.lookup([1, 2, 3, 4])  # hit

    info = cache.hit_rate_info
    assert info["hit_count"] == 2
    assert info["miss_count"] == 1
    assert abs(info["hit_rate"] - 2 / 3) < 0.01


def test_session_id_passed_through():
    """session_id should be accepted without error (multi-user ready)."""
    cache = CacheManager(num_blocks=10, block_size=4, eviction_policy="lru")
    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0, session_id="user-123")
    matched, _ = cache.lookup([1, 2, 3, 4], session_id="user-123")
    assert matched == 4
    cache.release([1, 2, 3, 4], session_id="user-123")


# --- Attention Sink LRU Policy unit tests ---

def test_attention_sink_skips_first_block():
    policy = AttentionSinkLRUPolicy()
    b1 = Block(block_id=0, block_size=16, is_first_block=True)
    b2 = Block(block_id=1, block_size=16, is_first_block=False)
    b1.last_accessed = 100.0  # older
    b2.last_accessed = 200.0

    victim = policy.select_victim([b1, b2])
    assert victim.block_id == 1  # b1 is older but protected


def test_attention_sink_returns_none_if_all_protected():
    policy = AttentionSinkLRUPolicy()
    b1 = Block(block_id=0, block_size=16, is_first_block=True)
    b2 = Block(block_id=1, block_size=16, is_first_block=True)

    assert policy.select_victim([b1, b2]) is None


def test_attention_sink_selects_oldest_non_protected():
    policy = AttentionSinkLRUPolicy()
    b1 = Block(block_id=0, block_size=16, is_first_block=True)
    b2 = Block(block_id=1, block_size=16, is_first_block=False)
    b3 = Block(block_id=2, block_size=16, is_first_block=False)
    b1.last_accessed = 50.0
    b2.last_accessed = 100.0
    b3.last_accessed = 200.0

    victim = policy.select_victim([b1, b2, b3])
    assert victim.block_id == 1  # oldest non-protected


def test_create_attention_sink_policy():
    policy = create_eviction_policy("attention_sink_lru")
    assert isinstance(policy, AttentionSinkLRUPolicy)


# --- Attention Sink integration with CacheManager ---

def test_attention_sink_protects_first_block_during_eviction():
    """First block of a sequence should survive eviction."""
    cache = CacheManager(num_blocks=3, block_size=4, eviction_policy="attention_sink_lru")

    # Store a sequence that uses 2 blocks (first block is protected)
    cache.store([1, 2, 3, 4, 5, 6, 7, 8], kv_tensors=[], skip_tokens=0)
    time.sleep(0.01)
    # Store another that uses 1 block
    cache.store([9, 10, 11, 12], kv_tensors=[], skip_tokens=0)

    assert cache.block_manager.free_blocks == 0

    # Force eviction — should NOT evict the first block of [1,2,3,4,5,6,7,8]
    cache.store([20, 21, 22, 23], kv_tensors=[], skip_tokens=0)

    # First block of original sequence should still be findable
    matched, _ = cache.lookup([1, 2, 3, 4])
    assert matched >= 4  # at least the first block survived


def test_attention_sink_marks_first_block():
    """store() should mark the first block of a new sequence."""
    cache = CacheManager(num_blocks=10, block_size=4, eviction_policy="attention_sink_lru")
    cache.store([1, 2, 3, 4, 5, 6, 7, 8], kv_tensors=[], skip_tokens=0)

    # Find the blocks that were stored
    _, blocks = cache.lookup([1, 2, 3, 4, 5, 6, 7, 8])
    first_block_flags = [b.is_first_block for b in blocks]
    assert first_block_flags[0] is True
    if len(first_block_flags) > 1:
        assert first_block_flags[1] is False


# --- H2O Policy unit tests ---

def test_h2o_evicts_lowest_score():
    policy = H2OPolicy()
    b1 = Block(block_id=0, block_size=16)
    b2 = Block(block_id=1, block_size=16)
    b3 = Block(block_id=2, block_size=16)

    policy.record_attention(b1, 10.0)
    policy.record_attention(b2, 2.0)
    policy.record_attention(b3, 50.0)

    victim = policy.select_victim([b1, b2, b3])
    assert victim.block_id == 1  # lowest score


def test_h2o_accumulates_scores():
    policy = H2OPolicy()
    block = Block(block_id=0, block_size=16)

    policy.record_attention(block, 5.0)
    policy.record_attention(block, 3.0)
    policy.record_attention(block, 2.0)

    assert policy.get_score(block) == 10.0


def test_h2o_protects_first_block():
    policy = H2OPolicy()
    b1 = Block(block_id=0, block_size=16, is_first_block=True)
    b2 = Block(block_id=1, block_size=16, is_first_block=False)

    policy.record_attention(b1, 1.0)  # low score but protected
    policy.record_attention(b2, 100.0)  # high score

    victim = policy.select_victim([b1, b2])
    assert victim.block_id == 1  # b1 protected even with lower score


def test_h2o_returns_none_if_all_protected():
    policy = H2OPolicy()
    b1 = Block(block_id=0, block_size=16, is_first_block=True)
    assert policy.select_victim([b1]) is None


def test_h2o_zero_score_evicted_first():
    policy = H2OPolicy()
    b1 = Block(block_id=0, block_size=16)
    b2 = Block(block_id=1, block_size=16)

    # b1 has no recorded attention, b2 has some
    policy.record_attention(b2, 5.0)

    victim = policy.select_victim([b1, b2])
    assert victim.block_id == 0  # zero score = first to go


def test_h2o_clear_scores():
    policy = H2OPolicy()
    block = Block(block_id=0, block_size=16)
    policy.record_attention(block, 10.0)
    assert policy.get_score(block) == 10.0

    policy.clear_scores(block)
    assert policy.get_score(block) == 0.0


def test_create_h2o_policy():
    policy = create_eviction_policy("h2o")
    assert isinstance(policy, H2OPolicy)


# --- H2O integration with CacheManager ---

def test_h2o_eviction_with_cache_manager():
    cache = CacheManager(num_blocks=2, block_size=4, eviction_policy="h2o")

    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)
    time.sleep(0.01)
    cache.store([5, 6, 7, 8], kv_tensors=[], skip_tokens=0)

    # Record attention — make [1,2,3,4] important, [5,6,7,8] unimportant
    for block in cache.block_manager.blocks.values():
        if block.token_ids == [1, 2, 3, 4]:
            cache.eviction_policy.record_attention(block, 100.0)
        elif block.token_ids == [5, 6, 7, 8]:
            cache.eviction_policy.record_attention(block, 1.0)

    # Force eviction — should evict [5,6,7,8] (low attention)
    cache.store([9, 10, 11, 12], kv_tensors=[], skip_tokens=0)

    # High-attention block should survive
    matched, _ = cache.lookup([1, 2, 3, 4])
    assert matched == 4
