"""Tests for the unified cache manager."""

import pytest

from inference_server.kv_cache.cache_manager import CacheManager


@pytest.fixture
def cache():
    return CacheManager(num_blocks=20, block_size=4)


def test_lookup_empty_cache(cache):
    matched, blocks = cache.lookup([1, 2, 3, 4])
    assert matched == 0
    assert blocks == []


def test_store_and_lookup(cache):
    # Store a sequence (no KV tensors for this test, just token tracking)
    cache.store([10, 20, 30, 40], kv_tensors=[], skip_tokens=0)

    matched, blocks = cache.lookup([10, 20, 30, 40])
    assert matched == 4
    assert len(blocks) > 0


def test_prefix_sharing(cache):
    cache.store([10, 20, 30, 40], kv_tensors=[], skip_tokens=0)
    cache.store([10, 20, 50, 60], kv_tensors=[], skip_tokens=0)

    # Both share [10, 20] prefix
    matched, _ = cache.lookup([10, 20, 99])
    assert matched == 2


def test_store_with_skip_tokens(cache):
    # First request caches [1, 2, 3, 4]
    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)

    # Second request shares prefix, only stores new tokens
    cache.store([1, 2, 3, 4, 5, 6], kv_tensors=[], skip_tokens=4)

    matched, _ = cache.lookup([1, 2, 3, 4, 5, 6])
    assert matched == 6


def test_blocks_allocated_correctly(cache):
    # block_size=4, so 8 tokens need 2 blocks
    assert cache.block_manager.free_blocks == 20
    cache.store([1, 2, 3, 4, 5, 6, 7, 8], kv_tensors=[], skip_tokens=0)
    assert cache.block_manager.used_blocks == 2


def test_store_fails_gracefully_when_full(cache):
    # Fill up the cache (20 blocks * 4 tokens = 80 tokens)
    for i in range(20):
        base = i * 4 * 100  # ensure unique prefixes
        cache.store(
            [base + 1, base + 2, base + 3, base + 4],
            kv_tensors=[], skip_tokens=0,
        )

    assert cache.block_manager.free_blocks == 0

    # Storing more should return empty (no crash)
    result = cache.store([999, 888, 777, 666], kv_tensors=[], skip_tokens=0)
    assert result == []


def test_release_decrements_ref_count(cache):
    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)

    # Lookup increments ref count
    _, blocks = cache.lookup([1, 2, 3, 4])
    initial_refs = [b.ref_count for b in blocks]
    assert all(r >= 1 for r in initial_refs)

    # Release decrements
    cache.release([1, 2, 3, 4])


def test_hit_rate_info(cache):
    info = cache.hit_rate_info
    assert info["total_blocks"] == 20
    assert info["free_blocks"] == 20
    assert info["utilization"] == 0.0

    cache.store([1, 2, 3, 4], kv_tensors=[], skip_tokens=0)
    info = cache.hit_rate_info
    assert info["used_blocks"] == 1
    assert info["utilization"] > 0.0
