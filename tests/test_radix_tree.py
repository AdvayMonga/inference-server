"""Tests for the radix tree prefix lookup."""

import pytest

from inference_server.kv_cache.block import Block
from inference_server.kv_cache.radix_tree import RadixTree


def make_blocks(token_lists: list[list[int]]) -> list[Block]:
    """Helper — create blocks with token_ids set."""
    blocks = []
    for i, tokens in enumerate(token_lists):
        b = Block(block_id=i, block_size=16)
        b.token_ids = tokens
        blocks.append(b)
    return blocks


# --- Insert and find tests ---

def test_insert_and_find_exact():
    tree = RadixTree()
    blocks = make_blocks([[10, 20], [30, 40]])
    tree.insert([10, 20, 30, 40], blocks)

    matched, found_blocks = tree.find_prefix([10, 20, 30, 40])
    assert matched == 4
    assert len(found_blocks) == 2


def test_find_partial_prefix():
    tree = RadixTree()
    blocks = make_blocks([[10, 20, 30, 40, 50]])
    tree.insert([10, 20, 30, 40, 50], blocks)

    matched, found_blocks = tree.find_prefix([10, 20, 30, 99, 99])
    assert matched == 3


def test_find_no_match():
    tree = RadixTree()
    blocks = make_blocks([[10, 20]])
    tree.insert([10, 20], blocks)

    matched, found_blocks = tree.find_prefix([99, 88])
    assert matched == 0
    assert found_blocks == []


def test_find_empty_tree():
    tree = RadixTree()
    matched, found_blocks = tree.find_prefix([10, 20])
    assert matched == 0
    assert found_blocks == []


def test_find_empty_query():
    tree = RadixTree()
    blocks = make_blocks([[10, 20]])
    tree.insert([10, 20], blocks)

    matched, found_blocks = tree.find_prefix([])
    assert matched == 0


# --- Multiple sequences with shared prefix ---

def test_shared_prefix_two_sequences():
    tree = RadixTree()
    blocks_a = make_blocks([[10, 20], [30, 40]])
    blocks_b = make_blocks([[10, 20], [50, 60]])
    tree.insert([10, 20, 30, 40], blocks_a)
    tree.insert([10, 20, 50, 60], blocks_b)

    # Query matching sequence A
    matched, _ = tree.find_prefix([10, 20, 30, 40])
    assert matched == 4

    # Query matching sequence B
    matched, _ = tree.find_prefix([10, 20, 50, 60])
    assert matched == 4

    # Query matching only the shared prefix
    matched, _ = tree.find_prefix([10, 20, 99])
    assert matched == 2


def test_three_sequences_branching():
    # One token per block so every position is a valid split boundary.
    tree = RadixTree()
    tree.insert([1, 2, 3], make_blocks([[1], [2], [3]]))
    tree.insert([1, 2, 4], make_blocks([[1], [2], [4]]))
    tree.insert([1, 5], make_blocks([[1], [5]]))

    matched, _ = tree.find_prefix([1, 2, 3])
    assert matched == 3

    matched, _ = tree.find_prefix([1, 2, 4])
    assert matched == 3

    matched, _ = tree.find_prefix([1, 5])
    assert matched == 2

    matched, _ = tree.find_prefix([1, 2])
    assert matched == 2

    matched, _ = tree.find_prefix([1])
    assert matched == 1


# --- Remove tests ---

def test_remove_sequence():
    tree = RadixTree()
    blocks = make_blocks([[10, 20]])
    tree.insert([10, 20], blocks)

    freed = tree.remove([10, 20])
    assert len(freed) > 0

    # Should no longer match
    matched, _ = tree.find_prefix([10, 20])
    assert matched == 0


def test_remove_one_of_two_branches():
    tree = RadixTree()
    tree.insert([1, 2, 3], make_blocks([[1], [2], [3]]))
    tree.insert([1, 2, 4], make_blocks([[1], [2], [4]]))

    tree.remove([1, 2, 3])

    # Removed branch should not match
    matched, _ = tree.find_prefix([1, 2, 3])
    assert matched == 2  # shared prefix [1, 2] still exists

    # Other branch should still work
    matched, _ = tree.find_prefix([1, 2, 4])
    assert matched == 3


def test_remove_nonexistent():
    tree = RadixTree()
    tree.insert([1, 2, 3], make_blocks([[1, 2, 3]]))

    freed = tree.remove([9, 9, 9])
    assert freed == []
