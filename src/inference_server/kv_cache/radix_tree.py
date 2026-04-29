"""Radix tree — compressed prefix trie for fast KV cache prefix lookup."""

from __future__ import annotations

from dataclasses import dataclass, field

from inference_server.kv_cache.block import Block


@dataclass
class RadixNode:
    """A node in the radix tree. Stores a token sequence and its cached blocks."""

    token_ids: list[int] = field(default_factory=list)
    children: dict[int, RadixNode] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    ref_count: int = 0

    def acquire(self):
        """Mark this node as in use by a request."""
        self.ref_count += 1
        for block in self.blocks:
            block.acquire()

    def release(self):
        """Release this node from a request."""
        self.ref_count = max(0, self.ref_count - 1)
        for block in self.blocks:
            block.release()


class RadixTree:
    """Compressed prefix trie for indexing cached KV blocks by token sequence."""

    def __init__(self):
        self.root = RadixNode()

    def insert(self, token_ids: list[int], blocks: list[Block]) -> None:
        """Insert a token sequence and its associated KV blocks into the tree."""
        if not token_ids:
            return

        node = self.root
        remaining = list(token_ids)
        block_idx = 0

        while remaining:
            first_token = remaining[0]

            if first_token not in node.children:
                # No matching child — create a new leaf with all remaining tokens
                tokens_for_node = remaining
                blocks_for_node = blocks[block_idx:]
                new_node = RadixNode(
                    token_ids=tokens_for_node, blocks=blocks_for_node
                )
                node.children[first_token] = new_node
                return

            child = node.children[first_token]

            # Find how many tokens match between remaining and child's edge
            match_len = 0
            while (
                match_len < len(child.token_ids)
                and match_len < len(remaining)
                and child.token_ids[match_len] == remaining[match_len]
            ):
                match_len += 1

            if match_len == len(child.token_ids):
                # Full match on this edge — advance past it
                remaining = remaining[match_len:]
                # Count how many blocks this matched edge covers
                block_idx += len(child.blocks)
                node = child
            else:
                # Partial match — try to split the edge at a block boundary
                actual_split = self._split_node(node, child, first_token, match_len)
                if actual_split == 0:
                    # Could not split (first block straddles boundary).
                    # Bail: don't index the rest of this sequence. Safe but loses sharing.
                    return
                remaining = remaining[actual_split:]
                block_idx += len(node.children[first_token].blocks)
                node = node.children[first_token]

        # If we consumed all tokens and landed on an existing node, update its blocks
        if not node.blocks:
            node.blocks = blocks[block_idx:]

    def find_prefix(self, token_ids: list[int]) -> tuple[int, list[Block]]:
        """Find the longest cached prefix. Returns (num_matched_tokens, matched_blocks)."""
        if not token_ids:
            return 0, []

        node = self.root
        remaining = list(token_ids)
        matched_tokens = 0
        matched_blocks: list[Block] = []

        while remaining:
            first_token = remaining[0]

            if first_token not in node.children:
                break

            child = node.children[first_token]

            match_len = 0
            while (
                match_len < len(child.token_ids)
                and match_len < len(remaining)
                and child.token_ids[match_len] == remaining[match_len]
            ):
                match_len += 1

            if match_len == 0:
                break

            if match_len == len(child.token_ids):
                # Full edge match
                matched_tokens += match_len
                matched_blocks.extend(child.blocks)
                remaining = remaining[match_len:]
                node = child
            else:
                # Partial edge match — reuse blocks that are fully covered
                # but not a partially filled block
                matched_tokens += match_len
                # Only include blocks whose tokens are fully matched
                tokens_covered = 0
                for block in child.blocks:
                    tokens_covered += block.num_tokens_stored
                    if tokens_covered <= match_len:
                        matched_blocks.append(block)
                    else:
                        break
                break

        return matched_tokens, matched_blocks

    def remove(self, token_ids: list[int]) -> list[Block]:
        """Remove a sequence from the tree. Returns the freed blocks."""
        if not token_ids:
            return []

        # Walk to the leaf, collecting the path
        path: list[tuple[RadixNode, int]] = []  # (parent, first_token_of_child)
        node = self.root
        remaining = list(token_ids)

        while remaining:
            first_token = remaining[0]
            if first_token not in node.children:
                return []

            child = node.children[first_token]
            path.append((node, first_token))

            match_len = 0
            while (
                match_len < len(child.token_ids)
                and match_len < len(remaining)
                and child.token_ids[match_len] == remaining[match_len]
            ):
                match_len += 1

            if match_len < len(child.token_ids):
                return []  # Sequence not fully in tree

            remaining = remaining[match_len:]
            node = child

        # Collect all blocks from the leaf
        freed_blocks = list(node.blocks)

        # Remove leaf and clean up empty parents
        if not node.children:
            # It's a leaf — remove it
            parent, key = path[-1]
            del parent.children[key]

            # Merge parent with its only remaining child if possible
            if len(parent.children) == 1 and parent is not self.root:
                self._merge_single_child(path)
        else:
            # Has children — just clear its blocks
            node.blocks = []

        return freed_blocks

    def _split_node(
        self, parent: RadixNode, child: RadixNode, edge_key: int, split_at: int
    ) -> int:
        """Split a child edge at the largest block boundary <= split_at.

        Returns the actual split position (block-aligned). Returns 0 if no clean split
        is possible (first block straddles the split point) — caller must handle.
        """
        # Find largest block-aligned position <= split_at
        aligned_split = 0
        blocks_in_mid = 0
        for block in child.blocks:
            next_pos = aligned_split + block.num_tokens_stored
            if next_pos <= split_at:
                aligned_split = next_pos
                blocks_in_mid += 1
            else:
                break

        if aligned_split == 0:
            return 0

        mid_node = RadixNode(
            token_ids=child.token_ids[:aligned_split],
            blocks=child.blocks[:blocks_in_mid],
        )
        child.token_ids = child.token_ids[aligned_split:]
        child.blocks = child.blocks[blocks_in_mid:]

        mid_node.children[child.token_ids[0]] = child
        parent.children[edge_key] = mid_node
        return aligned_split

    def _blocks_for_tokens(self, node: RadixNode, num_tokens: int) -> int:
        """Count how many blocks cover the first num_tokens of a node."""
        tokens_covered = 0
        for i, block in enumerate(node.blocks):
            tokens_covered += block.num_tokens_stored
            if tokens_covered >= num_tokens:
                return i + 1
        return len(node.blocks)

    def _merge_single_child(
        self, path: list[tuple[RadixNode, int]]
    ) -> None:
        """Merge a node with its only child after a deletion."""
        if len(path) < 2:
            return

        parent, parent_key = path[-1]
        if len(parent.children) != 1 or parent is self.root:
            return

        only_key = next(iter(parent.children))
        only_child = parent.children[only_key]

        # Merge: parent absorbs child's tokens and children
        parent.token_ids = parent.token_ids + only_child.token_ids
        parent.blocks = parent.blocks + only_child.blocks
        parent.children = only_child.children
