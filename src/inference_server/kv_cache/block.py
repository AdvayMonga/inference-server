"""KV cache block — fixed-size unit of cached attention state.

Tensors live in pre-allocated per-layer pools owned by BlockManager. Block
holds an integer block_id and reads/writes its slice of the pool through
properties, so existing call sites (cache_manager, eviction policies,
radix_tree) need no API changes.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class Block:
    """A fixed-size block holding K and V tensors for a range of tokens."""

    block_id: int
    block_size: int  # number of token positions in this block
    pool_view: "BlockPoolView | None" = None  # backref to pools; None until pool-aware ctor wires it
    token_ids: list[int] = field(default_factory=list)  # which tokens are stored here
    ref_count: int = 0  # number of active requests using this block
    last_accessed: float = 0.0  # timestamp for LRU eviction
    is_first_block: bool = False  # protected from eviction by attention sink policy

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0 and len(self.token_ids) == 0

    @property
    def num_tokens_stored(self) -> int:
        return len(self.token_ids)

    @property
    def k_tensor(self) -> list[torch.Tensor] | None:
        """List of per-layer K slices for this block's stored tokens. Pool-backed."""
        if self.pool_view is None or not self.token_ids:
            return None
        n = self.num_tokens_stored
        return [self.pool_view.k_pools[l][self.block_id, :, :n, :]
                for l in range(self.pool_view.num_layers)]

    @property
    def v_tensor(self) -> list[torch.Tensor] | None:
        if self.pool_view is None or not self.token_ids:
            return None
        n = self.num_tokens_stored
        return [self.pool_view.v_pools[l][self.block_id, :, :n, :]
                for l in range(self.pool_view.num_layers)]

    def acquire(self):
        """Increment ref count — a request is using this block."""
        self.ref_count += 1

    def release(self):
        """Decrement ref count — a request stopped using this block."""
        self.ref_count = max(0, self.ref_count - 1)

    def clear(self):
        """Reset the block to empty state. Pool slot is left as-is (zeroed lazily on next store)."""
        self.token_ids = []
        self.ref_count = 0
        self.last_accessed = 0.0
        self.is_first_block = False


class BlockPoolView:
    """Tiny container giving a Block back-references to the pools it indexes into."""
    __slots__ = ("k_pools", "v_pools", "num_layers")

    def __init__(self, k_pools: list[torch.Tensor], v_pools: list[torch.Tensor]):
        self.k_pools = k_pools
        self.v_pools = v_pools
        self.num_layers = len(k_pools)
