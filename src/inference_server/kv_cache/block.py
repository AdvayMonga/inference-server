"""KV cache block — fixed-size unit of cached attention state."""

from dataclasses import dataclass, field

import torch


@dataclass
class Block:
    """A fixed-size block holding K and V tensors for a range of tokens."""

    block_id: int
    block_size: int  # number of token positions in this block
    k_tensor: torch.Tensor | None = None  # [num_layers, num_heads, block_size, head_dim]
    v_tensor: torch.Tensor | None = None
    token_ids: list[int] = field(default_factory=list)  # which tokens are stored here
    ref_count: int = 0  # number of active requests using this block
    last_accessed: float = 0.0  # timestamp for LRU eviction

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0 and len(self.token_ids) == 0

    @property
    def num_tokens_stored(self) -> int:
        return len(self.token_ids)

    def acquire(self):
        """Increment ref count — a request is using this block."""
        self.ref_count += 1

    def release(self):
        """Decrement ref count — a request stopped using this block."""
        self.ref_count = max(0, self.ref_count - 1)

    def clear(self):
        """Reset the block to empty state."""
        self.k_tensor = None
        self.v_tensor = None
        self.token_ids = []
        self.ref_count = 0
        self.last_accessed = 0.0
