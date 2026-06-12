"""Window-aware per-pool KV admission (CustomTorchBackend.kv_reserve/kv_release).

Sliding pools cap a request's footprint at the window, so a small sliding pool still admits
long requests; the full pool (grows with sequence) is the binding constraint. No model needed.
"""

from __future__ import annotations

import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.paged_kv_cache import BlockPool


def _backend():
    b = CustomTorchBackend(device="cpu")
    bs = 16
    full = BlockPool(10, bs, 1, 8, torch.float32, torch.device("cpu"), window=None)      # binding
    sliding = BlockPool(100, bs, 1, 8, torch.float32, torch.device("cpu"), window=32)    # window=32 → 3 blocks
    b.pools = [full, sliding, None]   # None = a KV-shared layer
    b._reserved = [0, 0, 0]
    return b


def test_sliding_footprint_capped_at_window():
    b = _backend()
    # 1000-token request: full = ceil(1000/16)=63 blocks; sliding capped at 32//16+1 = 3.
    assert b._kv_footprints(900, 100) == [63, 3, 0]


def test_admission_bound_by_full_pool_not_sliding():
    b = _backend()
    # Long request: sliding fits easily (3 ≤ 100) but full (63) exceeds the 10-block full pool.
    assert b.kv_reserve(900, 100) is False
    assert b._reserved == [0, 0, 0], "rejected reservation must not mutate state"

    # Short requests fit the full pool; admit two until the full pool binds.
    assert b.kv_reserve(80, 20) is True      # full 7, sliding 3
    assert b._reserved == [7, 3, 0]
    assert b.kv_reserve(80, 20) is False     # full 7+7=14 > 10 → full pool binds
    assert b._reserved == [7, 3, 0]

    b.kv_release(80, 20)
    assert b._reserved == [0, 0, 0]


def test_small_sliding_pool_supports_long_requests():
    """A tiny sliding pool (sized to ~window) still admits a long request — the win: sliding
    pools can be small, freeing memory for the binding full pools."""
    b = CustomTorchBackend(device="cpu")
    bs = 16
    full = BlockPool(200, bs, 1, 8, torch.float32, torch.device("cpu"), window=None)
    sliding = BlockPool(4, bs, 1, 8, torch.float32, torch.device("cpu"), window=32)  # only 4 blocks
    b.pools = [full, sliding]
    b._reserved = [0, 0]
    assert b.kv_reserve(2000, 100) is True   # 2100 tokens: full=132≤200, sliding=min(132,3)=3≤4
    assert b._reserved == [132, 3]
