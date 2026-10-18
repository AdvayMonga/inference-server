"""Per-pool KV admission (CustomTorchBackend.kv_reserve/kv_release/kv_capacity_error).

A request's footprint is prompt + max_tokens in blocks, in every pool that owns K/V; the
smallest pool is the binding constraint. No model needed.
"""

from __future__ import annotations

import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.paged_kv_cache import BlockPool


def _backend():
    b = CustomTorchBackend(device="cpu")
    bs = 16
    small = BlockPool(10, bs, 1, 8, torch.float32, torch.device("cpu"))      # binding
    large = BlockPool(100, bs, 1, 8, torch.float32, torch.device("cpu"))
    b.pools = [small, large, None]   # None = a layer that owns no K/V
    b._reserved = [0, 0, 0]
    return b


def test_footprint_is_the_same_in_every_pool():
    b = _backend()
    # 1000-token request: ceil(1000/16)=63 blocks in each pool that owns K/V, none for a None layer.
    assert b._kv_footprints(900, 100) == [63, 63, 0]


def test_admission_bound_by_the_smallest_pool():
    b = _backend()
    # Long request: fits the large pool (63 ≤ 100) but not the 10-block small pool.
    assert b.kv_reserve(900, 100) is False
    assert b._reserved == [0, 0, 0], "rejected reservation must not mutate state"

    # Short requests fit; admit one until the small pool binds.
    assert b.kv_reserve(80, 20) is True      # 7 blocks per pool
    assert b._reserved == [7, 7, 0]
    assert b.kv_reserve(80, 20) is False     # 7+7=14 > 10 → the small pool binds
    assert b._reserved == [7, 7, 0]

    b.kv_release(80, 20)
    assert b._reserved == [0, 0, 0]


def test_capacity_error_only_when_a_pool_can_never_fit():
    b = _backend()
    assert b.kv_capacity_error(80, 20) is None
    b._reserved = [10, 0, 0]                  # a pool that is merely busy is not an error
    assert b.kv_capacity_error(80, 20) is None
    assert b.kv_capacity_error(900, 100) == (
        "request needs 1000 KV tokens (prompt + max_tokens) but the cache holds 160")
