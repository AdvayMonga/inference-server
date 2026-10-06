"""auto_num_blocks: KV pool sizing from free device memory (pure arithmetic, fake numbers)."""

import pytest

from inference_server.models.paged_kv_cache import auto_num_blocks

GiB = 2**30
QWEN = [(4, 128, False)] * 48   # Qwen3-30B-A3B: 48 full layers, 4 KV heads, head_dim 128


def test_qwen_h200_budget():
    # 96 KiB/token bf16 → 1.5 MiB per 16-token block; (0.9 * 84 GiB - 4 GiB) / 1.5 MiB
    n = auto_num_blocks(84 * GiB, 0.9, 4 * GiB, QWEN, block_size=16, elem_size=2)
    assert n == int(0.9 * 84 * GiB - 4 * GiB) // (3 * 2**19)
    assert n * 16 > 400_000   # ~0.4M tokens, vs 8k at the old 512-block default


def test_sliding_pools_share_the_budget_or_are_pinned():
    layers = [(1, 256, True)] * 4 + [(1, 512, False)]
    per = 2 * 16 * 2   # bytes per block per (head * head_dim) unit
    shared = auto_num_blocks(GiB, 1.0, 0, layers, 16, 2)
    assert shared == GiB // (per * (4 * 256 + 512))
    pinned = auto_num_blocks(GiB, 1.0, 0, layers, 16, 2, sliding_blocks=100)
    assert pinned == (GiB - 100 * per * 4 * 256) // (per * 512)
    assert pinned > shared


def test_capped_at_int32_pool_offsets():
    n = auto_num_blocks(10**6 * GiB, 1.0, 0, QWEN, 16, 2)
    assert n == (2**31 - 1) // (4 * 16 * 128)


def test_no_room_raises():
    with pytest.raises(RuntimeError, match="no device memory"):
        auto_num_blocks(4 * GiB, 0.9, 4 * GiB, QWEN, 16, 2)
