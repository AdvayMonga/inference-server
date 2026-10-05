"""Host-memory KV tier: evicted radix blocks come back from host and give identical greedy output."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from inference_server import config
from inference_server.models.paged_kv_cache import BlockPool, RadixPrefixCache

# Odd lengths so no prompt is a full-block hit (see test_qwen3_moe_parity.PROMPTS).
A = [5, 17, 300, 42, 8, 99, 123, 7, 55]
FILLERS = [[200 + i, 201 + i, 202 + i, 203 + i, 204 + i, 205 + i, 206 + i] for i in range(0, 40, 8)]


def _backend(tiny_dir, host_bytes):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUSTOM_BACKEND_BLOCKS", "64")
        mp.setenv("CUSTOM_BACKEND_BLOCK_SIZE", "2")
        # Trie capped at 6 blocks, so a few fillers evict A's prefix from the device.
        mp.setattr(config, "settings", dataclasses.replace(
            config.settings, prefix_cache_impl="radix", prefix_cache_block_fraction=0.1,
            kv_host_tier_bytes=host_bytes))
        b = CustomTorchBackend(device="cpu", model_name=tiny_dir)
        b.load_model(tiny_dir)
    return b


def _run(b):
    """Turn 1, evict it, then the same prompt and a multi-turn extension of it."""
    outs = [b.generate(A, 6)]
    for f in FILLERS:
        b.generate(f, 2)
    outs.append(b.generate(A, 6))
    hits = b.last_cache_hit_tokens
    outs.append(b.generate(A + outs[0] + [77, 31], 6))
    return outs, hits


def test_host_tier_restores_evicted_prefix_with_identical_output(tiny_qwen3_moe_dir):
    off, off_hits = _run(_backend(tiny_qwen3_moe_dir, 0))
    on_b = _backend(tiny_qwen3_moe_dir, 1 << 20)
    free0 = [p.free_count for p in on_b.pools]
    on, on_hits = _run(on_b)
    assert on == off
    assert off_hits == 0 and on_hits == 8          # device missed; host gave back all 4 full blocks
    st = on_b.prefix_cache.stats()
    assert st["host"]["hits"] >= 1 and st["host"]["blocks_restored"] >= 4 and st["host"]["bytes"] > 0
    held = st["blocks_held"]                        # every non-cached block went back to the pool
    assert [free0[i] - p.free_count for i, p in enumerate(on_b.pools)] == [held] * len(on_b.pools)


def test_host_tier_off_by_default(tiny_qwen3_moe_dir):
    b = _backend(tiny_qwen3_moe_dir, 0)
    assert b.prefix_cache.host is None and "host" not in b.prefix_cache.stats()


def _pools(n):
    return [BlockPool(n, 2, 1, 4, torch.float32, torch.device("cpu")) for _ in range(2)]


def test_restore_falls_back_when_device_full():
    pools = _pools(4)
    pc = RadixPrefixCache(pools, max_block_fraction=0.5, host_bytes=1 << 20)
    tables = [[p.alloc(), p.alloc()] for p in pools]
    for p, t in zip(pools, tables):
        p.k[t], p.v[t] = torch.randn(2, 1, 2, 4), torch.randn(2, 1, 2, 4)
    want = [(p.k[t].clone(), p.v[t].clone()) for p, t in zip(pools, tables)]
    pc.store([1, 2, 3, 4], tables)
    for p, t in zip(pools, tables):
        for b in t:
            p.release(b)
    pc.reclaim(4)                                   # both nodes (2 layers each) evicted to host
    assert pc.host.stats()["entries"] == 2
    live = [[p.alloc() for _ in range(4)] for p in pools]   # live requests hold every block
    assert pc.lookup([1, 2, 3, 4, 5]) == (0, {})
    assert pc.host.fallbacks == 1 and all(p.free_count == 0 for p in pools)
    for p, t in zip(pools, live):
        for b in t:
            p.release(b)
    matched, shared = pc.lookup([1, 2, 3, 4, 5])
    assert matched == 4 and pc.host.hits == 1
    for (k, v), p, L in zip(want, pools, range(2)):
        assert torch.equal(p.k[shared[L]], k) and torch.equal(p.v[shared[L]], v)
