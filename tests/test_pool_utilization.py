"""pool_utilization must describe one pool, not two (since 2026-10-20 every layer's pool shares one allocator, so
there is exactly one to describe).

Symptom that exposed this: across a sweep of 8 -> 128 req/s (a 16x load range, with
kv_admit_blocked climbing 0 -> 1176 over the same span) pool_utilization read exactly 0.5002 in
every panel. A KV pool genuinely under rising pressure does not hold four decimal places steady.

Cause: `free` and `total` were each computed with an independent `min()` across pools, so the
numerator could come from one pool and the denominator from another. With a mixed layout —
CUSTOM_BACKEND_BLOCKS=8192 for full-attention layers, SLIDING_BLOCKS=4096 for windowed ones —
`total` is pinned to the small pool while `free` tracks whichever pool happens to be tightest.
The ratio then describes no pool that exists, and can even exceed 1.0.

This matters beyond the number being wrong: attribute.py uses pool_utilization to decide whether
a run is KV-pressure-bound, so a stuck value can misdirect an entire iteration.
"""

from __future__ import annotations

import pytest

from inference_server.models.paged_kv_cache import PrefixCache, RadixPrefixCache


class _StubAlloc:
    """Only the two attributes stats() reads of the pools' shared allocator."""

    def __init__(self, num_blocks: int, free_count: int):
        self.num_blocks = num_blocks
        self.free_count = free_count


class _StubPool:
    def __init__(self, alloc: _StubAlloc):
        self.allocator = alloc
        self.num_blocks = alloc.num_blocks


def _pools(num_blocks: int, free_count: int, n: int = 2):
    """`n` pools over one allocator, as make_pools builds them (every layer fills at the same rate)."""
    alloc = _StubAlloc(num_blocks, free_count)
    return [_StubPool(alloc) for _ in range(n)]


def _caches(pools):
    """Both implementations share the defect, so both are held to the fix."""
    plain = PrefixCache.__new__(PrefixCache)
    plain.pools = pools
    plain.entries, plain.max_entries = {}, 1024
    plain.lookups = plain.hits = plain.evictions = 0
    plain._blocks_held, plain.max_blocks = 0, 100

    radix = RadixPrefixCache.__new__(RadixPrefixCache)
    radix.pools = pools
    radix._live = [i for i, p in enumerate(pools) if p is not None]
    radix._nodes = radix.lookups = radix.hits = radix.evictions = 0
    radix.max_entries = 1024
    radix.max_blocks = 100          # _capacity() reads it
    return {"plain": plain, "radix": radix}


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_utilization_describes_the_one_allocator(impl):
    """Every layer's pool draws its block ids from one shared allocator (a request's block id is the same in every
    layer), so free and total come from that one object: the ratio describes the pool that exists."""
    stats = _caches(_pools(4096, 1024, n=48))[impl].stats()

    assert stats["pool_utilization"] == 0.75
    assert stats["pool_free_blocks"] == 1024
    assert stats["pool_total_blocks"] == 4096


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_utilization_moves_with_pressure(impl):
    """The property whose absence raised the alarm: it must respond to load."""
    idle = _caches(_pools(4096, 4096))[impl].stats()["pool_utilization"]
    half = _caches(_pools(4096, 2048))[impl].stats()["pool_utilization"]
    full = _caches(_pools(4096, 0))[impl].stats()["pool_utilization"]

    assert idle == 0.0 and full == 1.0
    assert idle < half < full


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_utilization_stays_in_range(impl):
    u = _caches(_pools(256, 4))[impl].stats()["pool_utilization"]

    assert 0.0 <= u <= 1.0


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_no_pools_does_not_divide_by_zero(impl):
    stats = _caches([None, None])[impl].stats()

    assert stats["pool_utilization"] == 0.0
    assert stats["pool_total_blocks"] == 0


def test_pool_stats_are_reporting_only():
    """The claim that licensed fixing this without a GPU A/B: correcting these numbers cannot
    change what the engine DOES, because nothing in the engine's control flow reads them.

    Enforced rather than grepped once. Admission deliberately reads different properties —
    cache.free_blocks / cache.total_blocks / cache.pressure on the adapter — so if anyone later
    makes a scheduling decision depend on a reported stat, this fails and the exemption that
    allowed a CPU-only merge no longer holds.
    """
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]

    keys = ("pool_utilization", "pool_free_blocks", "pool_total_blocks")
    allowed = {"models/paged_kv_cache.py",     # where they are produced
               "server.py"}                    # /cache/stats, Prometheus gauges
    offenders = []
    for py in (Path(REPO_ROOT) / "src" / "inference_server").rglob("*.py"):
        rel = py.relative_to(Path(REPO_ROOT) / "src" / "inference_server").as_posix()
        if rel in allowed:
            continue
        text = py.read_text()
        if any(k in text for k in keys):
            offenders.append(rel)

    assert not offenders, (
        f"{offenders} now read a reported pool stat. If engine behaviour depends on it, these "
        f"numbers are no longer reporting-only and changes to them need an experiment.")
