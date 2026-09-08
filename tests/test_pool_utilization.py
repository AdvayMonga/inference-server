"""pool_utilization must describe one pool, not two.

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


class _StubPool:
    """Only the two attributes stats() reads."""

    def __init__(self, num_blocks: int, free_count: int):
        self.num_blocks = num_blocks
        self.free_count = free_count


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
def test_utilization_never_mixes_pools(impl):
    """The production layout: full-attention pools of 8192, sliding pools of 4096.

    Here the sliding pool is 75% consumed and the full pool only 50%, so the sliding pool binds.
    The old code paired free=1024 (from the sliding pool) with total=4096 (also the sliding
    pool's, but only by coincidence of both being the min) — and as soon as the tightest pool is
    not also the smallest, the two halves come from different pools entirely.
    """
    pools = [_StubPool(8192, 4096), _StubPool(4096, 1024)]
    stats = _caches(pools)[impl].stats()

    assert stats["pool_utilization"] == 0.75
    assert stats["pool_free_blocks"] == 1024, "free must come from the pool total describes"
    assert stats["pool_total_blocks"] == 4096, "and total from that same pool"


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_utilization_reports_the_binding_pool(impl):
    """Admission blocks when ANY pool cannot fit, so the tightest pool is the whole story.
    A mean across pools would hide an exhausted one behind three empty ones."""
    pools = [_StubPool(8192, 8000), _StubPool(8192, 100), _StubPool(4096, 4000)]
    stats = _caches(pools)[impl].stats()

    assert stats["pool_utilization"] == pytest.approx(1 - 100 / 8192, abs=1e-4)
    assert stats["pool_free_blocks"] == 100


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_utilization_moves_with_pressure(impl):
    """The property whose absence raised the alarm: it must respond to load."""
    idle = _caches([_StubPool(4096, 4096)])[impl].stats()["pool_utilization"]
    half = _caches([_StubPool(4096, 2048)])[impl].stats()["pool_utilization"]
    full = _caches([_StubPool(4096, 0)])[impl].stats()["pool_utilization"]

    assert idle == 0.0 and full == 1.0
    assert idle < half < full


@pytest.mark.parametrize("impl", ["plain", "radix"])
def test_utilization_stays_in_range(impl):
    """Mixing pools could produce a ratio above 1.0. A utilization cannot exceed 1."""
    pools = [_StubPool(16384, 16000), _StubPool(256, 4)]
    u = _caches(pools)[impl].stats()["pool_utilization"]

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

    from inference_server.research.schemas import REPO_ROOT

    keys = ("pool_utilization", "pool_free_blocks", "pool_total_blocks")
    allowed = {"models/paged_kv_cache.py",     # where they are produced
               "server.py"}                    # /cache/stats, Prometheus gauges
    offenders = []
    for py in (Path(REPO_ROOT) / "src" / "inference_server").rglob("*.py"):
        rel = py.relative_to(Path(REPO_ROOT) / "src" / "inference_server").as_posix()
        if rel in allowed or rel.startswith("research/"):
            continue
        text = py.read_text()
        if any(k in text for k in keys):
            offenders.append(rel)

    assert not offenders, (
        f"{offenders} now read a reported pool stat. If engine behaviour depends on it, these "
        f"numbers are no longer reporting-only and changes to them need an experiment.")
