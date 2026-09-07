"""Batch occupancy — how full the batch actually was, per decode step.

The number that decides throughput on a batching engine: a decode step streams the whole weight
matrix from HBM whether it serves one row or two hundred, so tokens/sec is set by how many rows
share that read. It was unmeasurable until now, and its absence produced a wrong conclusion:
49% K=1 prefill waves under overload was read as a starved batch, when a FULL batch produces
exactly that (each completion frees one slot, so the next wave is K=1 by construction).

`active_size` could not settle it — an instantaneous len() read when stats() is called, which for
a benchmark is after the run has drained, so it reports 0. These tests pin the replacement.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from tests.test_chunked_prefill import FakeChunkBackend


async def _run(n_requests: int, max_batch_size: int, max_tokens: int = 6, prompt_len: int = 8):
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=max_batch_size,
                                     prefill_mode="batched")
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(range(prompt_len)), max_tokens=max_tokens,
                                 session_id=f"s{i}", future=loop.create_future())
                for i in range(n_requests)]
        await asyncio.gather(*(sched.submit(r) for r in reqs))
        return sched.stats()
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_occupancy_is_reported_after_the_run_drains():
    """The defect being fixed: active_size reads 0 once the queue empties, so every benchmark
    panel recorded 0 concurrency no matter how loaded the server had been."""
    stats = await _run(n_requests=8, max_batch_size=8)

    assert stats["active_size"] == 0, "instantaneous read still drains to 0 — that is expected"
    assert stats["decode_steps"] > 0
    assert stats["active_high_water"] >= 1
    assert stats["active_mean"] > 0


@pytest.mark.asyncio
async def test_occupancy_rises_with_offered_load():
    """A metric that does not move with load cannot localise a bottleneck. pool_utilization
    reading a constant 0.5002 across a 16x load range is the failure this guards against."""
    light = await _run(n_requests=2, max_batch_size=16)
    heavy = await _run(n_requests=16, max_batch_size=16)

    assert heavy["active_high_water"] > light["active_high_water"]
    assert heavy["active_mean"] > light["active_mean"]


@pytest.mark.asyncio
async def test_occupancy_never_exceeds_the_batch_limit():
    """A high-water above max_batch_size would mean the admission gate leaked."""
    stats = await _run(n_requests=20, max_batch_size=4)

    assert stats["active_high_water"] <= 4
    assert stats["active_mean"] <= 4.0


@pytest.mark.asyncio
async def test_mean_is_per_decode_step_not_per_request():
    """Mean occupancy must be an average over steps. Averaging over requests would report the
    batch as full whenever ANY step was full, which is the reading that misled us."""
    stats = await _run(n_requests=6, max_batch_size=6, max_tokens=10)

    # 10 tokens per request = 1 from prefill + 9 decode steps. Steps must accumulate across
    # the run, not reset per request.
    assert stats["decode_steps"] >= 9, "decode steps must accumulate across the run"
    assert stats["active_mean"] <= stats["active_high_water"]


@pytest.mark.asyncio
async def test_no_decode_steps_reports_zero_not_a_crash():
    """An idle scheduler must not divide by zero when something scrapes /scheduler/stats."""
    sched = ContinuousBatchScheduler(FakeChunkBackend(), max_batch_size=4, prefill_mode="batched")
    sched.start()
    try:
        stats = sched.stats()
        assert stats["active_mean"] == 0.0
        assert stats["decode_steps"] == 0
        assert stats["active_high_water"] == 0
    finally:
        await sched.stop()


def test_panel_carries_occupancy_and_concurrency_is_not_queue_depth():
    """The panel's `concurrency_observed` used to be fed pending_high_water — queue depth — while
    LOOP.md lists that field as catching 'claiming a concurrency the run never reached'. It was
    reporting the opposite of what it claimed."""
    from inference_server.research import harness as H

    sched_stats = {"active_high_water": 42, "active_mean": 31.5, "decode_steps": 900,
                   "pending_high_water": 2985}
    v = H.build_validity("bench_serving", {"rates": "128"}, n_samples=5,
                         workload_regime="cache_miss_heavy",
                         concurrency_observed=sched_stats["active_high_water"])
    panel = H.panel_from_stats(v, scheduler_stats=sched_stats)

    assert panel.active_high_water == 42
    assert panel.active_mean == 31.5
    assert panel.decode_steps == 900
    assert panel.validity.concurrency_observed == 42, "must be occupancy, not the 2985 queued"
    panel.validate()


def test_old_panels_without_occupancy_still_load():
    """Adding an optional field must not strand 42 existing panels: PANEL_VERSION did not move,
    because a field that was not recorded cannot change a measurement that was already taken."""
    from inference_server.research.schemas import PANEL_VERSION, Validity, Vitals

    v = Validity(engine_sha="abc", dirty=False, harness="bench_serving",
                 harness_config={"rates": "2"}, workload_regime="cache_miss_heavy",
                 n_samples=5, run_group="old")
    old = Vitals(validity=v, ttft_p95=100.0)

    assert old.panel_version == PANEL_VERSION
    assert old.active_high_water is None and old.active_mean is None
    old.validate()
