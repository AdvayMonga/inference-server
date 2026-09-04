"""Shared instrument plumbing: build a panel, emit it, keep provenance honest.

Instruments (scripts/) know engine internals; this module knows the contract. Every instrument
prints its human table AND writes a machine record, so a human and the loop read the same run.

The run_group is the mechanism that makes cross-session comparison impossible rather than merely
discouraged: arms measured in one process share it, and compare.py refuses across groups.
"""

from __future__ import annotations

import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from inference_server.research.schemas import (
    PANEL_VERSION,
    REPO_ROOT,
    Validity,
    Vitals,
    git_dirty,
    git_sha,
)

RUNS_DIR = REPO_ROOT / "runs"

# One group per process. Arms measured together are comparable; anything else is not.
_RUN_GROUP = os.environ.get("RESEARCH_RUN_GROUP") or f"grp-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"


def run_group() -> str:
    return _RUN_GROUP


def pct(values: Iterable[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    return float(xs[min(int(q * (len(xs) - 1)), len(xs) - 1)])


def stderr(values: Iterable[float]) -> float | None:
    xs = list(values)
    if len(xs) < 2:
        return None
    return statistics.stdev(xs) / (len(xs) ** 0.5)


def infer_regime(hit_rate: float | None, pool_size: int | None = None) -> str:
    """Name the workload regime rather than leaving it implicit.

    POOL_SIZE=64 against a warm cache is a cache-HIT benchmark; quoting a prefill number from it
    without saying so is how this project mis-attributed prefill work for a while.
    """
    if hit_rate is None:
        return "synthetic"
    if hit_rate >= 0.5:
        return "cache_hit_heavy"
    if hit_rate <= 0.1:
        return "cache_miss_heavy"
    return "mixed"


def build_validity(
    harness: str,
    harness_config: dict[str, Any],
    *,
    n_samples: int,
    workload_regime: str,
    stderr_value: float | None = None,
    concurrency_observed: int | None = None,
    notes: str = "",
) -> Validity:
    return Validity(
        engine_sha=git_sha(),
        dirty=git_dirty(),
        harness=harness,
        harness_config=harness_config,
        workload_regime=workload_regime,
        n_samples=n_samples,
        run_group=run_group(),
        stderr=stderr_value,
        concurrency_observed=concurrency_observed,
        notes=notes,
    )


def panel_from_stats(
    validity: Validity,
    *,
    scheduler_stats: dict[str, Any] | None = None,
    cache_stats: dict[str, Any] | None = None,
    **fields: Any,
) -> Vitals:
    """Assemble a panel from the engine's own stats dicts plus measured fields.

    Reading the engine's stats verbatim (rather than recomputing) keeps the panel and the live
    /scheduler/stats and /cache/stats endpoints from drifting apart.
    """
    s = scheduler_stats or {}
    c = cache_stats or {}
    return Vitals(
        validity=validity,
        wave_sizes={str(k): v for k, v in (s.get("wave_sizes") or {}).items()},
        active_size=s.get("active_size"),
        pending_depth=s.get("pending_depth"),
        pending_high_water=s.get("pending_high_water"),
        kv_admit_blocked=s.get("kv_admit_blocked"),
        total_rejected=s.get("total_rejected"),
        total_expired=s.get("total_expired"),
        total_preempted=s.get("total_preempted"),
        total_iteration_errors=s.get("total_iteration_errors"),
        cache_hit_rate=c.get("hit_rate"),
        cache_lookups=c.get("lookups"),
        cache_entries=c.get("entries"),
        cache_evictions=c.get("evictions"),
        cache_blocks_held=c.get("blocks_held"),
        cache_max_blocks=c.get("max_blocks"),
        pool_free_blocks=c.get("pool_free_blocks"),
        pool_total_blocks=c.get("pool_total_blocks"),
        pool_utilization=c.get("pool_utilization"),
        panel_version=PANEL_VERSION,
        **fields,
    )


def emit(panel: Vitals, *, label: str = "", runs_dir: Path | None = None) -> Path:
    """Validate, write `runs/<run_id>.json`, and print where it went.

    Validation is deliberately fatal: a panel missing its provenance is worse than no panel,
    because it looks usable.
    """
    panel.validate()
    target = (runs_dir or RUNS_DIR) / f"{panel.validity.run_id}.json"
    panel.to_json(target)
    rel = target.relative_to(REPO_ROOT) if target.is_relative_to(REPO_ROOT) else target
    print(f"[panel] {label or panel.validity.harness} -> {rel} "
          f"(regime={panel.validity.workload_regime}, n={panel.validity.n_samples}, "
          f"group={panel.validity.run_group})", flush=True)
    return target
