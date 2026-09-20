"""Shared instrument plumbing: build a panel, emit it, keep provenance honest.

Instruments (scripts/) know engine internals; this module knows the contract. Every instrument
prints its human table AND writes a machine record, so a human and the loop read the same run.

The run_group is the mechanism that makes cross-session comparison impossible rather than merely
discouraged: arms measured in one process share it, and compare.py refuses across groups.
"""

from __future__ import annotations

import json
import math
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


# One trial per process: the panels of a single rate sweep. The run_group spans every arm and
# replicate of an experiment, so it is too coarse to say "these points came from one sweep" —
# and the sweep is the unit that has a saturation knee.
_TRIAL = f"t-{uuid.uuid4().hex[:8]}"


def run_group() -> str:
    return _RUN_GROUP


def trial_id() -> str:
    return _TRIAL


def rank_index(q: float, n: int) -> int:
    """0-based NEAREST-RANK index of the q-th percentile of n items: `ceil(q * n) - 1`.

    The textbook definition, and deliberately not the `floor(q * (n - 1))` this module used
    until panel version 2. Two reasons it has to be this one:

    * At n=8 it makes p95 the maximum rather than the second largest. A tail metric that a
      merge is gated on should err toward the slow end, and 8 is the size of the cold_start
      reference configs, so this is the regime actually run rather than a limit case.
    * It is the convention `session.primary_metric` already used. Its failure rule is
      `n_ok < ceil(0.95 * n)`, which is exactly the condition under which this index falls
      among the requests that never answered — so the two agreed on paper and disagreed in
      code. `kb-20260918-4f4c85b7` had already named the floor variant as the bug ("an earlier
      floor-interpolation version let 1 failure in 8 through"); `pct_over_attempts` reintroduced
      it, and `tests/test_panel_failure_accounting.py` now pins the two rules together.
    """
    return min(max(math.ceil(q * n) - 1, 0), n - 1)


def pct(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile over the values given. See `rank_index` for the convention."""
    xs = sorted(values)
    if not xs:
        return 0.0
    return float(xs[rank_index(q, len(xs))])


def pct_over_attempts(values: Iterable[float], q: float, n_attempted: int) -> float:
    """`pct`, but over ATTEMPTS: the requests that never answered rank worst.

    A shed, expired or timed-out request is a first token that never arrived, so it sorts above
    every measured one rather than dropping out of the sample. `inf` when the percentile lands
    among them: the tail is genuinely worse than anything this run measured, and quoting the
    survivors' number there is what let a config improve its p95 by refusing its slowest
    requests. Identical to `pct` when nothing failed — the panel has ONE percentile convention,
    which is the point; two of them is how the simulator and the hardware ended up ranking
    `ttft_p95` against each other under different definitions for nine configs.
    """
    xs = sorted(values)
    n = max(int(n_attempted), len(xs))
    if not n:
        return 0.0
    idx = rank_index(q, n)
    return float(xs[idx]) if idx < len(xs) else math.inf


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


def device_state_from_env() -> tuple[dict[str, Any] | None, bool | None]:
    """The machine record the venue stamped into `RESEARCH_DEVICE_STATE`, or (None, None).

    A laptop or CPU run has no such env and is unaffected — the fields stay None and
    compare.py treats "unknown" as "not a reason to refuse".
    """
    raw = os.environ.get("RESEARCH_DEVICE_STATE")
    if not raw:
        return None, None
    try:
        state = json.loads(raw)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(state, dict):
        return None, None
    locked = state.get("clocks_locked")
    return state, (locked if isinstance(locked, bool) else None)


def build_validity(
    harness: str,
    harness_config: dict[str, Any],
    *,
    n_samples: int,
    workload_regime: str,
    stderr_value: float | None = None,
    concurrency_observed: int | None = None,
    notes: str = "",
    chat_template: dict[str, Any] | None = None,
) -> Validity:
    device_state, clocks_locked = device_state_from_env()
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
        device_state=device_state,
        clocks_locked=clocks_locked,
        chat_template=chat_template,
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
        active_high_water=s.get("active_high_water"),
        active_mean=s.get("active_mean"),
        decode_steps=s.get("decode_steps"),
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
