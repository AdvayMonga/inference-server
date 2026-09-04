"""Comparing panels — and, more importantly, refusing to.

The costliest errors in this project were not bad optimisations, they were valid-looking
comparisons between things that were not comparable:

  * a closed-loop harness measured the wrong bottleneck for ~a month
  * POOL_SIZE=64 quietly made the SLO sweep a cache-HIT benchmark, hiding every miss-path bug
  * MAX_BATCH_SIZE 256 vs 32 across runs read as a 2.4x TPOT regression
  * 957 vs 1151 tok/s for identical config, measured in different sessions

So the API makes the unsafe thing hard: `comparable()` must pass before any delta is computed,
and it refuses by default rather than warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from inference_server.research.schemas import Vitals

# Knobs that change the answer. Two panels differing on any of these measure different things,
# regardless of how similar their numbers look.
CONFIG_KEYS_THAT_MATTER = (
    "model", "gpu", "max_batch_size", "prefill_mode", "compile", "prefill_graph",
    "blocks", "sliding_blocks", "context_window", "rates", "duration", "pool_size",
    "max_queue_wait_s", "prefix_cache_impl", "wave_window_mult",
)


@dataclass
class Comparability:
    ok: bool
    reasons: list[str]

    def __bool__(self) -> bool:
        return self.ok


def comparable(a: Vitals, b: Vitals, *, same_group_required: bool = True) -> Comparability:
    """Can these two panels be compared at all?

    `same_group_required` defaults True: arms of an experiment must be run in the same session.
    Set it False only for deliberate historical inspection, never to judge an experiment.
    """
    reasons: list[str] = []

    if a.panel_version != b.panel_version:
        reasons.append(f"panel_version {a.panel_version} vs {b.panel_version}: the panels "
                       f"measure different things")

    if a.validity.workload_regime != b.validity.workload_regime:
        reasons.append(f"workload_regime {a.validity.workload_regime} vs "
                       f"{b.validity.workload_regime}: e.g. a cache-hit run flatters prefill "
                       f"work that a cache-miss run does not")

    if a.validity.harness != b.validity.harness:
        reasons.append(f"different harness ({a.validity.harness} vs {b.validity.harness})")

    for key in CONFIG_KEYS_THAT_MATTER:
        av, bv = a.validity.harness_config.get(key), b.validity.harness_config.get(key)
        if av != bv:
            reasons.append(f"harness_config[{key}]: {av!r} vs {bv!r}")

    if same_group_required and a.validity.run_group != b.validity.run_group:
        reasons.append(
            f"different run_group ({a.validity.run_group} vs {b.validity.run_group}): arms must "
            f"be measured in the same session. This repo has seen 957 vs 1151 tok/s for "
            f"identical config across sessions")

    return Comparability(ok=not reasons, reasons=reasons)


@dataclass
class Significance:
    # significant | noise | insufficient_samples | not_comparable | non_numeric_metric
    verdict: str
    metric: str
    before: float | None
    after: float | None
    delta: float | None
    pct: float | None
    detail: str

    @property
    def improved(self) -> bool:
        return self.verdict == "significant" and (self.delta or 0) != 0

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "metric": self.metric, "before": self.before,
                "after": self.after, "delta": self.delta, "pct": self.pct, "detail": self.detail}


def _welch(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    """Welch's t statistic. Unequal variances assumed — arms genuinely differ in spread."""
    se = math.sqrt((s1 ** 2) / max(n1, 1) + (s2 ** 2) / max(n2, 1))
    if se == 0:
        return math.inf if m1 != m2 else 0.0
    return (m2 - m1) / se


def significance(
    baseline: Vitals,
    treatment: Vitals,
    metric: str,
    *,
    direction: str,
    min_samples: int = 3,
    t_threshold: float = 2.0,
) -> Significance:
    """Judge one metric. Returns a typed verdict — never a bare number.

    An effect that does not clear the variance budget is recorded as `noise`, which is a real
    outcome and gets written to the knowledge base like any other.
    """
    cmp_ = comparable(baseline, treatment)
    if not cmp_:
        return Significance("not_comparable", metric, None, None, None, None,
                            "; ".join(cmp_.reasons))

    before = getattr(baseline, metric, None)
    after = getattr(treatment, metric, None)
    if before is None or after is None:
        return Significance("insufficient_samples", metric, before, after, None, None,
                            f"{metric} missing from one arm")

    # Some panel fields are structural, not scalar (wave_sizes, *_by_bucket). They are evidence
    # for the VALIDITY gate, not something a t-test can judge — say so instead of crashing.
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return Significance(
            "non_numeric_metric", metric, None, None, None, None,
            f"{metric} is structural ({type(before).__name__}); it can support a validity "
            f"check but cannot be tested for significance — pick a scalar metric")

    delta = after - before
    pct = (delta / before * 100.0) if before else None

    n1, n2 = baseline.validity.n_samples, treatment.validity.n_samples
    s1, s2 = baseline.validity.stderr, treatment.validity.stderr
    if n1 < min_samples or n2 < min_samples or s1 is None or s2 is None:
        return Significance(
            "insufficient_samples", metric, before, after, delta, pct,
            f"need n>={min_samples} and stderr on both arms (got n={n1}/{n2}, "
            f"stderr={s1}/{s2}); a single measurement cannot clear this repo's variance")

    t = _welch(before, s1 * math.sqrt(n1), n1, after, s2 * math.sqrt(n2), n2)
    if abs(t) < t_threshold:
        return Significance("noise", metric, before, after, delta, pct,
                            f"|t|={abs(t):.2f} < {t_threshold}: within the variance budget")

    moved_right_way = (direction == "increase" and delta > 0) or \
                      (direction == "decrease" and delta < 0)
    return Significance(
        "significant", metric, before, after, delta, pct,
        f"|t|={abs(t):.2f} >= {t_threshold}, moved "
        f"{'as predicted' if moved_right_way else 'AGAINST the prediction'}")
