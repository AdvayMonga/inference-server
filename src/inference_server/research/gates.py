"""The four gates a change must clear before it may reach main. Judged IN ORDER.

Order is the point. Validity comes first because a significant-looking result from a harness
that never exercised the change is worse than no result — it is a confident wrong answer, and
this project shipped one for about a month.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from inference_server.research.compare import Significance, comparable, significance
from inference_server.research.schemas import REPO_ROOT, GateResult, Hypothesis, Vitals

# A change to X is only believable if the run actually exercised X. Maps a hypothesis's target
# metric / tags to the panel evidence that proves the harness reached it.
_EXERCISE_CHECKS: dict[str, Callable[[Vitals], tuple[bool, str]]] = {
    "wave_sizes": lambda v: (
        sum(int(k) * n for k, n in v.wave_sizes.items() if int(k) > 1) > 0,
        "every prefill wave was K=1, so wave planning could not have had any effect "
        "(this is exactly how length-grouped waves looked like a win before measurement)",
    ),
    "cache_hit_rate": lambda v: (
        (v.cache_lookups or 0) > 0,
        "no cache lookups were made; a cache change cannot be judged by this run",
    ),
    "ttft_prefill_p95": lambda v: (
        v.ttft_prefill_p95 is not None,
        "panel has no prefill split; TTFT changes cannot be attributed to prefill vs queueing",
    ),
    "ttft_queue_p95": lambda v: (
        v.ttft_queue_p95 is not None,
        "panel has no queue split; queueing changes cannot be attributed",
    ),
}


def validity_gate(hypothesis: Hypothesis, baseline: Vitals, treatment: Vitals) -> GateResult:
    """Did this harness exercise the thing that changed, and are the arms comparable at all?"""
    cmp_ = comparable(baseline, treatment)
    if not cmp_:
        return GateResult("validity", False,
                          "arms are not comparable: " + "; ".join(cmp_.reasons),
                          {"reasons": cmp_.reasons})

    for panel in (baseline, treatment):
        check = _EXERCISE_CHECKS.get(hypothesis.predicted_metric)
        if check:
            ok, why = check(panel)
            if not ok:
                return GateResult("validity", False,
                                  f"harness did not exercise the change: {why}",
                                  {"run_id": panel.validity.run_id})

    # A cache-hit-heavy run cannot judge a cache change; it flatters prefill work generally.
    if "cache" in hypothesis.gap_id.lower() and \
            baseline.validity.workload_regime == "cache_hit_heavy":
        return GateResult("validity", False,
                          "cache hypothesis judged on a cache_hit_heavy workload; raise "
                          "BENCH_POOL_SIZE or use the stress harness",
                          {"regime": baseline.validity.workload_regime})

    return GateResult("validity", True, "arms comparable and the harness exercised the change",
                      {"regime": baseline.validity.workload_regime,
                       "harness": baseline.validity.harness})


def significance_gate(hypothesis: Hypothesis, baseline: Vitals, treatment: Vitals,
                      **kw: Any) -> GateResult:
    """Does the effect clear the variance budget, on the metric predicted beforehand?"""
    s: Significance = significance(baseline, treatment, hypothesis.predicted_metric,
                                   direction=hypothesis.predicted_direction, **kw)
    if s.verdict != "significant":
        return GateResult("significance", False, f"{s.verdict}: {s.detail}", s.to_dict())
    if "AGAINST the prediction" in s.detail:
        return GateResult("significance", False,
                          f"moved against the prediction ({s.pct:+.1f}%) — a regression, "
                          f"not a win", s.to_dict())
    return GateResult("significance", True,
                      f"{s.pct:+.1f}% on {s.metric} ({s.detail})", s.to_dict())


def correctness_gate(
    baseline: Vitals | None = None,
    treatment: Vitals | None = None,
    *,
    run_tests: bool = True,
    test_args: tuple[str, ...] = ("-q", "-p", "no:cacheprovider"),
) -> GateResult:
    """Fast suite green, and no new engine errors under load.

    Deliberately runs the DEFAULT (fast) selection: the model-heavy tests load a 9.6GB model
    repeatedly and are not safe to run unattended on a 24GB machine, let alone concurrently.
    """
    evidence: dict[str, Any] = {}

    if baseline is not None and treatment is not None:
        b = baseline.total_iteration_errors or 0
        t = treatment.total_iteration_errors or 0
        evidence["iteration_errors"] = {"before": b, "after": t}
        if t > b:
            return GateResult("correctness", False,
                              f"scheduler iteration errors rose {b} -> {t}", evidence)

    if run_tests:
        proc = subprocess.run(["python", "-m", "pytest", "tests", *test_args],
                              cwd=REPO_ROOT, capture_output=True, text=True)
        tail = (proc.stdout or proc.stderr).strip().splitlines()[-1:] or [""]
        evidence["pytest"] = tail[0]
        if proc.returncode != 0:
            return GateResult("correctness", False, f"fast suite failed: {tail[0]}", evidence)

    return GateResult("correctness", True, "fast suite green; no new iteration errors", evidence)


def cost_gate(baseline: Vitals, treatment: Vitals, *,
              max_startup_regression_s: float = 60.0,
              max_mem_regression_gb: float = 2.0) -> GateResult:
    """Latency wins that cost startup or memory must declare it, not hide it.

    Bucketed decode graphs were a real 1.8x win that quietly added ~21 minutes of startup; this
    gate is why that has to be stated rather than discovered later.
    """
    evidence: dict[str, Any] = {}
    problems: list[str] = []

    b_cap, t_cap = baseline.graph_capture_s, treatment.graph_capture_s
    if b_cap is not None and t_cap is not None:
        evidence["graph_capture_s"] = {"before": b_cap, "after": t_cap}
        if t_cap - b_cap > max_startup_regression_s:
            problems.append(f"startup +{t_cap - b_cap:.0f}s (graph capture)")

    b_mem, t_mem = baseline.peak_gpu_mem_gb, treatment.peak_gpu_mem_gb
    if b_mem is not None and t_mem is not None:
        evidence["peak_gpu_mem_gb"] = {"before": b_mem, "after": t_mem}
        if t_mem - b_mem > max_mem_regression_gb:
            problems.append(f"peak GPU memory +{t_mem - b_mem:.1f}GB")

    if problems:
        return GateResult("cost", False, "; ".join(problems) + " — declare it or reduce it",
                          evidence)
    return GateResult("cost", True, "no undeclared startup or memory regression", evidence)


@dataclass
class Judgement:
    passed: bool
    gates: dict[str, GateResult]

    @property
    def verdict(self) -> str:
        if self.passed:
            return "confirmed"
        if not self.gates["validity"].passed:
            return "invalid"
        sig = self.gates.get("significance")
        if sig and not sig.passed and sig.evidence.get("verdict") in ("noise",
                                                                     "insufficient_samples"):
            return "noise"
        return "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {name: g.to_dict() for name, g in self.gates.items()}


def judge(hypothesis: Hypothesis, baseline: Vitals, treatment: Vitals,
          *, run_tests: bool = True) -> Judgement:
    """Run all four gates in order. Later gates still run so the record is complete, but the
    verdict is decided by the earliest failure."""
    gates = {
        "validity": validity_gate(hypothesis, baseline, treatment),
        "significance": significance_gate(hypothesis, baseline, treatment),
        "correctness": correctness_gate(baseline, treatment, run_tests=run_tests),
        "cost": cost_gate(baseline, treatment),
    }
    return Judgement(passed=all(g.passed for g in gates.values()), gates=gates)
