"""The four gates a change must clear before it may reach main. Judged IN ORDER.

Order is the point. Validity comes first because a significant-looking result from a harness
that never exercised the change is worse than no result — it is a confident wrong answer, and
this project shipped one for about a month.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from inference_server.research.compare import (
    Significance,
    comparable,
    significance,
    significance_replicated,
)
from inference_server.research.schemas import REPO_ROOT, GateResult, Hypothesis, Vitals

def _k1_share(wave_sizes: dict[str, int]) -> float:
    total = sum(wave_sizes.values())
    return (wave_sizes.get("1", 0) / total) if total else 1.0


# A change to X is only believable if the run actually exercised X. Maps a hypothesis's target
# metric / tags to the panel evidence that proves the harness reached it.
_EXERCISE_CHECKS: dict[str, Callable[[Vitals], tuple[bool, str]]] = {
    # Not "is any wave wider than 1" — that is too lenient. The measured reality was 83-91%
    # K=1, which still leaves a few K=2 waves while making wave planning inert. Require that
    # K=1 is not overwhelmingly dominant, matching the threshold attribute.py flags.
    "wave_sizes": lambda v: (
        _k1_share(v.wave_sizes) < 0.8,
        f"{_k1_share(v.wave_sizes):.0%} of prefill waves were K=1, so there is nothing for "
        f"wave composition to change (this is exactly how length-grouped waves looked like a "
        f"win before the histogram was measured)",
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


SANITY_TOLERANCE = 0.25     # a metric the change cannot touch may drift this much, no more


def sanity_gate(hypothesis: Hypothesis, baseline: Vitals, treatment: Vitals,
                tolerance: float = SANITY_TOLERANCE) -> GateResult:
    """Did something move that this change could not possibly have moved?

    The cheapest contamination detector there is. A prefill change cannot alter decode speed; if
    TPOT moves 2x between arms, the arms differ by something other than the treatment — a
    different machine, a warmed cache, a different container — and the headline number is
    meaningless no matter how significant it looks.
    """
    if not hypothesis.sanity_metrics:
        return GateResult("sanity", True, "no sanity metrics declared (consider adding some)",
                          {"declared": []})
    moved = {}
    for m in hypothesis.sanity_metrics:
        b, t = getattr(baseline, m, None), getattr(treatment, m, None)
        if not isinstance(b, (int, float)) or not isinstance(t, (int, float)) or not b:
            continue
        drift = abs(t - b) / abs(b)
        moved[m] = {"before": b, "after": t, "drift_pct": round(drift * 100, 1)}
        if drift > tolerance:
            return GateResult(
                "sanity", False,
                f"{m} moved {drift:.0%} ({b} -> {t}) but this change cannot affect it — the arms "
                f"are contaminated, so the headline result is not evidence",
                moved)
    return GateResult("sanity", True,
                      f"{len(moved)} untouchable metric(s) held within {tolerance:.0%}", moved)


def significance_gate(hypothesis: Hypothesis, baseline, treatment, **kw: Any) -> GateResult:
    """Does the effect clear the variance budget, on the metric predicted beforehand?

    Accepts a list of runs per arm (the honest path — variance across runs) or a single panel
    per arm, which can now only ever return insufficient_samples. See compare.py for why.
    """
    if isinstance(baseline, list) or isinstance(treatment, list):
        s: Significance = significance_replicated(
            list(baseline), list(treatment), hypothesis.predicted_metric,
            direction=hypothesis.predicted_direction, **kw)
    else:
        s = significance(baseline, treatment, hypothesis.predicted_metric,
                         direction=hypothesis.predicted_direction, **kw)
    # For an instrument_change, "no change" IS the success condition — the point is to see
    # something, or to make something testable, without moving the engine. Demanding a
    # significant effect there would either block behaviour-neutral instrumentation or push
    # people to dress it up as an optimisation.
    if hypothesis.kind in ("instrument_change", "harness_change"):
        if s.verdict in ("noise", "significant"):
            direction = ("no measurable change, as intended" if s.verdict == "noise"
                         else f"NOTE: it also moved {s.pct:+.1f}%")
            passed = s.verdict == "noise"
            return GateResult("significance", passed,
                              f"{hypothesis.kind}: {direction}", s.to_dict())
        return GateResult("significance", False,
                          f"{hypothesis.kind} needs a measurement showing it changed nothing; "
                          f"got {s.verdict}: {s.detail}", s.to_dict())

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
        if not self.gates.get("sanity", GateResult("sanity", True, "")).passed:
            return "contaminated"
        sig = self.gates.get("significance")
        if sig and not sig.passed and sig.evidence.get("verdict") in ("noise",
                                                                     "insufficient_samples"):
            return "noise"
        return "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {name: g.to_dict() for name, g in self.gates.items()}


def judge(hypothesis: Hypothesis, baseline, treatment,
          *, run_tests: bool = True) -> Judgement:
    """Run the gates in order. Later gates still run so the record is complete, but the
    verdict is decided by the earliest failure.

    Each arm is either one panel or a list of replicate runs. Replicates are required for a
    significant verdict; the other gates read the arm's representative (last) run.
    """
    b_rep = baseline[-1] if isinstance(baseline, list) else baseline
    t_rep = treatment[-1] if isinstance(treatment, list) else treatment
    gates = {
        "validity": validity_gate(hypothesis, b_rep, t_rep),
        "sanity": sanity_gate(hypothesis, b_rep, t_rep),
        "significance": significance_gate(hypothesis, baseline, treatment),
        "correctness": correctness_gate(b_rep, t_rep, run_tests=run_tests),
        "cost": cost_gate(b_rep, t_rep),
    }
    return Judgement(passed=all(g.passed for g in gates.values()), gates=gates)
