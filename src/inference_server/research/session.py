"""Steps 4-6 as one procedure, so every iteration runs the same one.

This module exists because the expensive mistakes in this project were never bad arithmetic —
they were an iteration that quietly differed from the last. Hand-assembling the judging step each
time is how a rebase-after-measuring slipped through and how the arm labels ended up parsed out of
a free-text field. The rigidity belongs in code that runs every time, not in remembering.

The library is the interface. `loop.py` is a thin shim over these same functions so the two
surfaces cannot drift apart again.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

from inference_server.research.accounting import Accounting
from inference_server.research.gates import Judgement, judge
from inference_server.research.kb import save_experiment
from inference_server.research.noise import NoiseBand, band_for
from inference_server.research.schemas import (
    REPO_ROOT,
    Arm,
    Experiment,
    Hypothesis,
    Vitals,
)

RUNS_DIR = REPO_ROOT / "runs"
ARM_TOKEN = "arm="

# Engine paths whose change invalidates a measurement. Mirrors premerge_check.py; kept in sync by
# the test that asserts they agree.
BEHAVIOURAL_PREFIXES = ("src/inference_server/",)
EXEMPT_PREFIXES = (
    "src/inference_server/research/",
    "src/inference_server/static/",
    "docs/", "tests/", "scripts/", "benchmarks/", "knowledge/", "experiments/", "monitoring/",
)


class NotMeasurable(Exception):
    """The panels cannot support a judgement. Raised rather than returned: a caller that ignores
    this is about to record a result that does not describe the code it names."""


def arm_label(panel: Vitals) -> str:
    """The arm a panel belongs to. Encoded in `notes` as `arm=<name>`.

    Deliberately NOT a Validity field: the validity block is provenance of a measurement, and
    adding to it bumps PANEL_VERSION and invalidates comparison to every existing panel. One
    parser here beats the same string split copied into each caller.
    """
    notes = panel.validity.notes or ""
    if ARM_TOKEN not in notes:
        return ""
    return notes.split(ARM_TOKEN, 1)[1].split()[0].strip().rstrip(",;")


def load_group(run_group: str, runs_dir: Path = RUNS_DIR) -> list[Vitals]:
    """Every panel in a run group, in the order it was measured."""
    panels = []
    for f in sorted(runs_dir.glob("*.json")):
        try:
            p = Vitals.load(f)
        except Exception:                      # a half-written or foreign file is not our concern
            continue
        if p.validity.run_group == run_group:
            panels.append(p)
    return sorted(panels, key=lambda p: p.validity.started_at)


def arms_for(run_group: str, runs_dir: Path = RUNS_DIR) -> dict[str, list[Vitals]]:
    """Split a run group into arms, checking the things that silently invalidate an A/B.

    Raises NotMeasurable when the panels cannot decide anything, rather than letting the gates
    render a verdict on them.
    """
    panels = load_group(run_group, runs_dir)
    if not panels:
        raise NotMeasurable(f"no panels in runs/ for run_group={run_group!r}")

    arms: dict[str, list[Vitals]] = {}
    for p in panels:
        label = arm_label(p)
        if not label:
            raise NotMeasurable(
                f"panel {p.validity.run_id} has no {ARM_TOKEN}<name> in its notes — the "
                f"instrument must tag which arm it measured")
        arms.setdefault(label, []).append(p)

    if len(arms) != 2:
        raise NotMeasurable(
            f"expected exactly 2 arms in {run_group!r}, found {sorted(arms)} — one variable "
            f"per experiment")

    thin = {n: len(v) for n, v in arms.items() if len(v) < 3}
    if thin:
        raise NotMeasurable(
            f"arm(s) {thin} have fewer than 3 runs. The first run of each arm is discarded as "
            f"warmup (three identical runs measured 1494/401/229ms), so >=3 leaves <2 to judge.")

    order = [arm_label(p) for p in panels]
    if not _alternating(order):
        raise NotMeasurable(
            f"arms ran in blocks, not alternating: {' '.join(o[:1].upper() for o in order)}. "
            f"Warmup drift then favours whichever arm ran later; interleave them (ABBA).")
    return arms


def _alternating(order: list[str]) -> bool:
    """Reject a run order where one arm is measured entirely before the other.

    ABBA and ABAB both pass. AABB does not: with a monotonic warmup drift the second block wins
    regardless of the change.
    """
    first_half, second_half = order[: len(order) // 2], order[len(order) // 2:]
    return not (len(set(first_half)) == 1 and len(set(second_half)) == 1
                and first_half[0] != second_half[0])


def _changed_files(a: str, b: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", a, b],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def _behavioural(files: list[str]) -> list[str]:
    return [f for f in files
            if any(f.startswith(p) for p in BEHAVIOURAL_PREFIXES)
            and not any(f.startswith(x) for x in EXEMPT_PREFIXES)]


def _is_ancestor(a: str, b: str) -> bool | None:
    """True/False when git can relate the two commits; None when it cannot answer at all.

    Tri-state on purpose. `--is-ancestor` exits 1 for "no" and 128 for "I cannot resolve that sha"
    (a shallow clone, a truncated panel), and collapsing those to one False would make a guard
    depend on an unanswered question reading as a negative.
    """
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                        cwd=REPO_ROOT, capture_output=True).returncode
    return rc == 0 if rc in (0, 1) else None


def _experiment_tip(shas: list[str]) -> str:
    """The arm sha that every arm sha is an ancestor of — the commit the experiment ends at.

    Empty when the arms do not lie on one chain (measured on divergent branches) or when git
    cannot resolve one of the shas, which leaves every arm checked against its own sha as before —
    the stricter rule, so an unresolvable sha loses the two-commit relief rather than the guard.
    """
    for c in shas:
        answers = [_is_ancestor(o, c) for o in shas]
        if None in answers:
            return ""
        if all(answers):
            return c
    return ""


def check_still_measurable(arms: dict[str, list[Vitals]], ref: str = "HEAD") -> None:
    """Refuse to judge panels that no longer describe the working tree.

    An arm is stale iff an engine file differs between `ref` and the experiment's tip — the arm
    sha every arm sha is an ancestor of — so a two-commit A/B's own treatment commit is not read
    as drift, while anything landed after every arm still fails every arm. Same-sha experiments,
    where the tip is that one sha, judge exactly as they did before.

    The rebase-after-measuring rule, enforced instead of remembered: an experiment vouches for the
    shas it measured, and once engine files move past all of them, the panels describe code that
    is gone. Cost of learning this the other way: one A100 run.
    """
    shas = [p[-1].validity.engine_sha for p in arms.values() if p[-1].validity.engine_sha]
    tip = _experiment_tip(shas)
    for name, panels in arms.items():
        sha = panels[-1].validity.engine_sha
        if not sha:
            continue
        drifted = _behavioural(_changed_files(tip or sha, ref))
        if drifted:
            measured = sha[:12] if tip in ("", sha) else f"{sha[:12]} (tip {tip[:12]})"
            raise NotMeasurable(
                f"arm {name!r} measured {measured}, but engine files changed since:\n  "
                + "\n  ".join(drifted)
                + "\n  Re-measure the current commit — do not rebase after measuring.")
        if any(p.validity.dirty for p in panels):
            raise NotMeasurable(
                f"arm {name!r} was measured from a dirty tree; the sha names code that was "
                f"not what ran.")


def judge_group(
    hypothesis: Hypothesis,
    run_group: str,
    *,
    baseline: str = "baseline",
    treatment: str = "treatment",
    branch: str = "",
    cost_usd: float = 0.0,
    run_tests: bool = True,
    check_drift: bool = True,
    record: bool = True,
    runs_dir: Path = RUNS_DIR,
    band: NoiseBand | None = None,
) -> tuple[Judgement, Experiment]:
    """Steps 5 and 6 in one call: load the group, judge five gates, record the experiment.

    Returns the judgement and the record. The record is saved unless `record=False` — a rejected
    hypothesis is written with the same weight as a confirmed one, which is the whole point.

    `band` is the measured null spread this harness shows with nothing changed. It is looked up
    from the baseline arm's own validity block when not passed, so the calibration applies by
    default rather than by anyone remembering it — that is the whole reason for measuring one.
    A situation with no band recorded judges exactly as before.

    `check_drift=False` is the escape hatch (`loop judge --no-drift-check`) and should now almost
    never be needed: the staleness rule measures drift from the experiment's tip, so a two-commit
    A/B no longer trips it. Turning it off disables the check for the treatment arm too, which is
    the arm where staleness actually matters.
    """
    hypothesis.validate()
    arms = arms_for(run_group, runs_dir)
    missing = {baseline, treatment} - set(arms)
    if missing:
        raise NotMeasurable(f"run group {run_group!r} has arms {sorted(arms)}, expected "
                            f"{baseline!r} and {treatment!r}")
    if check_drift:
        check_still_measurable(arms)

    base, treat = arms[baseline], arms[treatment]
    if band is None:
        band = band_for(base[-1])
    j = judge(hypothesis, base, treat, run_tests=run_tests, band=band)
    exp = Experiment(
        hypothesis_id=hypothesis.id,
        engine_sha_base=base[0].validity.engine_sha,
        branch=branch,
        arms=[Arm(baseline, base[-1].validity.engine_sha, [p.validity.run_id for p in base]),
              Arm(treatment, treat[-1].validity.engine_sha, [p.validity.run_id for p in treat])],
        verdict=j.verdict,
        gates=j.to_dict(),
        delta={hypothesis.predicted_metric: j.gates["significance"].evidence or {}},
        cost_usd=cost_usd,
        notes=(f"noise band {band.entry_id or band.key()} "
               f"({band.n_runs} null runs) applied" if band else
               "no noise band recorded for this situation; significance judged on the t-test "
               "alone"),
    )
    if record:
        save_experiment(exp)
    return j, exp


def format_judgement(j: Judgement, hypothesis: Hypothesis, exp: Experiment | None = None) -> str:
    """One rendering of a verdict, shared by the CLI and anything else that reports one."""
    lines = [f"hypothesis: {hypothesis.statement}",
             f"predicted:  {hypothesis.predicted_metric} {hypothesis.predicted_direction} "
             f"by {hypothesis.predicted_magnitude}", ""]
    for name in ("validity", "sanity", "significance", "correctness", "cost"):
        g = j.gates[name]
        lines.append(f"  [{'PASS' if g.passed else 'FAIL'}] {name:13} {g.reason}")
    lines += ["", f"verdict: {j.verdict.upper()}"]
    if exp is not None:
        lines.append(f"recorded {exp.id}")
        if exp.notes.startswith("no noise band"):   # saying it only in the JSON is not saying it
            lines.append(f"  !! {exp.notes}")
    if j.verdict != "confirmed":
        lines += ["", "This is a RESULT, not a failure. Write it to knowledge/ so the loop does "
                  "not re-propose it."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- derived metrics
#
# The headline metrics are properties of a SWEEP, not of a single rate: "how many tokens/s can we
# serve without breaking the latency budget" only means something once you have several rates and
# can see where it broke. The instrument records every rate-point and derives nothing; the
# derivation happens here, at read time.
#
# That ordering matters. An instrument that reduces its own data to one summary number has already
# made a judgement about what mattered, in the place where it is hardest to revisit. Recording
# everything and deriving late means a new question can be asked of old panels.

TRIAL_TOKEN = "trial="


def trial_label(panel: Vitals) -> str:
    """Which sweep a panel came from. Same encoding as arm=; see arm_label."""
    notes = panel.validity.notes or ""
    if TRIAL_TOKEN not in notes:
        return ""
    return notes.split(TRIAL_TOKEN, 1)[1].split()[0].strip().rstrip(",;")


def _rate_of(panel: Vitals) -> float | None:
    try:
        return float(panel.validity.harness_config.get("rates"))
    except (TypeError, ValueError):
        return None                    # a multi-rate string means this panel is not one point


def sweeps(panels: list[Vitals]) -> dict[str, list[Vitals]]:
    """Group rate-points by the sweep that produced them, ordered by rate.

    Falls back to one sweep per panel when the instrument did not tag a trial — an untagged panel
    is a sweep of one, which yields no knee rather than a wrong one.
    """
    out: dict[str, list[Vitals]] = {}
    for p in panels:
        out.setdefault(trial_label(p) or p.validity.run_id, []).append(p)
    for k in out:
        out[k].sort(key=lambda p: (_rate_of(p) is None, _rate_of(p) or 0.0))
    return out


def within_slo(panel: Vitals) -> bool | None:
    """Did this rate-point meet both latency budgets? None when the panel does not carry them."""
    if panel.slo_ttft_ms is None or panel.slo_tpot_ms is None:
        return None
    if panel.ttft_p95 is None or panel.tpot_p95 is None:
        return None
    return panel.ttft_p95 < panel.slo_ttft_ms and panel.tpot_p95 < panel.slo_tpot_ms


def sweep_headline(points: list[Vitals]) -> dict[str, float | None]:
    """The project's definition of success, derived from one sweep's rate-points.

    `max_tok_s_within_slo` is throughput at a fixed tail-latency budget — the primary metric in
    CLAUDE.md, and until now present only in the instrument's printed table, never in a machine
    record. `saturation_knee_rate` is the lowest rate that broke the budget: capacity, in the
    units the scheduler actually feels.
    """
    ok, broken = [], []
    for p in points:
        verdict, rate = within_slo(p), _rate_of(p)
        if verdict is None or rate is None:
            continue
        (ok if verdict else broken).append((rate, p))

    best = max((p.tok_s_within_slo or 0.0, r) for r, p in ok) if ok else None
    return {
        "max_tok_s_within_slo": best[0] if best else None,
        "best_rate_within_slo": best[1] if best else None,
        "saturation_knee_rate": min(r for r, _ in broken) if broken else None,
        "rates_measured": len(points),
        "rates_within_slo": len(ok),
    }


def headline(panels: list[Vitals]) -> list[dict[str, float | None]]:
    """One headline per sweep. Replicates stay separate — averaging them here would hide the
    run-to-run spread that decides significance."""
    return [{"trial": t, **sweep_headline(pts)} for t, pts in sorted(sweeps(panels).items())]


# --------------------------------------------------------------------------- primary metric
#
# notes/01: **GPU-seconds per session, subject to p95 TTFT under a fixed ceiling.**
#
# Pinning the latency and optimising the cost is what keeps both from degenerating: optimise
# latency alone and the loop discovers warm pools on day one; optimise cost alone and it scales
# to zero and latency explodes. The cost side only works if idle time counts, which is why the
# numerator is wall clock from PROCESS START and not the serving window.
#
# Derived late, from the panel's accounting block, for the same reason sweep_headline is: the
# instrument records the terms and this decides what they mean.


@dataclass
class PrimaryMetric:
    """GPU-seconds per session at a stated p95 TTFT ceiling, or the reasons it is unknowable."""

    gpu_s_per_session: float | None
    ceiling_ms: float | None
    ceiling_met: bool | None
    ttft_p95_ms: float | None
    gpu_s: float | None
    sessions: int | None
    refusals: list[str] = dc_field(default_factory=list)   # why there is no number
    caveats: list[str] = dc_field(default_factory=list)    # terms nobody measured
    runs: int = 0

    def __bool__(self) -> bool:
        return self.gpu_s_per_session is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format(self) -> str:
        if not self:
            return ("primary metric: REFUSED — the accounting is incomplete\n  "
                    + "\n  ".join(self.refusals))
        verdict = ("MET" if self.ceiling_met else "BROKEN")
        lines = [f"primary metric: {self.gpu_s_per_session:.2f} GPU-seconds per session "
                 f"at a p95 TTFT ceiling of {self.ceiling_ms:.0f}ms [{verdict}]",
                 f"  {self.gpu_s:.1f} GPU-s (wall from process start, idle included) / "
                 f"{self.sessions} session(s) over {self.runs} run(s)",
                 f"  measured p95 TTFT {self.ttft_p95_ms:.0f}ms"]
        lines += [f"  CAVEAT: {c}" for c in self.caveats]
        return "\n".join(lines)


def primary_metric(panels: Vitals | list[Vitals], *,
                   ceiling_ms: float | None = None) -> PrimaryMetric:
    """The project's primary metric, or a refusal naming every term that is missing.

    It refuses rather than defaulting a missing term to zero. That is the whole point: an
    unmeasured term reads as free, and notes/07 says a loop will eventually move cost into
    whatever reads as free. In particular it will NOT fall back to `Vitals.wall_s` when
    `wall_s_from_process_start` is absent — wall_s starts at the first arrival, so substituting
    it would silently exclude model load, which on a cold replica is most of the bill.

    Several panels aggregate by summing GPU-seconds and sessions, and the ceiling must be met by
    every one of them. Each panel must come from its OWN process, or its share of the load time
    is counted twice; `replay_local.py` guarantees that by starting one server per run.
    """
    points = [panels] if isinstance(panels, Vitals) else list(panels)
    refusals: list[str] = []
    caveats: list[str] = []
    gpu_s = 0.0
    sessions = 0
    ttft_p95: float | None = None
    ceilings: set[float] = set()
    met = True

    if not points:
        return PrimaryMetric(None, ceiling_ms, None, None, None, None,
                             refusals=["no panels"], runs=0)

    for p in points:
        rid = p.validity.run_id
        acc = Accounting.of(p)
        if acc is None:
            refusals.append(f"{rid}: no accounting block — this panel cannot say what the run "
                            f"cost before its first request")
            continue
        if acc.wall_s_from_process_start is None:
            refusals.append(f"{rid}: wall_s_from_process_start is unmeasured, and wall_s is not "
                            f"a substitute for it (it starts at the first arrival, so model "
                            f"load would not be counted)")
        else:
            gpu_s += acc.wall_s_from_process_start
        if acc.sessions_served is None:
            refusals.append(f"{rid}: sessions_served is unmeasured, so there is nothing to "
                            f"divide the GPU-seconds by")
        else:
            sessions += acc.sessions_served
        caveats += [f"{rid}: {term} not accounted — {why}"
                    for term, why in sorted(acc.unmeasured.items())]

        ceiling = ceiling_ms if ceiling_ms is not None else p.slo_ttft_ms
        if ceiling is None:
            refusals.append(f"{rid}: no p95 TTFT ceiling — the metric is meaningless without "
                            f"the latency it is subject to")
        else:
            ceilings.add(float(ceiling))
        if p.ttft_p95 is None:
            refusals.append(f"{rid}: no measured ttft_p95, so the ceiling cannot be checked")
        else:
            ttft_p95 = p.ttft_p95 if ttft_p95 is None else max(ttft_p95, p.ttft_p95)
            if ceiling is not None:
                met = met and p.ttft_p95 < ceiling

        # ttft_p95 is computed over SUCCESSFUL requests only, so failing a request would
        # improve it — a reward hack. A failed request is a first token that never arrived, so
        # it ranks above every real one. Nearest-rank p95 over ALL requests is finite only if
        # at least ceil(0.95 * n) of them got a first token. Integer arithmetic, not 0.95 * n:
        # the floor-interpolation index let 1 failure in 8 through. Closed HERE only — the
        # gates still read ttft_p95 as the instruments compute it (kb-20260918-4f4c85b7).
        n_req, n_ok = p.validity.harness_config.get("n_requests"), p.validity.n_samples
        if not isinstance(n_req, int) or n_req < n_ok:
            refusals.append(f"{rid}: harness_config['n_requests'] is {n_req!r} against "
                            f"n_samples={n_ok}, so failed requests cannot be counted — and an "
                            f"uncounted failure would flatter the ceiling")
        elif n_req > n_ok:
            caveats.append(f"{rid}: {n_req - n_ok} of {n_req} requests produced no first token")
            if n_ok < (95 * n_req + 99) // 100:
                met = False

    if len(ceilings) > 1:
        refusals.append(f"panels state different ceilings {sorted(ceilings)}; one metric needs "
                        f"one SLO")
    if not sessions and not refusals:
        refusals.append("no sessions served — cost per session is undefined")

    if refusals:
        return PrimaryMetric(None, next(iter(ceilings), ceiling_ms), None, ttft_p95, None, None,
                             refusals=refusals, caveats=caveats, runs=len(points))
    return PrimaryMetric(
        gpu_s_per_session=round(gpu_s / sessions, 3),
        ceiling_ms=ceilings.pop(),
        ceiling_met=met,
        ttft_p95_ms=ttft_p95,
        gpu_s=round(gpu_s, 3),
        sessions=sessions,
        caveats=caveats,
        runs=len(points),
    )
