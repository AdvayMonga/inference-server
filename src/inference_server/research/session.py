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
from pathlib import Path

from inference_server.research.gates import Judgement, judge
from inference_server.research.kb import save_experiment
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


def check_still_measurable(arms: dict[str, list[Vitals]], ref: str = "HEAD") -> None:
    """Refuse to judge panels that no longer describe the working tree.

    The rebase-after-measuring rule, enforced instead of remembered: an experiment vouches for the
    sha it measured, and once engine files move, the panels describe code that is gone. Cost of
    learning this the other way: one A100 run.
    """
    for name, panels in arms.items():
        sha = panels[-1].validity.engine_sha
        if not sha:
            continue
        drifted = _behavioural(_changed_files(sha, ref))
        if drifted:
            raise NotMeasurable(
                f"arm {name!r} measured {sha[:12]}, but engine files changed since:\n  "
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
) -> tuple[Judgement, Experiment]:
    """Steps 5 and 6 in one call: load the group, judge five gates, record the experiment.

    Returns the judgement and the record. The record is saved unless `record=False` — a rejected
    hypothesis is written with the same weight as a confirmed one, which is the whole point.
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
    j = judge(hypothesis, base, treat, run_tests=run_tests)
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
    if j.verdict != "confirmed":
        lines += ["", "This is a RESULT, not a failure. Write it to knowledge/ so the loop does "
                  "not re-propose it."]
    return "\n".join(lines)
