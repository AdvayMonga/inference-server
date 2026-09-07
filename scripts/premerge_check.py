"""Refuse a merge that has no experiment behind it.

The rule: anything that changes engine behaviour reaches main only with an `experiments/*.json`
record whose gates are all green. This is what makes "clear cause and effect" enforceable
instead of aspirational — without it the loop can change code indefinitely and nobody can say
which change did what.

    python scripts/premerge_check.py <branch-or-sha>        # check
    python scripts/premerge_check.py <branch> --explain     # check and print the record

Docs, tests, benchmarks and the research package itself are exempt: they cannot change engine
behaviour, so requiring a GPU experiment for them would only teach people to bypass the gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_server.research.kb import load_experiments  # noqa: E402
from inference_server.research.schemas import REPO_ROOT  # noqa: E402

# Paths whose change can alter what the engine does at runtime.
BEHAVIOURAL_PREFIXES = ("src/inference_server/",)
EXEMPT_PREFIXES = (
    "src/inference_server/research/",   # the loop is not the engine
    "docs/", "tests/", "scripts/", "benchmarks/", "knowledge/", "experiments/", "monitoring/",
)


def changed_files(ref: str, base: str = "main") -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...{ref}"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def behavioural(files: list[str], diff_ref: str | None = None,
                base: str = "main") -> list[str]:
    """Engine files whose change could alter what the engine DOES.

    A file that only gains counters is not one of them. This gate exists to stop unproven
    PERFORMANCE claims reaching main, and adding a metric makes no performance claim — while
    demanding a GPU A/B in order to add a counter means observability can never be fixed when
    there is no GPU budget. Not hypothetical: the panel could not measure batch occupancy at
    all, that gap produced a wrong conclusion about the scheduler, and this gate blocked the fix.

    Pass `diff_ref` to enable the exemption. Without it every engine file counts, so callers
    that cannot supply a diff keep the conservative behaviour.
    """
    hits = []
    for f in files:
        if any(f.startswith(x) for x in EXEMPT_PREFIXES):
            continue
        if any(f.startswith(p) for p in BEHAVIOURAL_PREFIXES):
            if diff_ref is not None and observability_only(f, diff_ref, base):
                continue
            hits.append(f)
    return hits


# Prefixes for counter state that only accumulates observations. Assigning to anything else —
# existing state, a flag the engine reads — is behaviour, not instrumentation.
_COUNTER_PREFIXES = ("self._active_", "self._decode_")


def _is_instrumentation(line: str) -> bool:
    """Is this ADDED line a counter update, a stats-dict key, a comment or blank?"""
    t = line[1:].strip()                       # drop the leading '+'
    if not t or t.startswith("#") or t.startswith('"""') or t.startswith("'''"):
        return True
    if t.startswith('"') and '":' in t:        # a key added to a stats dict
        return True
    if any(t.startswith(p) for p in _COUNTER_PREFIXES):
        # Accumulation only. `len(...)` of existing state is allowed as the sampled value —
        # it reads, it cannot mutate. Any other call could do anything, so it is not.
        rhs = t.split("=")[-1]
        return "=" in t and ("(" not in rhs or rhs.strip().startswith("len("))
    if t.startswith("if ") and any(p in t for p in _COUNTER_PREFIXES):
        return True                            # the high-water guard
    return False


def observability_only(path: str, ref: str, base: str = "main") -> bool:
    """True when every ADDED line in this file is instrumentation and nothing was removed.

    Deliberately strict. One added line that does anything else disqualifies the file, and any
    deletion or modification of an existing line does too — this recognises purely additive
    instrumentation, nothing more.
    """
    diff = subprocess.run(["git", "diff", "-U0", f"{base}...{ref}", "--", path],
                          cwd=REPO_ROOT, capture_output=True, text=True).stdout
    saw_add = False
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            return False                       # a removed or changed line is not pure addition
        if line.startswith("+"):
            saw_add = True
            if not _is_instrumentation(line):
                return False
    return saw_add


def is_ancestor(sha: str, ref: str) -> bool:
    """Is `sha` reachable from `ref`? Cheap guard so a record cannot vouch for unrelated code."""
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, ref],
                       cwd=REPO_ROOT, capture_output=True)
    return r.returncode == 0


def resolve_sha(ref: str) -> str:
    return subprocess.run(["git", "rev-parse", ref], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="branch or sha proposed for merge")
    ap.add_argument("--base", default="main")
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()

    files = changed_files(args.ref, args.base)
    behaviour = behavioural(files, diff_ref=args.ref, base=args.base)

    if not behaviour:
        print(f"PASS  no engine-behaviour changes in {args.ref} "
              f"({len(files)} file(s) touched, all exempt)")
        return 0

    print(f"engine behaviour changed in {len(behaviour)} file(s):")
    for f in behaviour:
        print(f"    {f}")

    sha = resolve_sha(args.ref)

    # An experiment validates the commit it MEASURED, and committing the experiment record
    # itself moves HEAD past it. So accept a record whose treatment sha is an ancestor of the
    # tip — but only if nothing behavioural changed since, or the record would be vouching for
    # code it never ran.
    # Try EVERY candidate before giving up. Bailing on the first ancestor match let one stale
    # record block a valid one — the drift report is only useful if nothing else vouches.
    exp = None
    stale: list[tuple[str, str, list[str]]] = []
    for candidate in load_experiments():
        if candidate.source != "loop" or not candidate.all_gates_green():
            continue
        for arm in candidate.arms:
            if arm.name != "treatment" or not arm.sha or not is_ancestor(arm.sha, args.ref):
                continue
            drifted = behavioural(changed_files(args.ref, arm.sha),
                                  diff_ref=args.ref, base=arm.sha)
            if drifted:
                stale.append((candidate.id, arm.sha, drifted))
                continue
            exp = candidate
            break
        if exp:
            break

    if exp is None and stale:
        print("\nFAIL  every matching experiment predates an engine change:")
        for eid, vsha, vfiles in stale[:3]:
            print(f"        {eid} validated {vsha[:12]}, but since then:")
            for f in vfiles:
                print(f"            {f}")
        print("      Re-run the experiment against the current code "
              "(and do not rebase after measuring).")
        return 1

    if exp is None:
        print(f"\nFAIL  no loop-produced experiment validates {sha[:12]} or an ancestor of it.")
        print("      An engine change needs an experiments/*.json with green gates.")
        print("      Run the loop:  python -m inference_server.research.loop judge ...")
        return 1
    if args.explain:
        for name, g in exp.gates.items():
            mark = "ok  " if g.get("passed") else "FAIL"
            print(f"    [{mark}] {name}: {g.get('reason', '')}")

    ok, why = exp.authorises_merge()
    if not ok:
        print(f"\nFAIL  {why}")
        return 1

    delta = ", ".join(f"{k} {v.get('pct', 0):+.1f}%" for k, v in exp.delta.items()) or "recorded"
    print(f"\nPASS  {exp.id} verdict={exp.verdict}, all gates green ({delta})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
