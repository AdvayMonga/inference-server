"""Refuse a merge that has no experiment behind it.

The rule: anything that changes engine behaviour reaches main only with an `experiments/*.json`
record whose four gates are green. This is what makes "clear cause and effect" enforceable
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


def behavioural(files: list[str]) -> list[str]:
    hits = []
    for f in files:
        if any(f.startswith(x) for x in EXEMPT_PREFIXES):
            continue
        if any(f.startswith(p) for p in BEHAVIOURAL_PREFIXES):
            hits.append(f)
    return hits


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
    behaviour = behavioural(files)

    if not behaviour:
        print(f"PASS  no engine-behaviour changes in {args.ref} "
              f"({len(files)} file(s) touched, all exempt)")
        return 0

    print(f"engine behaviour changed in {len(behaviour)} file(s):")
    for f in behaviour:
        print(f"    {f}")

    sha = resolve_sha(args.ref)
    matches = [e for e in load_experiments()
               if any(sha.startswith(a.sha) or a.sha.startswith(sha[:7]) for a in e.arms)]

    if not matches:
        print(f"\nFAIL  no experiment record references {sha[:12]}.")
        print("      An engine change needs an experiments/*.json with four green gates.")
        print("      Run the loop:  python -m inference_server.research.loop experiment <hyp-id>")
        return 1

    exp = matches[0]
    if args.explain:
        for name, g in exp.gates.items():
            mark = "ok  " if g.get("passed") else "FAIL"
            print(f"    [{mark}] {name}: {g.get('reason', '')}")

    if not exp.all_gates_green():
        failed = [n for n, g in exp.gates.items() if not g.get("passed")]
        print(f"\nFAIL  experiment {exp.id} has failing gate(s): {', '.join(failed) or 'missing'}")
        return 1

    delta = ", ".join(f"{k} {v.get('pct', 0):+.1f}%" for k, v in exp.delta.items()) or "recorded"
    print(f"\nPASS  {exp.id} verdict={exp.verdict}, all four gates green ({delta})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
