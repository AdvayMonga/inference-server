"""check: code-only referee pass on a build — audit, kind, lint, tests; cheapest first."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..method.runner import Result
from ..method.state import Record
from ..safety.grader import TestRun, audit, diff, pristine_tree, run_lint, run_tests


def kind_consistent(kind: str, patch: str) -> TestRun:
    """The diff does what its declared kind allows; `obs` may only add lines.

    Textual only: added lines can still change behaviour. The reviewers judge that.
    """
    if kind == "obs":
        removed = [ln for ln in patch.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        if removed:
            return TestRun(False, 1, "obs may only add lines; removes:\n" + "\n".join(removed[:20]))
    return TestRun(True, 0, "")


@dataclass
class Check:
    repo: Path
    base: str
    turn_dir: Path

    def __call__(self, inputs: dict[str, Record]) -> Result:
        change = inputs["Change"].body
        workspace, kind = Path(change["workspace"]), change["kind"]
        a = audit(self.repo, self.base, workspace, kind)
        body = {"kind": kind, "added": a.added, "modified": a.modified, "deleted": a.deleted,
                "steps": []}
        if not a.ok:
            return Result("violation", body | {"violations": a.violations})
        tree = self.turn_dir / f"pristine-{change['attempt']}"
        pristine_tree(self.repo, self.base, workspace, a, tree)
        patch = diff(self.repo, self.base, tree, a)
        body |= {"tree": str(tree), "diff": patch}
        for name, run in (("kind", lambda: kind_consistent(kind, patch)),
                          ("lint", lambda: run_lint(tree)),
                          ("tests", lambda: run_tests(tree))):
            r = run()
            body["steps"].append({"step": name, "passed": r.passed, "output": r.output})
            if not r.passed:
                return Result("fail", body)
        # PLACEHOLDER: the equivalence step is not designed yet (on hold, 2026-09-29).
        body["steps"].append({"step": "equivalence", "passed": None, "output": "not designed yet"})
        return Result("pass", body)
