"""The agent's metered tools. Each runs outside the jail, snapshots the workspace, and writes the ledger whether the agent likes it or not."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from lab import ledger
from lab.agent import ToolSpec
from lab.safety import grader

NOT_WIRED = {
    "bench": "the eval harness is not connected yet; nothing can be scored",
    "equiv": "output equivalence is not wired yet",
    "submit": "submit refuses until the eval harness is wired; nothing can become a win yet",
}


class Toolbox:
    """Bound to one session: knows the workspace, the budget, the ledger root and the run."""

    def __init__(self, session: Any):
        self.s = session
        self.violation: str | None = None

    # -- plumbing ------------------------------------------------------------------------
    def _record(self, kind: str, tool: str, args: dict, result: dict, snapshot, usd: float = 0.0) -> dict:
        body = {"kind": kind, "run": self.s.run_id, "session": self.s.session_id, "tool": tool,
                "args": args, "snapshot": snapshot.id, "snapshot_blob": snapshot.blob,
                "patch": snapshot.patch, "result": result, "cost": {"usd": usd}}
        return ledger.append(body, self.s.ledger_root)

    def _audited(self, tool: str, args: dict):
        """Snapshot, and stop the run on a surface violation: integrity failures are the one hard rule."""
        snap = self.s.workspace.snapshot()
        if snap.violations:
            self.violation = "; ".join(snap.violations)
            self._record("note", tool, args, {"violation": snap.violations}, snap)
            raise PermissionError(f"workspace violates the write surface: {self.violation}")
        return snap

    def _pristine(self) -> Path:
        a = self.s.workspace.audit()
        dest = self.s.run_dir / "pristine"
        grader.pristine_tree(self.s.workspace.repo, self.s.workspace.base, self.s.workspace.path, a, dest)
        return dest

    # -- tools ---------------------------------------------------------------------------
    def test(self, args: dict) -> str:
        snap = self._audited("test", args)
        tree = self._pristine()
        lint = grader.run_lint(tree)
        tests = grader.run_tests(tree) if lint.passed else grader.Run(False, -1, "skipped: lint failed")
        result = {"lint": lint.passed, "tests": tests.passed, "returncode": tests.returncode,
                  "lint_output": lint.output[-2000:], "test_output": tests.output[-4000:],
                  "changed": snap.files + snap.deleted, "scratch_left_out": self.s.workspace.audit().scratch}
        self._record("test", "test", args, result, snap)
        head = "PASS" if lint.passed and tests.passed else "FAIL"
        return f"{head} lint={'ok' if lint.passed else 'fail'} tests={'ok' if tests.passed else 'fail'}\n" \
               f"{lint.output[-1500:]}\n{tests.output[-3000:]}"

    def profile(self, args: dict) -> str:
        snap = self._audited("profile", args)
        tree = self._pristine()
        out = tree / "lab" / "runs"
        argv = [sys.executable, "-m", "lab.profile", "--requests", str(args.get("requests", 8)),
                "--prompt-len", str(args.get("prompt_len", 64)), "--max-tokens", str(args.get("max_tokens", 32)),
                "--out", str(out)]
        t0 = time.monotonic()
        proc = (self.s.profile_runner or default_profile_runner)(tree, argv)
        bundles = sorted(out.glob("*")) if out.exists() else []
        blob = ledger.put_blob(bundles[-1], self.s.ledger_root) if bundles else ""
        visible = ""
        if blob:   # a copy inside the workspace: the ledger is outside the jail, so the agent could not read it there
            dest = self.s.workspace.path / "lab" / "runs" / Path(blob).name
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(bundles[-1], dest)
            visible = str(Path("lab") / "runs" / Path(blob).name)
        result = {"returncode": proc.returncode, "bundle": blob, "workspace_copy": visible,
                  "seconds": time.monotonic() - t0, "output": (proc.stdout + proc.stderr)[-3000:]}
        self._record("profile", "profile", args, result, snap)
        if proc.returncode != 0 or not blob:
            return f"profile failed ({proc.returncode}):\n{result['output']}"
        return f"bundle at {visible} (in your workspace; left out of your change)\n" + "\n".join(
            f"  {p.name}" for p in sorted(dest.iterdir()))

    def ledger_tool(self, args: dict) -> str:
        match = {k: v for k, v in args.items() if k in ("kind", "session", "snapshot", "tool") and v}
        rows = list(ledger.records(self.s.ledger_root, **match))
        last = int(args.get("last") or 20)
        return "\n".join(json.dumps(r) for r in rows[-last:]) or "(no records)"

    def knowledge(self, args: dict) -> str:
        """Measured findings (`finding` records: knowledge/ seeded at run start, later ones as they land), raw."""
        q, status, tag = (args.get("query") or "").lower(), args.get("status"), args.get("tag")
        rows = []
        for r in ledger.records(self.s.ledger_root, kind="finding"):
            claim = r.get("claim")
            c = claim if isinstance(claim, dict) else {"text": claim}
            if status and c.get("status") != status:
                continue
            if tag and tag not in (c.get("tags") or []):
                continue
            if q and q not in json.dumps(c).lower():
                continue
            rows.append({"record": r["id"], "source": r.get("source"), "author": r.get("author"), "finding": c})
        last = int(args.get("last") or 20)
        return "\n".join(json.dumps(x) for x in rows[-last:]) or "(no findings match)"

    def budget(self, args: dict) -> str:
        b = self.s.budget
        return json.dumps({"cap_usd": b.cap_usd, "spent_usd": round(b.spent_usd, 4),
                           "remaining_usd": round(b.remaining_usd, 4)})

    def restore(self, args: dict) -> str:
        sid = args["snapshot"]
        self._audited("restore", args)
        self.s.workspace.restore(sid)
        snap = self._audited("restore", args)
        self._record("note", "restore", args, {"restored": sid, "now": snap.id}, snap)
        return f"workspace restored to {sid}"

    def note(self, args: dict) -> str:
        snap = self._audited("note", args)
        self._record("note", "note", args, {"text": args["text"]}, snap)
        return "noted"

    def _refused(self, name: str):
        def fn(args: dict) -> str:
            snap = self._audited(name, args)
            self._record(name if name in ledger.KINDS else "note", name, args,
                         {"verdict": "refused", "reason": NOT_WIRED[name]}, snap)
            return f"{name} refused: {NOT_WIRED[name]}"
        return fn

    def specs(self) -> list[ToolSpec]:
        obj = {"type": "object", "properties": {}}
        return [
            ToolSpec("test", "Lint and run the test suite on a pristine copy of your current workspace, "
                     "inside the referee's jail. Returns pass/fail and the tail of the output.", obj, self.test),
            ToolSpec("profile", "Run the engine on a synthetic workload under the profiler and store the raw "
                     "bundle (event timeline, chrome trace, memory, provenance) in the ledger.",
                     {"type": "object", "properties": {"requests": {"type": "integer"}, "prompt_len": {"type": "integer"},
                                                       "max_tokens": {"type": "integer"}}}, self.profile),
            ToolSpec("ledger", "Read past records: every tool call in this and earlier runs, with snapshots and results.",
                     {"type": "object", "properties": {"kind": {"type": "string"}, "session": {"type": "string"},
                                                       "snapshot": {"type": "string"}, "tool": {"type": "string"},
                                                       "last": {"type": "integer"}}}, self.ledger_tool),
            ToolSpec("knowledge", "Measured findings about this engine (hand-written ones and any recorded since), "
                     "raw. Filter by a text query, a status or a tag; newest last.",
                     {"type": "object", "properties": {"query": {"type": "string"}, "status": {"type": "string"},
                                                       "tag": {"type": "string"}, "last": {"type": "integer"}}},
                     self.knowledge),
            ToolSpec("budget", "Dollars left in this run.", obj, self.budget),
            ToolSpec("restore", "Put the workspace back to a snapshot id from the ledger ('base' resets it).",
                     {"type": "object", "required": ["snapshot"], "properties": {"snapshot": {"type": "string"}}}, self.restore),
            ToolSpec("note", "Leave a note for the human running the lab. It is recorded and changes nothing.",
                     {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}, self.note),
            ToolSpec("equiv", "Output equivalence of your change against the base.", obj, self._refused("equiv")),
            ToolSpec("bench", "Benchmark your change on the seen split.", obj, self._refused("bench")),
            ToolSpec("submit", "Submit your change for scoring on the held-out split. The only thing that can "
                     "produce a win.", obj, self._refused("submit")),
        ]


def default_profile_runner(tree: Path, argv: list[str]):
    """lab.profile in the pristine tree as agent code: jailed, weights read-only."""
    return grader.jailed(tree, argv, timeout_s=3600)


def clean_pristine(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "pristine", ignore_errors=True)
