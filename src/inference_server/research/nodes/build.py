"""build: a fresh agent session implements the hypothesis in the turn's workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..agents.brief import prompt, render
from ..agents.call import AgentReply, AgentSpec, call_agent
from ..method.runner import Result
from ..method.state import Record
from ..safety.change_kinds import KINDS
from ..safety.grader import prepare_workspace

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "exactness", "objection"],
    "properties": {
        "kind": {"type": "string", "enum": sorted(KINDS)},
        "exactness": {"type": "string", "enum": ["exact", "approximate"]},
        "objection": {"type": ["string", "null"]},
    },
}


@dataclass
class Build:
    repo: Path
    base: str                          # commit the turn started from
    turn_dir: Path
    telemetry: Path | None = None      # raw per-request rows, mounted read-only
    call: Callable[[AgentSpec], AgentReply] = call_agent
    # PLACEHOLDER budgets: proposed, not yet agreed.
    max_turns: int = 200
    max_budget_usd: float = 2.0
    timeout_s: float = 1800

    def __call__(self, inputs: dict[str, Record]) -> Result:
        workspace = self.turn_dir / "workspace"
        if not workspace.exists():   # first attempt; a retry keeps the previous attempt's edits
            prepare_workspace(self.repo, self.base, workspace)
        attempt = len(list(self.turn_dir.glob("build-*"))) + 1
        scratch = self.turn_dir / f"build-{attempt}"
        scratch.mkdir()
        reply = self.call(AgentSpec(
            node="build", system=prompt("build"), prompt=render("build", inputs),
            workspace=workspace, scratch=scratch,
            output_schema=OUTPUT_SCHEMA, max_turns=self.max_turns,
            max_budget_usd=self.max_budget_usd, timeout_s=self.timeout_s,
            readonly=[self.telemetry] if self.telemetry else [],
        ))
        body = {"workspace": str(workspace), "base": self.base, "attempt": attempt,
                "cost_usd": reply.cost_usd, "turns": reply.turns}
        if reply.output is None:   # the agent failed or ran out of budget: a human decides
            return Result("objection", body, objection=f"build agent failed: {reply.error}")
        body |= {"kind": reply.output["kind"], "exactness": reply.output["exactness"]}
        if reply.output.get("objection"):
            return Result("objection", body, objection=reply.output["objection"])
        return Result("ok", body)
