"""review_1: a fresh agent reviews the change read-only; code turns its findings into a verdict."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..agents.brief import prompt, render
from ..agents.call import REVIEW_TOOLS, AgentReply, AgentSpec, call_agent
from ..method.runner import Result
from ..method.state import Record

SEVERITIES = ["important", "nit", "pre_existing"]
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings", "note"],
    "properties": {
        "findings": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["severity", "file", "line", "problem", "why"],
            "properties": {
                "severity": {"type": "string", "enum": SEVERITIES},
                "file": {"type": "string"},
                "line": {"type": ["integer", "null"]},
                "problem": {"type": "string"},
                "why": {"type": "string"},
            },
        }},
        "note": {"type": ["string", "null"]},
    },
}


def verdict(check_failed: bool, findings: list[dict], nit_pass_used: bool) -> str:
    """The outcome the findings earn: a failed check is a fix, important findings or a first
    round of nits send it back, anything else moves on."""
    severities = {f["severity"] for f in findings}
    if check_failed:
        return "fix"
    if "important" in severities:
        return "changes"
    if "nit" in severities and not nit_pass_used:
        return "changes"
    return "ok"


@dataclass
class Review1:
    turn_dir: Path
    call: Callable[[AgentSpec], AgentReply] = call_agent
    # PLACEHOLDER budgets: not yet agreed.
    max_turns: int = 100
    max_budget_usd: float = 2.0
    timeout_s: float = 1200

    def __call__(self, inputs: dict[str, Record]) -> Result:
        report = inputs["CheckReport"]
        previous = inputs.get("Review")   # only to carry the nit-pass flag; the agent never sees it
        attempt = len(list(self.turn_dir.glob("review_1-*"))) + 1
        scratch = self.turn_dir / f"review_1-{attempt}"
        scratch.mkdir()
        reply = self.call(AgentSpec(
            node="review_1", system=prompt("review_1"),
            prompt=render("review_1", {t: r for t, r in inputs.items() if t != "Review"}),
            workspace=Path(report.body["tree"]), scratch=scratch, output_schema=OUTPUT_SCHEMA,
            max_turns=self.max_turns, max_budget_usd=self.max_budget_usd,
            timeout_s=self.timeout_s, tools=REVIEW_TOOLS, writable=False,
        ))
        used = bool(previous and previous.body.get("nit_pass_used"))
        body = {"attempt": attempt, "cost_usd": reply.cost_usd, "turns": reply.turns,
                "nit_pass_used": used}   # carried through crashes too
        if reply.output is None:   # the reviewer failed, not the change: review again
            return Result("crashed", body | {"error": reply.error})
        findings = reply.output["findings"]
        outcome = verdict(report.outcome == "fail", findings, used)
        nit_pass = outcome == "changes" and not any(f["severity"] == "important" for f in findings)
        body |= {"findings": findings, "nit_pass_used": used or nit_pass}
        return Result(outcome, body, note=reply.output.get("note"))
