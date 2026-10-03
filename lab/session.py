"""The one loop: while budget remains, start a session with the goal, the ledger and the tools, and record what it did."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from lab import agent, ledger
from lab.budget import Budget, BudgetExceeded
from lab.safety import grader
from lab.tools import Toolbox, clean_pristine
from lab.workspace import Workspace

REPO = Path(__file__).resolve().parents[1]
SYSTEM = (Path(__file__).parent / "prompts" / "system.md").read_text()
MIN_SESSION_USD = 0.25


@dataclass
class Session:
    """What one agent session sees and what its tools need: wired by `run`, read by `Toolbox`."""
    run_id: str
    session_id: str
    run_dir: Path
    workspace: Workspace
    budget: Budget
    ledger_root: Path
    profile_runner: object = None          # None: lab.tools.default_profile_runner


@dataclass
class RunConfig:
    goal: str
    budget_usd: float
    repo: Path = REPO
    base: str = "HEAD"
    runs_dir: Path = REPO / "lab" / "runs"
    ledger_root: Path = ledger.ROOT
    run_id: str | None = None             # resume an earlier run's workspace and spend
    max_sessions: int = 50
    max_turns: int = 200
    session_usd: float = 5.0              # per-session cap handed to the provider
    provider: str | None = None
    extra: dict = field(default_factory=dict)


def brief(cfg: RunConfig, s: Session, snapshot_id: str, n: int) -> str:
    tail = list(ledger.records(s.ledger_root, run=s.run_id))[-30:]
    return (f"Goal: {cfg.goal}\n\n"
            f"Run {s.run_id}, session {n}. Budget: ${s.budget.remaining_usd:.2f} of ${s.budget.cap_usd:.2f} left.\n"
            f"Workspace snapshot: {snapshot_id}. Base commit: {cfg.base}.\n\n"
            f"Last records of this run ({len(tail)}):\n" + "\n".join(json.dumps(r) for r in tail))


def _resolve(repo: Path, ref: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", ref], capture_output=True,
                          text=True, check=True).stdout.strip()


def run(cfg: RunConfig, provider: agent.Provider | None = None) -> dict:
    provider = provider or agent.load(cfg.provider)
    base = _resolve(cfg.repo, cfg.base)
    run_id = cfg.run_id or f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(cfg.runs_dir) / run_id
    ws = Workspace(cfg.repo, base, run_dir / "workspace", cfg.ledger_root)
    if not ws.path.exists():
        ws.create()
    budget = Budget.resume(cfg.budget_usd, run_id, cfg.ledger_root)
    ledger.seed(Path(cfg.repo) / "knowledge", cfg.ledger_root)   # new or changed findings only; read by `knowledge`
    done = sum(1 for _ in ledger.records(cfg.ledger_root, run=run_id, kind="session"))
    base_files = grader.base_files(cfg.repo, base)
    summary = {"run": run_id, "base": base, "sessions": 0, "stopped": None, "spent_usd": budget.spent_usd}

    for n in range(done + 1, done + cfg.max_sessions + 1):
        if budget.remaining_usd < MIN_SESSION_USD:
            summary["stopped"] = "budget"
            break
        s = Session(run_id, f"{run_id}-s{n}", run_dir, ws, budget, cfg.ledger_root)
        tools = Toolbox(s)
        snap = ws.snapshot()
        spec = agent.AgentSpec(system=SYSTEM, prompt=brief(cfg, s, snap.id, n), workspace=ws.path,
                               scratch=run_dir / "scratch" / s.session_id, tools=tools.specs(),
                               base_files=base_files, max_turns=cfg.max_turns,
                               max_budget_usd=min(cfg.session_usd, budget.remaining_usd))
        t0 = time.monotonic()
        try:
            reply = provider.run(spec)
        except Exception as e:                  # a provider that raises still gets charged and recorded
            reply = agent.AgentReply(None, spec.max_budget_usd, 0, f"provider raised: {e}", True)
        clean_pristine(run_dir)
        try:
            budget.charge(reply.cost_usd, s.session_id)
            over = None
        except BudgetExceeded as e:             # the provider overshot its cap; recorded, then the run ends
            budget.spent_usd += reply.cost_usd
            over = str(e)
        end = ws.snapshot()
        if end.violations and not tools.violation:      # a Bash write outside the surface, with no tool call after
            tools.violation = "; ".join(end.violations)
        status = (reply.output or {}).get("status") if reply.output else None
        ledger.append({"kind": "session", "run": run_id, "session": s.session_id, "provider": provider.name,
                       "model": spec.model, "snapshot": end.id, "snapshot_blob": end.blob, "patch": end.patch,
                       "cost": {"usd": reply.cost_usd, "estimated": reply.cost_estimated},
                       "turns": reply.turns, "seconds": time.monotonic() - t0,
                       "status": status, "error": reply.error, "violation": tools.violation,
                       "claim": {"note": (reply.output or {}).get("note")}}, cfg.ledger_root)
        summary["sessions"] = n - done
        summary["spent_usd"] = budget.spent_usd
        if tools.violation:
            summary["stopped"] = f"security: {tools.violation}"
            break
        if over:
            summary["stopped"] = over
            break
        if status == "stop":
            summary["stopped"] = "agent"
            break
    else:
        summary["stopped"] = "max_sessions"
    if summary["stopped"] is None:
        summary["stopped"] = "budget"
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.session")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--budget", type=float, required=True, help="dollars for the whole run")
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--run", help="resume this run id")
    ap.add_argument("--max-sessions", type=int, default=50)
    ap.add_argument("--session-usd", type=float, default=5.0)
    ap.add_argument("--provider")
    a = ap.parse_args(argv)
    out = run(RunConfig(goal=a.goal, budget_usd=a.budget, base=a.base, run_id=a.run,
                        max_sessions=a.max_sessions, session_usd=a.session_usd, provider=a.provider))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
