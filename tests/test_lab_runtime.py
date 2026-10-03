"""The whole loop on a tiny repo with a scripted provider: tools record, the budget bounds, violations stop the run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab import ledger, session
from lab.agent import AgentReply, AgentSpec
from lab.session import RunConfig
from tests.lab_fixtures import make_repo


class ScriptedProvider:
    """Calls the lab's tools in a scripted order, edits the workspace directly, returns a canned reply."""
    name = "scripted"

    def __init__(self, script):
        self.script, self.specs, self.outputs = script, [], []

    def run(self, spec: AgentSpec) -> AgentReply:
        self.specs.append(spec)
        tools = {t.name: t.fn for t in spec.tools}
        n = len(self.specs)
        reply = AgentReply({"status": "continue", "note": None}, 0.0, 0, None)
        for step in self.script(n, spec.workspace):
            if step[0] == "call":                       # like the SDK adapter: a refusal is text, not a crash
                try:
                    out = tools[step[1]](step[2] if len(step) > 2 else {})
                except Exception as e:
                    out = f"{step[1]} refused: {e}"
                self.outputs.append((step[1], out))
            elif step[0] == "reply":
                reply = step[1]
        return reply


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_NO_JAIL", "1")
    repo = make_repo(tmp_path)
    return RunConfig(goal="make speed() faster", budget_usd=2.0, repo=repo, base="HEAD",
                     runs_dir=tmp_path / "runs", ledger_root=tmp_path / "ledger", max_sessions=5)


def test_a_session_edits_tests_notes_and_stops(cfg):
    def script(n, ws: Path):
        (ws / "src/inference_server/engine.py").write_text("def speed():\n    return 2\n")
        yield ("call", "test")
        yield ("call", "budget")
        yield ("call", "note", {"text": "hello human"})
        yield ("call", "submit")
        yield ("call", "ledger", {"last": 5})
        yield ("reply", AgentReply({"status": "stop", "note": "done"}, 0.75, 12, None))
    p = ScriptedProvider(script)
    out = session.run(cfg, p)
    assert out["sessions"] == 1 and out["stopped"] == "agent" and out["spent_usd"] == pytest.approx(0.75)
    kinds = [r["kind"] for r in ledger.records(cfg.ledger_root)]
    assert kinds == ["test", "note", "submit", "session"]
    test_rec = next(ledger.records(cfg.ledger_root, kind="test"))
    assert test_rec["result"]["lint"] and test_rec["result"]["tests"]
    assert test_rec["snapshot"] != "base" and (cfg.ledger_root / test_rec["patch"]).exists()
    assert dict(p.outputs)["test"].startswith("PASS")
    assert "refused" in dict(p.outputs)["submit"]
    assert '"kind": "test"' in dict(p.outputs)["ledger"]
    sess = next(ledger.records(cfg.ledger_root, kind="session"))
    assert sess["status"] == "stop" and sess["claim"]["note"] == "done" and sess["cost"]["usd"] == 0.75
    assert "Goal: make speed() faster" in p.specs[0].prompt and "$2.00" in p.specs[0].prompt
    assert not (cfg.runs_dir / out["run"] / "pristine").exists()      # cleaned between sessions


def test_a_failing_change_is_reported_not_hidden(cfg):
    def script(n, ws: Path):
        (ws / "src/inference_server/engine.py").write_text("def speed():\n    return 0\n")
        yield ("call", "test")
        yield ("reply", AgentReply({"status": "stop", "note": None}, 0.1, 3, None))
    p = ScriptedProvider(script)
    session.run(cfg, p)
    assert dict(p.outputs)["test"].startswith("FAIL")
    assert next(ledger.records(cfg.ledger_root, kind="test"))["result"]["tests"] is False


def test_a_surface_violation_stops_the_run(cfg):
    def script(n, ws: Path):
        (ws / "tests/test_engine.py").write_text("def test_speed():\n    pass\n")
        yield ("call", "test")
        yield ("reply", AgentReply({"status": "continue", "note": None}, 0.2, 2, None))
    p = ScriptedProvider(script)
    out = session.run(cfg, p)
    assert out["sessions"] == 1 and out["stopped"].startswith("security: may not change: tests/test_engine.py")
    assert "refused: workspace violates" in dict(p.outputs)["test"]
    sess = next(ledger.records(cfg.ledger_root, kind="session"))
    assert "tests/test_engine.py" in sess["violation"]


def test_a_bash_only_violation_is_caught_at_session_end(cfg):
    def script(n, ws: Path):
        (ws / "pyproject.toml").write_text("")             # no tool call afterwards
        yield ("reply", AgentReply({"status": "continue", "note": None}, 0.1, 1, None))
    out = session.run(cfg, ScriptedProvider(script))
    assert out["sessions"] == 1 and out["stopped"].startswith("security:")
    assert "pyproject.toml" in next(ledger.records(cfg.ledger_root, kind="session"))["violation"]


def test_scratch_files_do_not_stop_the_run(cfg):
    def script(n, ws: Path):
        (ws / "lab/runs/x").mkdir(parents=True)
        (ws / "lab/runs/x/trace.json").write_text("{}")
        (ws / "notes.txt").write_text("thinking")
        yield ("call", "test")
        yield ("reply", AgentReply({"status": "stop", "note": None}, 0.1, 1, None))
    p = ScriptedProvider(script)
    out = session.run(cfg, p)
    assert out["stopped"] == "agent" and dict(p.outputs)["test"].startswith("PASS")
    assert set(next(ledger.records(cfg.ledger_root, kind="test"))["result"]["scratch_left_out"]) == {"lab/runs/x/trace.json", "notes.txt"}


def test_a_raising_provider_is_charged_at_the_cap_and_recorded(cfg):
    class Raising:
        name = "raising"
        def run(self, spec):
            raise RuntimeError("boom")
    out = session.run(cfg, Raising())
    sess = next(ledger.records(cfg.ledger_root, kind="session"))
    assert sess["error"].startswith("provider raised") and sess["cost"]["estimated"] is True
    assert sess["cost"]["usd"] == pytest.approx(2.0) and out["spent_usd"] == pytest.approx(2.0)


def test_the_budget_ends_the_run_and_resumes(cfg):
    def script(n, ws: Path):
        yield ("reply", AgentReply({"status": "continue", "note": None}, 0.9, 1, None))
    out = session.run(cfg, ScriptedProvider(script))
    assert out["sessions"] == 2 and out["stopped"] == "budget" and out["spent_usd"] == pytest.approx(1.8)
    cfg.run_id = out["run"]
    cfg.budget_usd = 3.0
    out2 = session.run(cfg, ScriptedProvider(script))     # one more fits, then the next overshoots
    assert out2["sessions"] == 2 and out2["spent_usd"] == pytest.approx(3.6) and "exceed" in out2["stopped"]
    ids = [r["session"] for r in ledger.records(cfg.ledger_root, kind="session")]
    assert ids == [f"{out['run']}-s{i}" for i in (1, 2, 3, 4)]      # numbering continues across resumes


def test_an_overspending_provider_is_recorded_and_stopped(cfg):
    def script(n, ws: Path):
        yield ("reply", AgentReply(None, 5.0, 1, "max_budget"))
    out = session.run(cfg, ScriptedProvider(script))
    assert out["sessions"] == 1 and "exceed" in out["stopped"] and out["spent_usd"] == pytest.approx(5.0)


def test_restore_and_profile_tools(cfg, monkeypatch):
    calls = []
    def fake_profile(tree, argv):
        calls.append(argv)
        out = tree / "lab" / "runs" / "b1"
        out.mkdir(parents=True)
        (out / "meta.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, "ok", "")
    from lab import tools as labtools
    monkeypatch.setattr(labtools, "default_profile_runner", fake_profile)
    first = {}
    def script(n, ws: Path):
        (ws / "src/inference_server/engine.py").write_text("def speed():\n    return 5\n")
        yield ("call", "test")
        first["snap"] = next(ledger.records(cfg.ledger_root, kind="test"))["snapshot"]
        (ws / "src/inference_server/engine.py").write_text("def speed():\n    return 6\n")
        yield ("call", "restore", {"snapshot": first["snap"]})
        yield ("call", "profile", {"requests": 2})
        yield ("reply", AgentReply({"status": "stop", "note": None}, 0.1, 1, None))
    p = ScriptedProvider(script)
    out = session.run(cfg, p)
    ws = cfg.runs_dir / out["run"] / "workspace"
    assert (ws / "src/inference_server/engine.py").read_text().endswith("return 5\n")
    prof = next(ledger.records(cfg.ledger_root, kind="profile"))
    assert prof["result"]["bundle"].startswith("blobs/") and (cfg.ledger_root / prof["result"]["bundle"] / "meta.json").exists()
    assert "--requests" in calls[0] and "2" in calls[0]
    assert "bundle at lab/runs/" in dict(p.outputs)["profile"]
    copy = ws / prof["result"]["workspace_copy"]
    assert (copy / "meta.json").exists()                      # readable from inside the jail
    assert prof["result"]["workspace_copy"].startswith("lab/runs/")   # scratch: never part of the change


def test_cli_parses(monkeypatch, cfg):
    seen = {}
    monkeypatch.setattr(session, "run", lambda c, provider=None: seen.setdefault("cfg", c) and {"ok": 1})
    session.main(["--goal", "g", "--budget", "3", "--base", "HEAD", "--max-sessions", "2"])
    assert seen["cfg"].goal == "g" and seen["cfg"].budget_usd == 3.0 and seen["cfg"].max_sessions == 2
    assert json.dumps({"ok": 1})


def test_knowledge_is_seeded_at_run_start_and_filterable(cfg):
    kdir = cfg.repo / "knowledge"
    kdir.mkdir()
    (kdir / "kb-1.json").write_text(json.dumps({"id": "kb-1", "status": "confirmed", "tags": ["kv", "decode"],
                                                 "summary": "Split-K decode attention cut TPOT 9% at batch 16"}))
    (kdir / "kb-2.json").write_text(json.dumps({"id": "kb-2", "status": "open", "tags": ["prefill"],
                                                 "summary": "Prefill graphs untested on long prompts"}))
    def script(n, ws: Path):
        yield ("call", "knowledge", {})
        yield ("call", "knowledge", {"query": "split-k"})
        yield ("call", "knowledge", {"status": "open"})
        yield ("call", "knowledge", {"tag": "nothing-here"})
        yield ("reply", AgentReply({"status": "stop", "note": None}, 0.1, 1, None))
    p = ScriptedProvider(script)
    session.run(cfg, p)
    outs = [o for name, o in p.outputs if name == "knowledge"]
    assert len(outs[0].splitlines()) == 2
    assert json.loads(outs[1])["finding"]["id"] == "kb-1"
    assert json.loads(outs[2])["finding"]["id"] == "kb-2"
    assert outs[3] == "(no findings match)"
    assert len(list(ledger.records(cfg.ledger_root, kind="finding"))) == 2
    session.run(cfg, ScriptedProvider(lambda n, ws: iter([("reply", AgentReply({"status": "stop", "note": None}, 0.1, 1, None))])))
    assert len(list(ledger.records(cfg.ledger_root, kind="finding"))) == 2     # unchanged files are not seeded twice
