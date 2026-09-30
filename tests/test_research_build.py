"""The build node, its brief, its write hook and its jail wrapper, with a fake model call."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess

import pytest

from inference_server.research.agents import brief
from inference_server.research.agents.call import AgentReply, AgentSpec, _wrapper
from inference_server.research.method import graph
from inference_server.research.method.runner import Result, run_turn
from inference_server.research.method.state import TurnState
from inference_server.research.nodes.build import OUTPUT_SCHEMA, Build
from inference_server.research.safety.change_kinds import KINDS
from inference_server.research.safety.hooks import allowed, write_guard

# Tests the safety stack itself, with throwaway git repos the jail cannot create.
pytestmark = pytest.mark.needs_host

needs_srt = pytest.mark.skipif(shutil.which("srt") is None and not os.environ.get("REQUIRE_SRT"), reason="sandbox-runtime not installed")

FILES = {
    "src/inference_server/__init__.py": "",
    "src/inference_server/engine.py": "def f():\n    return 1\n",
    "src/inference_server/research/gates.py": "BAR = 1\n",
    "tests/test_engine.py": "def test_f():\n    pass\n",
    "corpus/cold_start/heldout.jsonl": "{}\n",
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "base"], check=True)
    return root


class FakeAgent:
    """Stands in for the model: records each spec, edits the engine, returns `output`."""

    def __init__(self, output=None):
        self.specs: list[AgentSpec] = []
        self.output = output or {"kind": "perf", "exactness": "exact", "note": None}

    def __call__(self, spec: AgentSpec) -> AgentReply:
        self.specs.append(spec)
        engine = spec.workspace / "src/inference_server/engine.py"
        engine.write_text(engine.read_text() + f"# attempt {len(self.specs)}\n")
        return AgentReply(self.output, 0.01, 3, None)


def stubs(**overrides):
    impls = {n: (lambda _i, o=node.outcomes[0]: Result(o, {"n": 1})) for n, node in graph.NODES.items()}
    return {**impls, **overrides}


def turn(tmp_path):
    return TurnState(tmp_path / "runs" / "run1" / "turn-001")


def test_first_attempt_gets_a_clean_workspace_and_a_deterministic_brief(repo, tmp_path):
    agent, telemetry = FakeAgent(), tmp_path / "telemetry"
    state = turn(tmp_path)
    run_turn(state, stubs(build=Build(repo, "HEAD", state.turn_dir, telemetry, call=agent)))
    spec = agent.specs[0]
    assert not (spec.workspace / "corpus/cold_start/heldout.jsonl").exists()
    assert not (spec.workspace / ".git").exists()
    assert "# Hypothesis (from hypothesize" in spec.prompt and "# Panel (from measure" in spec.prompt
    assert "build --ok--> check  <- you are here" in spec.prompt
    assert "CheckReport" not in spec.prompt        # nothing to retry from yet
    assert spec.system == brief.prompt("build") and spec.readonly == [telemetry]
    assert state.latest("Change").body["kind"] == "perf"


def test_a_retry_is_a_fresh_session_on_the_same_workspace_with_the_failure(repo, tmp_path):
    agent, checks = FakeAgent(), iter(["fail", "pass"])
    state = turn(tmp_path)
    review_1 = lambda i: Result("fix" if i["CheckReport"].outcome == "fail" else "ok", {})  # noqa: E731
    run_turn(state, stubs(build=Build(repo, "HEAD", state.turn_dir, call=agent),
                          check=lambda _i: Result(next(checks), {"tests": "1 failed"}),
                          review_1=review_1))
    first, second = agent.specs
    assert first.workspace == second.workspace and first.scratch != second.scratch
    assert "# CheckReport (from check, outcome fail)" in second.prompt
    assert "# Review (from review_1, outcome fix)" in second.prompt
    assert "# attempt 1" in (second.workspace / "src/inference_server/engine.py").read_text()


def test_a_note_is_passed_on_and_the_turn_continues(repo, tmp_path):
    agent = FakeAgent({"kind": "perf", "exactness": "exact", "note": "measure prefill first"})
    state = turn(tmp_path)
    run_turn(state, stubs(build=Build(repo, "HEAD", state.turn_dir, call=agent)))
    assert state.latest("Change").note == "measure prefill first"
    assert state.records[-1].node == "record"


def test_a_crashed_builder_is_run_again_on_the_same_workspace(repo, tmp_path):
    good, calls = FakeAgent(), []

    def flaky(spec):
        calls.append(spec)
        return AgentReply(None, 2.0, 200, "error_max_budget_usd") if len(calls) == 1 else good(spec)

    state = turn(tmp_path)
    run_turn(state, stubs(build=Build(repo, "HEAD", state.turn_dir, call=flaky)))
    builds = [r for r in state.records if r.node == "build"]
    assert [b.outcome for b in builds] == ["crashed", "ok"]
    assert builds[0].body["error"] == "error_max_budget_usd"


def test_the_declared_kind_can_only_be_one_the_loop_may_start():
    assert OUTPUT_SCHEMA["properties"]["kind"]["enum"] == sorted(KINDS)


def test_knowledge_is_an_empty_seam_until_designed():
    assert brief.knowledge_for({}) == []


@pytest.mark.parametrize("path,ok", [
    ("src/inference_server/engine.py", True),
    ("src/inference_server/research/gates.py", False),
    ("../outside.py", False),
    ("tests/test_engine.py", False),          # existing test: no kind may edit it
    ("tests/test_new_bug.py", True),          # a new test: fix may add it
])
def test_the_write_hook_refuses_before_the_write(repo, tmp_path, path, ok):
    ws = tmp_path / "ws"
    shutil.copytree(repo, ws, ignore=shutil.ignore_patterns(".git"))
    assert allowed(ws, path) is ok
    out = asyncio.run(write_guard(ws)({"tool_input": {"file_path": path}}, None, None))
    assert (out == {}) is ok
    if not ok:
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@needs_srt
def test_the_cli_runs_jailed_with_a_wiped_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "sk-secret")
    ws, ro = tmp_path / "ws", tmp_path / "telemetry"
    ws.mkdir()
    spec = AgentSpec("build", "", "", ws, tmp_path / "scratch", {}, 1, 1.0, 1.0, readonly=[ro])
    spec.scratch.mkdir()
    script = _wrapper(spec, tmp_path / "claude").read_text()
    assert "env -i" in script and "RUNPOD" not in script
    settings = json.loads((spec.scratch / "srt.json").read_text())
    assert str(ro) in settings["filesystem"]["allowRead"]
    assert str(ro) not in settings["filesystem"]["allowWrite"]
    assert str(ws.resolve()) in settings["filesystem"]["allowWrite"]
