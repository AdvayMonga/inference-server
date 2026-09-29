"""The check node: audit, kind, lint, jailed tests on a clean tree; cheapest first."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from inference_server.research.method import graph
from inference_server.research.method.runner import Result, run_turn
from inference_server.research.method.state import TurnState
from inference_server.research.nodes.check import Check, kind_consistent
from inference_server.research.safety.grader import prepare_workspace

# Throwaway git repos, which the jail cannot create.
pytestmark = [pytest.mark.needs_host,
              pytest.mark.skipif(shutil.which("srt") is None, reason="sandbox-runtime not installed")]

FILES = {
    "src/inference_server/__init__.py": "",
    "src/inference_server/engine.py": "def f():\n    return 1\n",
    "tests/test_engine.py": "from inference_server.engine import f\n\n\n"
                            "def test_f():\n    assert f() == 1\n",
    "corpus/cold_start/heldout.jsonl": "{}\n",
}
ENGINE = "src/inference_server/engine.py"


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


def check_after(repo, tmp_path, edit, kind="perf"):
    """Run check on a workspace that `edit` changed; returns (result, turn_dir)."""
    turn = tmp_path / "turn"
    ws = turn / "workspace"
    prepare_workspace(repo, "HEAD", ws)
    edit(ws)
    change = SimpleNamespace(body={"workspace": str(ws), "kind": kind, "attempt": 1})
    return Check(repo, "HEAD", turn)({"Change": change}), turn


def steps(res: Result) -> dict:
    return {s["step"]: s["passed"] for s in res.body["steps"]}


def test_a_good_change_passes_with_a_computed_diff_and_a_clean_tree(repo, tmp_path):
    res, turn = check_after(repo, tmp_path, lambda w: (w / ENGINE).write_text(
        "def f():\n    return 1  # faster\n"))
    assert res.outcome == "pass"
    assert steps(res) == {"kind": True, "lint": True, "tests": True, "equivalence": None}
    assert "+    return 1  # faster" in res.body["diff"]
    tree = turn / "pristine-1"
    assert "# faster" in (tree / ENGINE).read_text()
    assert not (tree / "corpus/cold_start/heldout.jsonl").exists()   # agent code runs here


@pytest.mark.parametrize("edit,failed_step", [
    (lambda w: (w / ENGINE).write_text("def f():\n    return 2\n"), "tests"),
    (lambda w: (w / ENGINE).write_text("import os\n\n\ndef f():\n    return 1\n"), "lint"),
])
def test_the_first_failing_step_fails_the_check(repo, tmp_path, edit, failed_step):
    res, _ = check_after(repo, tmp_path, edit)
    assert res.outcome == "fail"
    assert res.body["steps"][-1]["step"] == failed_step and res.body["steps"][-1]["passed"] is False


def test_a_binary_file_is_reported_not_crashed_on(repo, tmp_path):
    res, _ = check_after(repo, tmp_path, lambda w: (w / "src/inference_server/blob.bin").write_bytes(
        b"\x00\xff\xfe"))
    assert "Binary file src/inference_server/blob.bin changed" in res.body["diff"]


def test_an_obs_change_may_only_add_lines(repo, tmp_path):
    res, _ = check_after(repo, tmp_path, lambda w: (w / ENGINE).write_text(
        "def f():\n    return 1 + 0\n"), kind="obs")
    assert res.outcome == "fail" and steps(res) == {"kind": False}
    assert kind_consistent("obs", "+++ b/x\n+counter += 1\n").passed


def test_touching_a_test_is_a_violation_that_stops_for_a_human(repo, tmp_path):
    res, _ = check_after(repo, tmp_path, lambda w: (w / "tests/test_engine.py").write_text(
        "def test_f():\n    pass\n"))
    assert res.outcome == "violation" and "tests/test_engine.py" in res.body["violations"][0]
    assert graph.next_node("check", "violation") == "ask_human"


def test_a_failed_check_goes_to_the_reviewer_not_straight_back(tmp_path):
    state = TurnState(tmp_path / "run" / "turn-001")
    impls = {n: (lambda _i, o=node.outcomes[0]: Result(o, {})) for n, node in graph.NODES.items()}
    impls["check"] = lambda _i: Result("fail", {})
    impls["review_1"] = lambda _i: Result("changes", {})
    run_turn(state, impls, pause=lambda rec: rec.node != "review_1")
    assert [r.node for r in state.records][-2:] == ["check", "review_1"]
