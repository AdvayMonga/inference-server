"""review_1: the verdict rule, what the reviewer sees, and the read-only two-commit tree it reviews."""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from inference_server.research.agents.call import REVIEW_TOOLS, AgentReply, AgentSpec, _wrapper
from inference_server.research.nodes.review_1 import Review1, verdict
from inference_server.research.safety.grader import audit, prepare_workspace, pristine_tree

needs_srt = pytest.mark.skipif(shutil.which("srt") is None, reason="sandbox-runtime not installed")


def f(severity):
    return {"severity": severity, "file": "src/x.py", "line": 1, "problem": "p", "why": "w"}


@pytest.mark.parametrize("failed,findings,used,expected", [
    (True, [], False, "fix"),                              # a failed check is never approved
    (True, [f("nit")], True, "fix"),
    (False, [f("important")], True, "changes"),            # important always goes back
    (False, [f("nit")], False, "changes"),                 # the one nit pass
    (False, [f("nit")], True, "ok"),                       # nits again: move on
    (False, [f("pre_existing")], False, "ok"),             # not this change's problem
    (False, [], False, "ok"),
])
def test_the_verdict_is_computed_from_the_findings(failed, findings, used, expected):
    assert verdict(failed, findings, used) == expected


class FakeReviewer:
    def __init__(self, findings):
        self.findings, self.specs = findings, []

    def __call__(self, spec: AgentSpec) -> AgentReply:
        self.specs.append(spec)
        return AgentReply({"findings": self.findings, "note": None}, 0.01, 2, None)


def rec(type_, node, outcome, body):
    return SimpleNamespace(type=type_, node=node, outcome=outcome, body=body)


def inputs(tmp_path, check_outcome="pass", previous=None):
    got = {"Hypothesis": rec("Hypothesis", "hypothesize", "ok", {"claim": "faster prefill"}),
           "Change": rec("Change", "build", "ok", {"kind": "perf"}),
           "CheckReport": rec("CheckReport", "check", check_outcome, {"tree": str(tmp_path / "tree")})}
    if previous is not None:
        got["Review"] = rec("Review", "review_1", "changes", previous)
    return got


def test_the_reviewer_reviews_the_clean_tree_read_only_without_its_old_review(tmp_path):
    agent = FakeReviewer([f("nit")])
    res = Review1(tmp_path, call=agent)(inputs(tmp_path, previous={"nit_pass_used": False}))
    spec = agent.specs[0]
    assert spec.workspace == tmp_path / "tree" and spec.writable is False
    assert spec.tools == REVIEW_TOOLS
    assert "# CheckReport (from check, outcome pass)" in spec.prompt
    assert "# Review" not in spec.prompt            # it judges the change, not its last opinion
    assert res.outcome == "changes" and res.body["nit_pass_used"] is True
    assert res.body["findings"] == [f("nit")]


def test_after_the_nit_pass_the_same_nits_move_on(tmp_path):
    res = Review1(tmp_path, call=FakeReviewer([f("nit")]))(
        inputs(tmp_path, previous={"nit_pass_used": True}))
    assert res.outcome == "ok" and res.body["nit_pass_used"] is True


def test_a_crash_between_reviews_keeps_the_nit_pass_spent(tmp_path):
    crashed = Review1(tmp_path, call=lambda spec: AgentReply(None, None, 0, "timed out"))(
        inputs(tmp_path, previous={"nit_pass_used": True}))
    after = Review1(tmp_path, call=FakeReviewer([f("nit")]))(
        inputs(tmp_path, previous=crashed.body))
    assert after.outcome == "ok"


def test_a_crashed_reviewer_reviews_again(tmp_path):
    res = Review1(tmp_path, call=lambda spec: AgentReply(None, None, 0, "timed out"))(inputs(tmp_path))
    assert res.outcome == "crashed" and res.body["error"] == "timed out"


@pytest.mark.needs_host   # builds git repos
def test_the_clean_tree_is_two_commits_base_then_change(tmp_path):
    repo = tmp_path / "repo"
    for rel, text in {"src/inference_server/engine.py": "x = 1\n",
                      "corpus/c/heldout.jsonl": "{}\n"}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "base"], check=True)
    ws, tree = tmp_path / "ws", tmp_path / "tree"
    prepare_workspace(repo, "HEAD", ws)
    (ws / "src/inference_server/engine.py").write_text("x = 2\n")
    pristine_tree(repo, "HEAD", ws, audit(repo, "HEAD", ws, "perf"), tree)
    log = subprocess.run(["git", "-C", str(tree), "log", "--format=%s"], capture_output=True,
                         text=True, check=True).stdout.split()
    changed = subprocess.run(["git", "-C", str(tree), "diff", "--name-only", "HEAD~1", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.split()
    files = subprocess.run(["git", "-C", str(tree), "ls-files"], capture_output=True, text=True,
                           check=True).stdout
    assert log == ["change", "base"] and changed == ["src/inference_server/engine.py"]
    assert "heldout" not in files


@needs_srt
def test_a_read_only_session_cannot_write_its_workspace(tmp_path):
    ws = tmp_path / "tree"
    ws.mkdir()
    spec = AgentSpec("review_1", "", "", ws, tmp_path / "scratch", {}, 1, 1.0, 1.0, writable=False)
    spec.scratch.mkdir()
    _wrapper(spec, tmp_path / "claude")
    fs = json.loads((spec.scratch / "srt.json").read_text())["filesystem"]
    assert str(ws.resolve()) in fs["allowRead"] and str(ws.resolve()) not in fs["allowWrite"]
