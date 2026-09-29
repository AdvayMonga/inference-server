"""The safety stack: deny-by-default write surfaces, the grader's audit and pristine tests, the jail."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from inference_server.research.safety.change_kinds import may_write
from inference_server.research.safety.grader import (
    MAX_FILE_BYTES, audit, prepare_workspace, pristine_tree, run_tests,
)
from inference_server.research.safety.jail import API_HOST, srt_settings

def pristine_tests(repo, base, ws, result, dest):
    pristine_tree(repo, base, ws, result, dest)
    return run_tests(dest)


# Tests the safety stack itself, with throwaway git repos the jail cannot create.
pytestmark = pytest.mark.needs_host

# CI sets REQUIRE_SRT so a missing jail fails these tests instead of silently skipping them.
needs_srt = pytest.mark.skipif(shutil.which("srt") is None and not os.environ.get("REQUIRE_SRT"),
                               reason="sandbox-runtime not installed")

FILES = {
    "README.md": "readme\n",
    "src/inference_server/__init__.py": "",
    "src/inference_server/engine.py": "def f():\n    return 1\n",
    "src/inference_server/research/gates.py": "BAR = 1\n",
    "tests/test_engine.py": "from inference_server.engine import f\n\n"
                            "def test_f():\n    assert f() == 1\n",
    "tests/test_where.py": "import os, inference_server\n\n"
                           "def test_imports_the_tree_under_test():\n"
                           "    assert inference_server.__file__.startswith(os.getcwd())\n",
    "corpus/cold_start/seen.jsonl": "{}\n",
    "corpus/cold_start/heldout.jsonl": "{\"secret\": 1}\n",
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


@pytest.fixture
def ws(repo, tmp_path):
    w = tmp_path / "ws"
    prepare_workspace(repo, "HEAD", w)
    return w


@pytest.mark.parametrize("kind,path,new,allowed", [
    ("perf", "src/inference_server/engine.py", False, True),
    ("perf", "src/inference_server/research/gates.py", False, False),   # the evaluator
    ("perf", "src/inference_server/conftest.py", True, False),          # import-time hook
    ("perf", "tests/test_engine.py", False, False),
    ("perf", "README.md", False, False),                                # unlisted = denied
    ("perf", "pyproject.toml", False, False),
    ("perf", "corpus/cold_start/heldout.jsonl", True, False),
    ("fix", "tests/test_new_bug.py", True, True),                       # fix may add a test
    ("fix", "tests/test_engine.py", False, False),                      # but never edit one
    ("perf", "src/inference_server/CONFTEST.PY", True, False),          # macOS ignores case
    ("perf", "src/inference_server/models/ruff.toml", True, False),     # would silence lint
    ("perf", "src/inference_server/pyproject.toml", True, False),
])
def test_write_surfaces_are_deny_by_default(kind, path, new, allowed):
    assert may_write(kind, path, new_file=new) is allowed


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="may not start"):
        may_write("method", "src/inference_server/research/method/graph.py", new_file=False)


def test_the_workspace_has_no_history_and_no_hidden_files(ws):
    assert not (ws / ".git").exists()
    assert not (ws / "corpus/cold_start/heldout.jsonl").exists()
    assert (ws / "corpus/cold_start/seen.jsonl").exists()


def test_an_engine_edit_passes_the_audit(repo, ws):
    (ws / "src/inference_server/engine.py").write_text("def f():\n    return 1  # faster\n")
    (ws / "src/inference_server/__pycache__").mkdir()
    (ws / "src/inference_server/__pycache__/engine.pyc").write_bytes(b"x")   # ignored
    a = audit(repo, "HEAD", ws, "perf")
    assert a.ok and a.modified == ["src/inference_server/engine.py"] and not a.added


@pytest.mark.parametrize("tamper,reason", [
    (lambda w: (w / "tests/test_engine.py").write_text("def test_f():\n    pass\n"), "may not change"),
    (lambda w: (w / "src/inference_server/research/gates.py").write_text("BAR = 0\n"), "may not change"),
    (lambda w: (w / "README.md").unlink(), "may not change"),
    (lambda w: (w / "corpus/cold_start/heldout.jsonl").write_text("{}\n"), "recreated hidden"),
    (lambda w: (w / "src/inference_server/link.py").symlink_to("/etc/hosts"), "symlink"),
    (lambda w: os.mkfifo(w / "src/inference_server/pipe"), "not a regular file"),
    (lambda w: (w / "src/inference_server/big.py").write_bytes(b"#" * (MAX_FILE_BYTES + 1)), "bytes"),
])
def test_the_audit_catches_tampering(repo, ws, tamper, reason):
    tamper(ws)
    a = audit(repo, "HEAD", ws, "perf")
    assert not a.ok and any(reason in v for v in a.violations)


@needs_srt
def test_pristine_tests_run_the_real_tree_and_real_tests(repo, ws, tmp_path):
    a = audit(repo, "HEAD", ws, "perf")
    run = pristine_tests(repo, "HEAD", ws, a, tmp_path / "pristine")
    assert run.passed, run.output   # includes test_where: imports come from the pristine tree


@needs_srt
def test_pristine_tests_catch_a_broken_change(repo, ws, tmp_path):
    (ws / "src/inference_server/engine.py").write_text("def f():\n    return 2\n")
    a = audit(repo, "HEAD", ws, "perf")
    assert a.ok
    assert not pristine_tests(repo, "HEAD", ws, a, tmp_path / "pristine").passed


def test_pristine_tests_refuse_a_failed_audit(repo, ws, tmp_path):
    (ws / "tests/test_engine.py").write_text("def test_f():\n    pass\n")
    with pytest.raises(ValueError, match="audit"):
        pristine_tests(repo, "HEAD", ws, audit(repo, "HEAD", ws, "perf"), tmp_path / "p")


@needs_srt
def test_agent_code_under_test_is_jailed(repo, ws, tmp_path, monkeypatch):
    """Engine code runs during tests, so the pristine run gets no network, home or secrets."""
    monkeypatch.setenv("RUNPOD_API_KEY", "sk-secret")
    (ws / "src/inference_server/engine.py").write_text(
        "import os, urllib.request\n"
        "def f():\n    return 1\n"
        "def leaks():\n    found = []\n"
        "    if os.environ.get('RUNPOD_API_KEY'): found.append('secret')\n"
        "    try: urllib.request.urlopen('https://example.com', timeout=5); found.append('net')\n"
        "    except Exception: pass\n"
        "    try: open(os.path.expanduser('~/.zshrc')).read(); found.append('home')\n"
        "    except Exception: pass\n"
        "    return found\n")
    (ws / "tests/test_leak.py").write_text(
        "from inference_server.engine import leaks\n\n"
        "def test_nothing_leaks():\n    assert leaks() == []\n")
    a = audit(repo, "HEAD", ws, "fix")
    run = pristine_tests(repo, "HEAD", ws, a, tmp_path / "pristine")
    assert run.passed, run.output


def test_the_jail_writes_only_the_workspace_and_talks_only_to_the_api(tmp_path):
    ws, venv = tmp_path / "ws", tmp_path / "venv"
    s = srt_settings([ws], venv, domains=[API_HOST])
    assert s["filesystem"]["allowWrite"] == [str(ws)]
    assert s["filesystem"]["denyRead"] == [str(Path.home())]
    assert set(s["filesystem"]["allowRead"]) == {str(ws), str(venv)}
    assert s["network"]["allowedDomains"] == [API_HOST]
    assert s["enableWeakerNestedSandbox"] is False
