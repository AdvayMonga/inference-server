"""A tiny engine-shaped git repo for lab tests: fast, and shaped like the real one where the referee cares."""

from __future__ import annotations

import subprocess
from pathlib import Path


def make_repo(root: Path) -> Path:
    repo = root / "repo"
    files = {
        "src/inference_server/__init__.py": "",
        "src/inference_server/engine.py": "def speed():\n    return 1\n",
        "tests/test_engine.py": "from inference_server.engine import speed\n\n\ndef test_speed():\n    assert speed() >= 1\n",
        "tests/conftest.py": "",
        "corpus/a/seen.jsonl": "{}\n",
        "corpus/a/heldout.jsonl": "{\"secret\": 1}\n",
        "pyproject.toml": "[tool.ruff]\nline-length = 100\n[tool.ruff.lint]\nselect = ['F']\n",
        "README.md": "# tiny\n",
    }
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "base"], check=True)
    return repo
