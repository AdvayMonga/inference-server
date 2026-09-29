"""Grader: checks an agent's workspace from outside the jail, trusting nothing inside it."""

from __future__ import annotations

import difflib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from .change_kinds import HIDDEN, may_write
from .jail import jail_command, srt_settings

# Caches a test run leaves behind; never copied into the pristine tree.
IGNORED = ("*/__pycache__/*", "__pycache__/*", "*.pyc", ".pytest_cache/*")
MAX_FILE_BYTES = 5_000_000
# Downloaded models only: its parent also holds the HF login token, which stays unreadable.
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class Audit:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


@dataclass
class TestRun:
    passed: bool
    returncode: int
    output: str


def _export(repo: Path, base: str, dest: Path) -> None:
    """Write the tree at `base` into `dest`, with no .git."""
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", base],
                         capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")


def _base_blobs(repo: Path, base: str) -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", base],
                         capture_output=True, check=True, text=True).stdout
    blobs = {}
    for entry in filter(None, out.split("\0")):
        meta, path = entry.split("\t", 1)
        blobs[path] = meta.split()[2]
    return blobs


def _blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _strip_hidden(tree: Path) -> None:
    for p in list(tree.rglob("*")):
        rel = p.relative_to(tree).as_posix()
        if p.is_file() and any(fnmatch(rel, h) for h in HIDDEN):
            p.unlink()


def prepare_workspace(repo: Path, base: str, dest: Path) -> None:
    """The agent's copy of `base`: no git history, hidden files removed."""
    _export(repo, base, dest)
    _strip_hidden(dest)


def audit(repo: Path, base: str, workspace: Path, kind: str) -> Audit:
    """Every difference between `workspace` and `base`, and which ones `kind` may not make."""
    blobs = _base_blobs(repo, base)
    a, seen = Audit(), set()
    for root, dirs, files in os.walk(workspace):
        for name in dirs + files:
            p = Path(root) / name
            rel = p.relative_to(workspace).as_posix()
            if p.is_symlink():
                a.violations.append(f"symlink: {rel}")
            elif p.is_dir() or any(fnmatch(rel, g) for g in IGNORED):
                continue
            elif not p.is_file():
                a.violations.append(f"not a regular file: {rel}")
            elif p.stat().st_size > MAX_FILE_BYTES:
                a.violations.append(f"over {MAX_FILE_BYTES} bytes: {rel}")
            else:
                seen.add(rel)
                if any(fnmatch(rel.lower(), h.lower()) for h in HIDDEN):
                    a.violations.append(f"recreated hidden file: {rel}")
                elif rel not in blobs:
                    a.added.append(rel)
                elif _blob_sha(p) != blobs[rel]:
                    a.modified.append(rel)
    a.deleted = sorted(p for p in blobs if p not in seen and not any(fnmatch(p, h) for h in HIDDEN))
    for rel in a.added:
        if not may_write(kind, rel, new_file=True):
            a.violations.append(f"{kind} may not add: {rel}")
    for rel in a.modified + a.deleted:
        if not may_write(kind, rel, new_file=False):
            a.violations.append(f"{kind} may not change: {rel}")
    return a


def pristine_tree(repo: Path, base: str, workspace: Path, result: Audit, dest: Path) -> None:
    """A fresh export of `base` plus only the audited changes; from here on, the change is this.

    Hidden files stay out: agent code runs in this tree, and its output goes back to agents.
    """
    if not result.ok:
        raise ValueError("refusing to build from a workspace that failed its audit")
    if dest.exists():
        shutil.rmtree(dest)
    _export(repo, base, dest)
    _strip_hidden(dest)
    for rel in result.added + result.modified:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / rel, dest / rel)
    for rel in result.deleted:
        (dest / rel).unlink()


def diff(repo: Path, base: str, tree: Path, result: Audit) -> str:
    """Unified diff of the audited changes, from `base` to `tree`; computed, never reported."""
    out = []
    for rel in sorted(result.added + result.modified + result.deleted):
        old = b"" if rel in result.added else subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:{rel}"], capture_output=True, check=True).stdout
        new = b"" if rel in result.deleted else (tree / rel).read_bytes()
        if b"\0" in old or b"\0" in new:
            out.append(f"Binary file {rel} changed\n")
            continue
        old = old.decode(errors="replace").splitlines(keepends=True)
        new = new.decode(errors="replace").splitlines(keepends=True)
        out += difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}")
    return "".join(out)


def run_lint(tree: Path) -> TestRun:
    """The same `ruff check .` CI runs; static, so it needs no jail."""
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=tree,
                          capture_output=True, text=True)
    return TestRun(proc.returncode == 0, proc.returncode, (proc.stdout + proc.stderr)[-5000:])


def run_tests(dest: Path, timeout_s: int = 1800) -> TestRun:
    """The fast suite on a pristine tree, inside the jail: its code is the agent's."""
    # Private to this run, but short: the jail's bridge sockets live here and Linux caps a
    # socket path at 108 bytes, which a directory beside a deep tree can exceed.
    tmp = Path(tempfile.mkdtemp(prefix="jail-")).resolve()
    env = {   # nothing inherited: agent code must not see our secrets
        "PATH": os.environ["PATH"], "HOME": str(Path.home()), "TMPDIR": str(tmp),
        "PYTHONPATH": str(dest / "src"),   # beats the editable install
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HOME": str(tmp / "hf-home"), "HF_HUB_CACHE": str(HF_HUB), "HF_HUB_OFFLINE": "1",
    }
    settings = srt_settings([dest.resolve(), tmp], Path(sys.prefix), domains=[],
                            readonly=[HF_HUB] if HF_HUB.exists() else [])
    argv = jail_command(settings, dest.with_suffix(".srt.json"),
                        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "-m", "not heavy and not needs_host"])   # CI runs the needs_host ones
    try:
        proc = subprocess.run(argv, cwd=dest, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return TestRun(proc.returncode == 0, proc.returncode, (proc.stdout + proc.stderr)[-5000:])
