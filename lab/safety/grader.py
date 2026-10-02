"""Grader: checks the agent's workspace from outside the jail, trusting nothing inside it."""

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

from lab.safety import jail
from lab.safety.surfaces import HIDDEN, may_write

IGNORED = ("*/__pycache__/*", "__pycache__/*", "*.pyc", ".pytest_cache/*", ".ruff_cache/*")
MAX_FILE_BYTES = 5_000_000
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"   # weights only; the token beside it stays unreadable


@dataclass
class Audit:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def changed(self) -> list[str]:
        return sorted(self.added + self.modified + self.deleted)


@dataclass
class Run:
    passed: bool
    returncode: int
    output: str


def export(repo: Path, base: str, dest: Path) -> None:
    """The tree at `base` into `dest`, with no .git and no hidden files."""
    tar = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", base],
                         capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(dest, filter="data")
    for p in list(dest.rglob("*")):
        rel = p.relative_to(dest).as_posix()
        if p.is_file() and any(fnmatch(rel, h) for h in HIDDEN):
            p.unlink()


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


def audit(repo: Path, base: str, workspace: Path) -> Audit:
    """Every difference between `workspace` and `base`, and which ones the agent may not make."""
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
        if not may_write(rel, new_file=True):
            a.violations.append(f"may not add: {rel}")
    for rel in a.modified + a.deleted:
        if not may_write(rel, new_file=False):
            a.violations.append(f"may not change: {rel}")
    return a


def pristine_tree(repo: Path, base: str, workspace: Path, result: Audit, dest: Path) -> None:
    """A fresh export of `base` plus only the audited changes, as a two-commit repo (base, change)."""
    if not result.ok:
        raise ValueError("refusing to build from a workspace that failed its audit")
    if dest.exists():
        shutil.rmtree(dest)
    export(repo, base, dest)
    _commit(dest, "base", init=True)
    for rel in result.added + result.modified:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / rel, dest / rel)
    for rel in result.deleted:
        (dest / rel).unlink()
    _commit(dest, "change")


def _commit(tree: Path, message: str, init: bool = False) -> None:
    git = ["git", "-C", str(tree), "-c", "user.name=lab", "-c", "user.email=lab@localhost",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    if init:
        subprocess.run(["git", "-C", str(tree), "init", "-q"], check=True)
    subprocess.run(git + ["add", "-A", "--force"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "--no-verify", "-m", message], check=True)


def diff(repo: Path, base: str, tree: Path, result: Audit) -> str:
    """Unified diff of the audited changes, from `base` to `tree`."""
    out = []
    for rel in result.changed:
        old = b"" if rel in result.added else subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:{rel}"], capture_output=True, check=True).stdout
        new = b"" if rel in result.deleted else (tree / rel).read_bytes()
        if b"\0" in old or b"\0" in new:
            out.append(f"Binary file {rel} changed\n")
            continue
        out += difflib.unified_diff(old.decode(errors="replace").splitlines(keepends=True),
                                    new.decode(errors="replace").splitlines(keepends=True),
                                    f"a/{rel}", f"b/{rel}")
    return "".join(out)


def run_lint(tree: Path) -> Run:
    """The same `ruff check .` CI runs; static, so it needs no jail."""
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=tree,
                          capture_output=True, text=True)
    return Run(proc.returncode == 0, proc.returncode, (proc.stdout + proc.stderr)[-5000:])


def jailed(tree: Path, argv: list[str], *, timeout_s: float, env_extra: dict | None = None,
           domains: list[str] = ()) -> subprocess.CompletedProcess:
    """Run `argv` in `tree` as agent code: wiped env, jail around the tree and a private tmp, weights read-only."""
    tmp = Path(tempfile.mkdtemp(prefix="jail-")).resolve()   # short: srt's sockets live here
    env = {"PATH": os.environ["PATH"], "HOME": str(Path.home()), "TMPDIR": str(tmp),
           "PYTHONPATH": str(tree / "src") + os.pathsep + str(tree),
           "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HOME": str(tmp / "hf-home"), "HF_HUB_CACHE": str(HF_HUB), "HF_HUB_OFFLINE": "1",
           **(env_extra or {})}
    config = jail.settings([tree.resolve(), tmp], Path(sys.prefix), list(domains),
                           readonly=[HF_HUB] if HF_HUB.exists() else [])
    try:
        return subprocess.run(jail.wrap(config, tree.with_suffix(".srt.json"), argv), cwd=tree,
                              env=env, capture_output=True, text=True, timeout=timeout_s)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_tests(tree: Path, timeout_s: float = 1800) -> Run:
    """The fast suite on a pristine tree, inside the jail: its code is the agent's."""
    proc = jailed(tree, [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "-m", "not heavy and not needs_host"], timeout_s=timeout_s)
    return Run(proc.returncode == 0, proc.returncode, (proc.stdout + proc.stderr)[-5000:])
