"""The agent's copy of the engine: exported from git without history, snapshotted into the ledger on every tool call, restorable."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lab import ledger
from lab.safety import grader
from lab.safety.grader import Audit


@dataclass
class Snapshot:
    id: str              # content hash of the changed files
    files: list[str]     # added and modified, relative
    deleted: list[str]
    blob: str            # ledger blob dir holding the changed files, or "" when nothing changed
    patch: str           # ledger blob of the unified diff, or ""
    violations: list[str]


class Workspace:
    def __init__(self, repo: Path, base: str, path: Path, ledger_root: Path = ledger.ROOT):
        self.repo, self.base, self.path, self.ledger_root = Path(repo), base, Path(path), Path(ledger_root)

    def create(self) -> None:
        """Fresh export; contents are cleared in place so a process whose cwd is the workspace keeps it."""
        self.path.mkdir(parents=True, exist_ok=True)
        for child in self.path.iterdir():
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        grader.export(self.repo, self.base, self.path)

    def audit(self) -> Audit:
        return grader.audit(self.repo, self.base, self.path)

    def snapshot(self) -> Snapshot:
        """The changed files as a content-addressed blob plus a diff; the id ties every ledger record to code."""
        a = self.audit()
        if not a.changed:
            return Snapshot("base", [], [], "", "", a.violations)
        stage = Path(tempfile.mkdtemp(prefix="snap-"))
        try:
            for rel in a.added + a.modified:
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.path / rel, stage / rel)
            (stage / ".deleted").write_text("\n".join(a.deleted))
            blob = ledger.put_blob(stage, self.ledger_root)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        patch = ledger.put_blob(grader.diff(self.repo, self.base, self.path, a).encode(),
                                self.ledger_root, ".patch")
        return Snapshot(Path(blob).name, sorted(a.added + a.modified), a.deleted, blob, patch, a.violations)

    def restore(self, snapshot_id: str) -> None:
        """Back to `base` plus the files of `snapshot_id` ("base" alone resets)."""
        blob = self.ledger_root / "blobs" / snapshot_id
        if snapshot_id != "base" and not (blob / ".deleted").is_file():
            raise FileNotFoundError(f"{snapshot_id} is not a workspace snapshot in {self.ledger_root}")
        self.create()
        if snapshot_id == "base":
            return
        for p in blob.rglob("*"):
            rel = p.relative_to(blob)
            if p.is_file() and rel.name != ".deleted":
                (self.path / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, self.path / rel)
        for rel in (blob / ".deleted").read_text().split("\n"):
            if rel:
                (self.path / rel).unlink(missing_ok=True)
