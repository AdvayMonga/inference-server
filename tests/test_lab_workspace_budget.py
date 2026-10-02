"""Workspace snapshots and restores through the ledger's blobs; the budget refuses to be crossed."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab import ledger
from lab.budget import Budget, BudgetExceeded
from lab.workspace import Workspace
from tests.lab_fixtures import make_repo


def test_snapshot_and_restore_round_trip(tmp_path):
    repo = make_repo(tmp_path)
    root = tmp_path / "ledger"
    ws = Workspace(repo, "HEAD", tmp_path / "ws", root)
    ws.create()
    assert ws.snapshot().id == "base"
    (ws.path / "src/inference_server/engine.py").write_text("def speed():\n    return 9\n")
    (ws.path / "src/inference_server/extra.py").write_text("y = 2\n")
    (ws.path / "README.md").unlink()
    snap = ws.snapshot()
    assert snap.files == ["src/inference_server/engine.py", "src/inference_server/extra.py"]
    assert snap.deleted == ["README.md"] and "may not change: README.md" in snap.violations
    assert (root / snap.patch).read_text().count("+++ b/") == 3     # two changes and the deletion
    assert ws.snapshot().id == snap.id                      # content-addressed, stable
    ws.restore("base")
    assert (ws.path / "README.md").exists() and not (ws.path / "src/inference_server/extra.py").exists()
    ws.restore(snap.id)
    assert (ws.path / "src/inference_server/engine.py").read_text().endswith("return 9\n")
    assert (ws.path / "src/inference_server/extra.py").exists() and not (ws.path / "README.md").exists()
    with pytest.raises(FileNotFoundError):
        ws.restore("nope")
    bundle = ledger.put_blob(b"not a snapshot", root)       # a blob that is not a workspace snapshot
    before = (ws.path / "src/inference_server/engine.py").read_text()
    with pytest.raises(FileNotFoundError):
        ws.restore(Path(bundle).name)
    assert (ws.path / "src/inference_server/engine.py").read_text() == before     # untouched


def test_budget_charges_and_refuses(tmp_path):
    b = Budget(1.0)
    b.charge(0.4, "a")
    assert b.remaining_usd == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded):
        b.charge(0.7, "b")
    assert b.spent_usd == pytest.approx(0.4)
    with pytest.raises(ValueError):
        b.charge(-1, "c")
    assert Budget.gpu_usd(1800, 3.70) == pytest.approx(1.85)


def test_budget_resumes_from_the_ledger(tmp_path):
    root = tmp_path / "ledger"
    ledger.append({"kind": "session", "run": "r1", "cost": {"usd": 0.5}}, root)
    ledger.append({"kind": "test", "run": "r1", "cost": {"usd": 0.25}}, root)
    ledger.append({"kind": "test", "run": "r2", "cost": {"usd": 9.0}}, root)
    assert Budget.resume(2.0, "r1", root).spent_usd == pytest.approx(0.75)
