"""The regime backfill (scripts/tools/backfill_kb_regime.py) leaves knowledge/ valid and scoped.

What this guards: a backfill that writes a regime `validate()` rejects, a range `covers()` cannot
read, or a table that drifts from the files it claims to describe.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "tools"))

import backfill_kb_regime as backfill  # noqa: E402

from inference_server.research.kb import KNOWLEDGE_DIR, covers, load_entries  # noqa: E402
from inference_server.research.schemas import KnowledgeEntry  # noqa: E402


def _bytes(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.json"))}


@pytest.fixture
def kb_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "knowledge"
    shutil.copytree(KNOWLEDGE_DIR, dst, ignore=shutil.ignore_patterns("README.md"))
    return dst


def test_every_entry_validates_with_a_known_regime():
    for e in load_entries():
        e.validate()
        assert e.regime is None or e.regime in KnowledgeEntry.REGIMES


def test_validity_ranges_are_readable_by_covers():
    for e in load_entries():
        for key, bound in e.validity_range.items():
            assert isinstance(bound, (str, int, float, bool, list)), (e.id, key)
            if isinstance(bound, list):
                assert bound, (e.id, key)
                if len(bound) == 2 and all(isinstance(b, (int, float)) for b in bound):
                    assert bound[0] <= bound[1], (e.id, key)


def test_table_names_only_entries_that_exist():
    """The table is a one-shot historical migration, so it must be a SUBSET of what is on disk.

    Not equality: entries written after the backfill set their own regime at write time and are
    not the migration's business. Requiring equality would mean every new finding breaks the suite.
    """
    on_disk = {p.stem for p in KNOWLEDGE_DIR.glob("*.json")}
    missing = set(backfill.MAPPING) - on_disk
    assert not missing, f"table names entries that no longer exist: {sorted(missing)}"


def test_committed_knowledge_already_matches_the_table(kb_copy: Path):
    # The table is the record of what was written; if knowledge/ drifts, this is what says so.
    assert backfill.apply(kb_copy)["written"] == []


def test_backfill_is_idempotent(kb_copy: Path):
    backfill.apply(kb_copy)
    once = _bytes(kb_copy)
    backfill.apply(kb_copy)
    assert _bytes(kb_copy) == once


def test_covers_selects_a100_findings_and_excludes_dev_machine(kb_copy: Path):
    backfill.apply(kb_copy)
    by_id = {e.id: e for e in load_entries(kb_copy)}
    situation = {"model": "gemma-4-e4b", "hardware": "A100-80GB", "concurrency": 8}
    hits = {i for i, e in by_id.items() if e.validity_range and covers(e, situation)}
    assert {"kb-20260903-001", "kb-20260905-b9bc66c6", "kb-20260906-f13d8e3d"} <= hits
    assert "kb-20260612-032" not in hits      # A10G / E2B
    assert "kb-20260514-025" not in hits      # MPS / E2B
