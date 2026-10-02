"""lab.ledger appends stamped records, stores blobs by hash, guards held-out detail, and seeds findings."""

from __future__ import annotations

import json

import pytest

from lab import ledger


def test_append_stamps_and_reads_back(tmp_path):
    a = ledger.append({"kind": "test", "session": "s1", "snapshot": "x", "pass": True}, tmp_path)
    b = ledger.append({"kind": "bench", "session": "s1", "snapshot": "y"}, tmp_path)
    assert a["schema"] == ledger.SCHEMA and a["id"].startswith("ev-") and a["at"]
    assert [r["id"] for r in ledger.records(tmp_path)] == [a["id"], b["id"]]
    assert [r["id"] for r in ledger.records(tmp_path, snapshot="y")] == [b["id"]]


@pytest.mark.parametrize("record", [
    {"kind": "nope"},
    {"kind": "test", "id": "forged"},
    {"kind": "submit", "config": {"split": "heldout"}, "raw": {"per_request": "blobs/x"}},
    {"kind": "submit", "config": {"split": "heldout"},
     "metrics": {"ttft_p50_ms": {"delta_pct": -3, "by_class": {"cold_start": -9}}}},
])
def test_rejects(tmp_path, record):
    with pytest.raises(ledger.LedgerError):
        ledger.append(record, tmp_path)
    assert list(ledger.records(tmp_path)) == []


def test_heldout_aggregate_is_allowed(tmp_path):
    m = {"ttft_p50_ms": {"base": 800, "new": 700, "delta_pct": -12.5, "band_pct": 3, "verdict": "win"}}
    ledger.append({"kind": "submit", "config": {"split": "heldout"}, "metrics": m}, tmp_path)


def test_blobs_dedupe_by_content(tmp_path):
    root = tmp_path / "ledger"
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    rel = ledger.put_blob(d, root)
    assert ledger.put_blob(d, root) == rel and (root / rel / "meta.json").exists()
    assert ledger.put_blob(b"diff", root, ".patch").endswith(".patch")
    assert ledger.put_blob(b"diff", root, ".patch") == ledger.put_blob(b"diff", root, ".patch")


def test_seed_is_lossless_and_idempotent(tmp_path):
    kb = tmp_path / "knowledge"
    kb.mkdir()
    entry = {"id": "kb-1", "summary": "s", "validity_range": {"model": "m"}}
    (kb / "kb-1.json").write_text(json.dumps(entry))
    root = tmp_path / "ledger"
    assert ledger.seed(kb, root) == 1 and ledger.seed(kb, root) == 0
    [r] = ledger.records(root)
    assert r["kind"] == "finding" and r["author"] == "human" and r["claim"] == entry
