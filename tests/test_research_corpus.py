"""The corpus contract: frozen, hashed, deterministic, split. A changed trace is a new version.

Runs in the loop CI lane (pytest + ruff only): the builder pulls prompt_bank.py, which is plain
strings, and nothing here touches the engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from inference_server.research.corpus import (
    CORPUS_DIR,
    SPLITS,
    CorpusError,
    TraceRequest,
    WorkloadClass,
    load_manifest,
    load_trace,
    read_trace,
    write_trace,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "tools"))
import build_corpus as bc  # noqa: E402

CLASSES = ("cold_start", "steady_interactive", "long_context")


# ---------------------------------------------------------------- the committed corpus

def test_committed_corpus_verifies_and_has_the_three_classes():
    m = load_manifest(CORPUS_DIR)
    assert set(m.classes) == set(CLASSES)
    for name in CLASSES:
        for split in SPLITS:
            _, trace = load_trace(name, split, CORPUS_DIR)
            assert 5 <= len(trace) <= 300, f"{name}/{split} should be reviewable, not huge"
            assert trace[0].arrival_s == 0.0
            assert all(a.arrival_s <= b.arrival_s for a, b in zip(trace, trace[1:]))
    assert "placeholder" in m.notes, "SLOs are placeholders pending the Phase 0 decision"


def test_committed_corpus_is_what_the_builder_produces(tmp_path):
    """Provenance: the data on disk came from build_corpus.py at the default seed, byte for
    byte. If prompt_bank.py or the class table changes, rebuild and commit a new version."""
    rebuilt = bc.build_corpus(tmp_path)
    committed = load_manifest(CORPUS_DIR)
    assert rebuilt.files == committed.files
    assert rebuilt.corpus_version == committed.corpus_version


def test_seen_and_heldout_prompts_are_disjoint():
    for name in CLASSES:
        _, seen = load_trace(name, "seen", CORPUS_DIR)
        _, held = load_trace(name, "heldout", CORPUS_DIR)
        assert not {r.prompt for r in seen} & {r.prompt for r in held}, name
        assert not {r.session_id for r in seen} & {r.session_id for r in held}, name


def test_every_prompt_has_a_unique_first_block_except_follow_up_turns():
    """The miss path is the default; only turn_index > 0 may share a prefix (with its own
    session's first turn), which is what makes multi-turn sessions meaningful."""
    _, trace = load_trace("steady_interactive", "seen", CORPUS_DIR)
    first = [r for r in trace if r.turn_index == 0]
    assert len({r.prompt[:32] for r in first}) == len(first)
    follow = [r for r in trace if r.turn_index > 0]
    assert follow, "steady_interactive must contain second turns"
    by_session = {r.session_id: r for r in first}
    for r in follow:
        assert r.prompt.startswith(by_session[r.session_id].prompt)
        assert r.arrival_s > by_session[r.session_id].arrival_s
    assert all(r.sampling["temperature"] == 0.0 for r in trace), "temperature 0 for equivalence"
    assert all(r.expected_output_hash is None for r in trace), "null until a reference run"


# ---------------------------------------------------------------- the rules

def test_tampered_trace_is_refused(tmp_path):
    bc.build_corpus(tmp_path)
    load_manifest(tmp_path)
    p = tmp_path / "cold_start" / "seen.jsonl"
    p.write_text(p.read_text().replace("Case ", "case ", 1))
    with pytest.raises(CorpusError, match="new corpus version"):
        load_manifest(tmp_path)
    with pytest.raises(CorpusError):
        load_trace("cold_start", "heldout", tmp_path)   # any bad file poisons the whole corpus


def test_manifest_version_must_match_its_hashes(tmp_path):
    bc.build_corpus(tmp_path)
    mp = tmp_path / "manifest.json"
    d = json.loads(mp.read_text())
    d["corpus_version"] = "0" * 64
    mp.write_text(json.dumps(d))
    with pytest.raises(CorpusError, match="corpus_version"):
        load_manifest(tmp_path)


def test_missing_trace_is_refused(tmp_path):
    bc.build_corpus(tmp_path)
    (tmp_path / "long_context" / "heldout.jsonl").unlink()
    with pytest.raises(CorpusError, match="missing"):
        load_manifest(tmp_path)


def test_build_is_deterministic_in_the_seed(tmp_path):
    a = bc.build_corpus(tmp_path / "a", seed=7)
    b = bc.build_corpus(tmp_path / "b", seed=7)
    c = bc.build_corpus(tmp_path / "c", seed=8)
    assert a.files == b.files and a.corpus_version == b.corpus_version
    assert a.corpus_version != c.corpus_version
    for rel in a.files:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


def test_trace_round_trip(tmp_path):
    reqs = [TraceRequest(0.0, "s-0", 0, "hello", 16),
            TraceRequest(1.5, "s-0", 1, "hello\n\nFollow-up: more", 16, expected_output_hash="ab")]
    write_trace(tmp_path / "t.jsonl", reqs)
    assert read_trace(tmp_path / "t.jsonl") == reqs


def test_class_slo_judgement():
    both = WorkloadClass("x", "", 200.0, 50.0, 4.0, "a", "b")
    assert both.within_slo(199.0, 49.0)
    assert not both.within_slo(200.0, 49.0)
    assert not both.within_slo(199.0, 50.0)
    assert not both.within_slo(199.0, None), "a TPOT ceiling with no TPOT measured is not met"
    ttft_only = WorkloadClass("y", "", 2000.0, None, 0.5, "a", "b")
    assert ttft_only.within_slo(1999.0, None)
    with pytest.raises(CorpusError):
        both.trace_file("test")
