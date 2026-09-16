"""Knowledge entries are queryable by regime and carry the bounds they were measured over.

The failure this guards: a finding measured at concurrency 8 on E4B/A100 being applied at
concurrency 64 on a different model because nothing recorded where it stopped being true.
`covers` is the primitive the brain and the controller use to ask "does this apply to me" —
and it must never answer "nearest".
"""

from __future__ import annotations

import argparse

import pytest

from inference_server.research.kb import (
    KNOWLEDGE_DIR,
    covers,
    generate_index,
    load_entries,
    query,
    save_entry,
)
from inference_server.research.loop import main, parse_situation
from inference_server.research.schemas import KnowledgeEntry, SchemaError

RANGE = {"model": "E4B", "hardware": "A100", "context_tokens": [128, 2048],
         "concurrency": [1, 16], "corpus_version": "v1"}


def _entry(**over):
    base = dict(title="t", summary="s", status="resolved", regime="steady_interactive",
                validity_range=RANGE, mechanism="decode was memory-bound so batching helped")
    base.update(over)
    return KnowledgeEntry(**base)


# ---------------------------------------------------------------- schema

def test_new_fields_round_trip(tmp_path):
    e = _entry(supersedes="kb-20260101-old", transfer_checked={"model": "E2B", "held": True})
    save_entry(e, tmp_path)
    back = load_entries(tmp_path)[0]
    assert back.regime == "steady_interactive"
    assert back.validity_range == RANGE
    assert back.mechanism == e.mechanism
    assert back.supersedes == "kb-20260101-old"
    assert back.transfer_checked == {"model": "E2B", "held": True}


def test_fields_default_empty_so_old_entries_load():
    e = KnowledgeEntry(title="t", summary="s")
    e.validate()
    assert (e.regime, e.validity_range, e.mechanism, e.supersedes, e.transfer_checked) \
        == (None, {}, None, None, None)


def test_unknown_regime_rejected():
    with pytest.raises(SchemaError, match="regime"):
        _entry(regime="steady_interactve").validate()   # a typo must not make a new bucket
    _entry(regime=None).validate()


@pytest.mark.parametrize("bad", [{"a": {"lo": 1}}, {"a": []}, {"a": [1, [2]]}, {"a": None}])
def test_malformed_validity_range_rejected(bad):
    with pytest.raises(SchemaError, match="validity_range"):
        _entry(validity_range=bad).validate()
    _entry(validity_range={"hardware": ["A100", "H100"], "compile": True}).validate()


def test_every_committed_entry_loads_and_validates_unchanged():
    entries = load_entries()
    assert len(entries) == len(list(KNOWLEDGE_DIR.glob("*.json")))
    for e in entries:
        e.validate()


@pytest.mark.parametrize("field", ["supersedes", "superseded_by"])
def test_supersession_refs_must_be_entry_ids(field):
    with pytest.raises(SchemaError, match=field):
        _entry(**{field: "exp-20260101-abc"}).validate()
    _entry(**{field: "kb-20260101-abc"}).validate()


# ---------------------------------------------------------------- covers

def test_covers_scalar_match_and_mismatch():
    assert covers(_entry(), {"model": "E4B"})
    assert not covers(_entry(), {"model": "E2B"})


def test_covers_range_inside_edges_and_outside():
    assert covers(_entry(), {"concurrency": 8})
    assert covers(_entry(), {"concurrency": 1}) and covers(_entry(), {"concurrency": 16})
    assert not covers(_entry(), {"concurrency": 17})   # no "nearest entry", ever
    assert not covers(_entry(), {"context_tokens": 4096})


def test_covers_absent_keys_are_unconstrained():
    assert covers(_entry(), {"gpu_count": 8})            # entry says nothing about it
    assert covers(_entry(validity_range={}), {"model": "anything"})
    assert covers(_entry(), {})


def test_covers_all_present_keys_must_hold():
    assert not covers(_entry(), {"model": "E4B", "concurrency": 64})


def test_covers_uncomparable_value_is_not_covered():
    assert not covers(_entry(), {"concurrency": "eight"})
    assert not covers(_entry(), {"concurrency": True})    # bool is not a number here


def test_covers_string_two_list_is_membership_not_range():
    e = _entry(validity_range={"hardware": ["A100", "H100"]})
    assert covers(e, {"hardware": "H100"})
    assert not covers(e, {"hardware": "B100"})   # lexicographically between, still not measured
    assert covers(_entry(validity_range={"model": ["E2B", "E4B", "E8B"]}), {"model": "E4B"})


# ---------------------------------------------------------------- query + CLI

def test_query_by_regime():
    items = [_entry(regime="cold_start", title="a"), _entry(title="b"), _entry(regime=None)]
    assert [e.title for e in query(items, regime="cold_start")] == ["a"]
    assert [e.title for e in query(items, regime="steady_interactive")] == ["b"]
    assert len(query(items)) == 3


def test_parse_situation_types_and_shape():
    assert parse_situation("model=E4B,concurrency=8,ratio=0.5") \
        == {"model": "E4B", "concurrency": 8, "ratio": 0.5}
    assert parse_situation(None) == {} and parse_situation("") == {}
    with pytest.raises(argparse.ArgumentTypeError):
        parse_situation("model=E4B,concurrency")
    with pytest.raises(SystemExit):
        main(["kb", "--situation", "bare"])


def test_cli_tags_unscoped_entries(tmp_path, monkeypatch, capsys):
    save_entry(_entry(title="scoped"), tmp_path)
    save_entry(_entry(title="no range", validity_range={}), tmp_path)
    monkeypatch.setattr("inference_server.research.kb.load_entries",
                        lambda directory=None: load_entries(tmp_path))
    main(["kb", "--situation", "model=E4B"])
    lines = capsys.readouterr().out.splitlines()
    assert any("no range" in ln and ln.endswith("[unscoped]") for ln in lines)
    assert any("scoped" in ln and "[unscoped]" not in ln for ln in lines)
    main(["kb"])
    assert "[unscoped]" not in capsys.readouterr().out    # only meaningful with --situation


def test_cli_regime_and_situation_filter(tmp_path, monkeypatch, capsys):
    save_entry(_entry(title="fits", regime="cold_start"), tmp_path)
    save_entry(_entry(title="wrong regime"), tmp_path)
    save_entry(_entry(title="too concurrent", regime="cold_start",
                      validity_range={"concurrency": [1, 4]}), tmp_path)
    monkeypatch.setattr("inference_server.research.kb.load_entries",
                        lambda directory=None: load_entries(tmp_path))
    assert main(["kb", "--regime", "cold_start", "--situation", "concurrency=8"]) == 0
    out = capsys.readouterr().out
    assert "fits" in out and "wrong regime" not in out and "too concurrent" not in out
    assert "1 entry" in out


# ---------------------------------------------------------------- generated view

def test_index_lists_regimes_and_new_fields():
    md = generate_index([_entry(title="Batched decode", id="kb-20260101-aaaaaaaa",
                                supersedes="kb-20260101-old",
                                transfer_checked={"model": "E2B", "held": True}),
                         _entry(title="No regime", regime=None, validity_range={},
                                mechanism=None)])
    assert "## By regime" in md
    assert "**steady_interactive** (1): `kb-20260101-aaaaaaaa`" in md
    assert "**unassigned** (1)" in md
    assert "**Regime:** `steady_interactive`" in md
    assert '**Valid over:** `{"concurrency": [1, 16]' in md
    assert "**Mechanism:** decode was memory-bound" in md
    assert "**Supersedes:** `kb-20260101-old`" in md
    assert '**Transfer checked:** `{"held": true, "model": "E2B"}`' in md
