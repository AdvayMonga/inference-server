"""A suspension is machine-readable, scoped, and enforced by `loop screen`.

The failure this guards, which was live on 2026-09-19: `kb-20260919-94acfdb8` recorded that
tier-1 `ttft_p95` screening cannot be trusted (the simulator ranks it at rho 0.644 against a
0.683 critical value, because it has no termination model) — in PROSE, on an entry whose status
is `open`. `screen` only ever blocked on `rejected`/`resolved`, so it would happily greenlight a
hypothesis the knowledge base already knew it could not falsify. That is precisely the
re-treading `screen` exists to prevent.
"""

from __future__ import annotations

import json

import pytest

from inference_server.research.kb import (
    generate_index,
    load_entries,
    save_entry,
    suspends,
    suspensions,
)
from inference_server.research.loop import main
from inference_server.research.schemas import KnowledgeEntry, SchemaError

SIM = dict(title="Simulator fails the p95 TTFT rank check", summary="rho 0.644 < 0.683",
           status="open", regime="cold_start", suspended_metrics=["ttft_p95"],
           suspended_tiers=[1], validity_range={"model": "E2B", "hardware": "Apple M4 Pro (MPS)"})


def _hyp(**over):
    h = dict(statement="a wider prefix cache cuts the p95 first-token tail", gap_id="ttft",
             predicted_metric="ttft_p95", predicted_direction="decrease",
             predicted_magnitude=">20%", falsification_tier=1,
             falsification_test="replay cold_start through the simulator", tags=["cache"])
    h.update(over)
    return h


def _screen(tmp_path, monkeypatch, entries, hyps):
    kb_dir = tmp_path / "knowledge"
    for e in entries:
        save_entry(e, kb_dir)
    for mod in ("kb", "loop"):
        monkeypatch.setattr(f"inference_server.research.{mod}.load_entries",
                            lambda directory=None: load_entries(kb_dir), raising=False)
    p = tmp_path / "h.json"
    p.write_text(json.dumps(hyps))
    return main(["screen", str(p)])


# ---------------------------------------------------------------- schema

def test_defaults_empty_so_old_entries_load():
    e = KnowledgeEntry(title="t", summary="s")
    e.validate()
    assert (e.suspended_metrics, e.suspended_tiers) == ([], [])


def test_suspended_metric_must_be_a_panel_field():
    with pytest.raises(SchemaError, match="not a panel field"):
        KnowledgeEntry(**{**SIM, "suspended_metrics": ["ttft_p95_typo"]}).validate()


def test_tier_must_be_in_range_and_cannot_stand_alone():
    with pytest.raises(SchemaError, match="1..4"):
        KnowledgeEntry(**{**SIM, "suspended_tiers": [5]}).validate()
    with pytest.raises(SchemaError, match="suspends nothing"):
        KnowledgeEntry(**{**SIM, "suspended_metrics": [], "suspended_tiers": [1]}).validate()


def test_round_trips(tmp_path):
    save_entry(KnowledgeEntry(**SIM), tmp_path)
    back = load_entries(tmp_path)[0]
    assert back.suspended_metrics == ["ttft_p95"] and back.suspended_tiers == [1]


# ---------------------------------------------------------------- matching and lifting

def test_suspends_matches_metric_and_tier():
    e = KnowledgeEntry(**SIM)
    assert suspends(e, "ttft_p95", 1)
    assert not suspends(e, "ttft_p95", 3)          # a GPU probe still measures it fine
    assert not suspends(e, "tpot_p50", 1)          # rho 0.811 — that metric still ranks
    assert suspends(KnowledgeEntry(**{**SIM, "suspended_tiers": []}), "ttft_p95", 4)


def test_a_suspension_is_lifted_by_superseding_not_by_editing():
    live = KnowledgeEntry(**SIM)
    lifted = KnowledgeEntry(**{**SIM, "superseded_by": "kb-20260101-newer"})
    filed_away = KnowledgeEntry(**{**SIM, "status": "obsolete"})
    assert [e.id for e in suspensions([live, lifted, filed_away])] == [live.id]


# ---------------------------------------------------------------- screen

def test_screen_blocks_a_hypothesis_that_targets_a_suspended_metric(tmp_path, monkeypatch,
                                                                    capsys):
    rc = _screen(tmp_path, monkeypatch, [KnowledgeEntry(**SIM)], [_hyp()])
    out = capsys.readouterr().out
    assert rc == 1
    assert "SUSPENDED" in out and "ttft_p95" in out
    assert "hardware=Apple M4 Pro (MPS)" in out           # the scope, so it is not over-applied
    # Nothing should point a suspended tier-1 hypothesis at the instrument that is suspended.
    assert "loop simulate" not in out


def test_screen_finds_a_suspension_the_relevance_search_would_miss(tmp_path, monkeypatch,
                                                                   capsys):
    """A suspension is a property of the instrument, not of the subject."""
    rc = _screen(tmp_path, monkeypatch, [KnowledgeEntry(**SIM)],
                 [_hyp(statement="bigger blocks lower fragmentation", tags=["kv"])])
    assert rc == 1 and "SUSPENDED" in capsys.readouterr().out


def test_lifting_the_suspension_unblocks_the_same_hypothesis(tmp_path, monkeypatch, capsys):
    entries = [KnowledgeEntry(**{**SIM, "id": "kb-20260919-94acfdb8",
                                 "superseded_by": "kb-20260920-repaired"}),
               KnowledgeEntry(title="Simulator ranks p95 TTFT again", summary="rho 0.79",
                              status="resolved", id="kb-20260920-repaired",
                              supersedes="kb-20260919-94acfdb8")]
    rc = _screen(tmp_path, monkeypatch, entries, [_hyp()])
    assert rc == 0 and "SUSPENDED" not in capsys.readouterr().out


def test_screen_does_not_block_an_unrelated_metric_or_tier(tmp_path, monkeypatch, capsys):
    rc = _screen(tmp_path, monkeypatch, [KnowledgeEntry(**SIM)],
                 [_hyp(predicted_metric="tpot_p50"), _hyp(falsification_tier=3)])
    assert rc == 0 and "SUSPENDED" not in capsys.readouterr().out


# ---------------------------------------------------------------- kb and the generated view

def test_kb_lists_live_suspensions_only(tmp_path, monkeypatch, capsys):
    save_entry(KnowledgeEntry(**SIM), tmp_path)
    save_entry(KnowledgeEntry(**{**SIM, "title": "older, lifted",
                                 "superseded_by": "kb-20260101-newer"}), tmp_path)
    monkeypatch.setattr("inference_server.research.kb.load_entries",
                        lambda directory=None: load_entries(tmp_path))
    assert main(["kb", "--suspended"]) == 0
    out = capsys.readouterr().out
    assert "1 entry" in out and "older, lifted" not in out
    assert "SUSPENDS ttft_p95 at tier(s) 1" in out


def test_index_renders_active_suspensions():
    md = generate_index([KnowledgeEntry(**{**SIM, "id": "kb-20260919-94acfdb8"}),
                         KnowledgeEntry(**{**SIM, "id": "kb-20260917-c07eb94b",
                                           "title": "older", "status": "resolved",
                                           "superseded_by": "kb-20260919-94acfdb8"})])
    assert "## Active suspensions" in md
    assert "`ttft_p95` at tier 1" in md and "kb-20260919-94acfdb8" in md
    assert md.count("**Suspends:**") == 2                # both entries carry the field
    assert "— LIVE" in md and "— lifted (superseded)" in md
    assert "kb-20260917-c07eb94b`: older" not in md      # the lifted one is not in the section


def test_the_live_suspension_is_filed_in_machine_readable_form():
    """The 2026-09-19 entry, in the real knowledge base, not a fixture."""
    live = {e.id: e for e in suspensions()}
    e = live.get("kb-20260919-94acfdb8")
    assert e is not None, "the tier-1 ttft_p95 suspension is prose again, so nothing enforces it"
    assert e.suspended_metrics == ["ttft_p95"] and e.suspended_tiers == [1]
