"""The corpus contract: frozen, hashed, deterministic, split. A changed trace is a new version.

Runs in the loop CI lane (pytest + ruff only): build_corpus.py imports only the stdlib at module
level, and the tests that rebuild from raw traces skip when the trace cache is absent.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from lab.corpus import (
    CORPUS_DIR,
    SPLITS,
    CorpusError,
    TraceRequest,
    WorkloadClass,
    build_manifest,
    corpus_version,
    load_manifest,
    load_trace,
    read_trace,
    write_trace,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "corpus"))
import build_corpus as bc  # noqa: E402

REGIMES = ("cold_start", "steady_interactive", "long_context", "spike")
MIXED_CLASSES = ("mixed_1", "mixed_2", "mixed_3")
CLASSES = REGIMES + MIXED_CLASSES
RAW = bc.DEFAULT_CACHE
# os.path.exists, not Path.exists: False rather than a raise when the jail denies the read
HAVE_RAW = all(os.path.exists(RAW / f)
               for f in (bc.BURSTGPT_CSV, *bc.BURSTGPT_EARLIER, bc.WILDCHAT_JSONL))
# CI's corpus lane sets CORPUS_REQUIRE_RAW=1: there a missing cache must fail, never skip.
REQUIRE_RAW = os.environ.get("CORPUS_REQUIRE_RAW") == "1"
needs_raw = pytest.mark.skipif(not HAVE_RAW and not REQUIRE_RAW, reason=f"raw traces absent "
                               f"from {RAW}; run scripts/corpus/fetch_traces.py")
BUDGET_FLOOR = 1024
CONTEXT = 32768             # Qwen3-30B-A3B native context: prompt + budget must fit


def _mini_corpus(d: Path):
    """A tiny hand-made corpus with the real manifest machinery, for the tamper rules."""
    classes = {}
    for name in ("cold_start", "long_context"):
        classes[name] = WorkloadClass(name, "", 1000.0, None, 1.0, f"{name}/seen.jsonl",
                                      f"{name}/heldout.jsonl")
        for split in SPLITS:
            write_trace(d / name / f"{split}.jsonl",
                        [TraceRequest(0.0, f"{name}-{split}-0", 0, f"Case {name} {split}", 64)])
    m = build_manifest(classes, d)
    (d / "manifest.json").write_text(json.dumps(m.to_dict(), indent=2) + "\n")
    return m


def _all(name):
    return [r for split in SPLITS for r in load_trace(name, split, CORPUS_DIR)[1]]


# ---------------------------------------------------------------- the committed corpus

@pytest.mark.needs_host
def test_committed_corpus_verifies_and_every_class_is_non_empty():
    m = load_manifest(CORPUS_DIR)
    assert set(m.classes) == set(CLASSES)
    for name in CLASSES:
        for split in SPLITS:
            _, trace = load_trace(name, split, CORPUS_DIR)
            hi = 1.1 * bc.MIXED_WINDOW_S if name in MIXED_CLASSES else 400
            assert 50 <= len(trace) <= hi, f"{name}/{split}: enough for a p95, still reviewable"
            assert trace[0].arrival_s == 0.0
            assert all(a.arrival_s <= b.arrival_s for a, b in zip(trace, trace[1:]))
    assert "pending owner confirmation" in m.notes, "SLOs are proposals until the owner decides"


@needs_raw
@pytest.mark.needs_host
def test_committed_corpus_is_what_the_builder_produces(tmp_path):
    """Provenance: the data on disk came from build_corpus.py at the default seed from the
    pinned raw inputs, byte for byte. A changed input or class table is a new version."""
    rebuilt = bc.build_corpus(tmp_path)
    committed = load_manifest(CORPUS_DIR)
    assert rebuilt.files == committed.files
    assert rebuilt.corpus_version == committed.corpus_version


@needs_raw
@pytest.mark.needs_host
def test_build_is_deterministic_in_the_seed(tmp_path):
    a = bc.build_corpus(tmp_path / "a", seed=7)
    b = bc.build_corpus(tmp_path / "b", seed=7)
    c = bc.build_corpus(tmp_path / "c", seed=8)
    assert a.files == b.files and a.corpus_version == b.corpus_version
    assert a.corpus_version != c.corpus_version
    for rel in a.files:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()


@pytest.mark.needs_host
def test_seen_and_heldout_are_disjoint():
    for name in CLASSES:
        _, seen = load_trace(name, "seen", CORPUS_DIR)
        _, held = load_trace(name, "heldout", CORPUS_DIR)
        assert not {r.prompt for r in seen} & {r.prompt for r in held}, name
        assert not {r.session_id for r in seen} & {r.session_id for r in held}, name
    every = [r for name in CLASSES for r in _all(name)]
    firsts = {r.session_id: r.chat_messages()[0]["content"] for r in every}
    assert len(set(firsts.values())) == len(firsts), "no conversation is used twice"


@pytest.mark.needs_host
def test_every_request_respects_the_floor_the_cap_and_the_context():
    """Checked on the committed bytes alone: build_prompt_tokens is the build tokenizer's count."""
    for name in CLASSES:
        for r in _all(name):
            n = r.build_prompt_tokens
            assert n is not None and bc.MIN_TARGET <= n <= bc.PROMPT_CAP, (name, r.session_id, n)
            assert r.max_tokens >= BUDGET_FLOOR, (name, r.session_id)
            assert n + r.max_tokens <= CONTEXT, (name, r.session_id)


@pytest.mark.needs_host
def test_only_mixed_requests_carry_a_regime_and_it_names_a_class():
    for name in CLASSES:
        for r in _all(name):
            assert (r.regime in REGIMES) if name in MIXED_CLASSES else r.regime is None, (name, r.regime)
    m = load_manifest(CORPUS_DIR)
    assert all(m.classes[n].slo_ttft_ms is None for n in MIXED_CLASSES), "judged per request"


@pytest.mark.needs_host
def test_every_trace_records_its_dilation():
    m = load_manifest(CORPUS_DIR)
    assert set(m.dilation) == set(m.files)
    assert all(d >= 1.0 for d in m.dilation.values())


@pytest.mark.needs_host
def test_no_committed_corpus_file_contains_a_credential():
    """GitHub push protection caught a real key in a WildChat chat; never again."""
    hits = [(str(f.relative_to(CORPUS_DIR)), m.group()[:12])
            for f in sorted(CORPUS_DIR.rglob("*")) if f.is_file()
            for m in bc.SECRET_RE.finditer(f.read_text(errors="replace"))]
    assert hits == []


@pytest.mark.needs_host
def test_no_oracle_is_set():
    for name in CLASSES:
        for r in _all(name):
            assert r.sampling["temperature"] == 0.0, "temperature 0 for equivalence"
            assert r.expected_output_hash is None and r.expected_output_tokens is None


@pytest.mark.needs_host
def test_multi_turn_requests_carry_real_assistant_turns():
    n_multi = 0
    for name in CLASSES:
        trace = _all(name)
        by_session: dict[str, list[TraceRequest]] = {}
        for r in trace:
            by_session.setdefault(r.session_id, []).append(r)
            if r.messages is None:
                assert r.turn_index == 0
                continue
            n_multi += 1
            roles = [m["role"] for m in r.messages]
            assert roles == ["user", "assistant"] * r.turn_index + ["user"]
            assert all(m["content"].strip() for m in r.messages)
            assert r.prompt == "\n\n".join(m["content"] for m in r.messages)
        for turns in by_session.values():     # a returning session extends its own history
            for a, b in zip(turns, turns[1:]):
                assert b.turn_index > a.turn_index and b.arrival_s >= a.arrival_s
                assert b.chat_messages()[:len(a.chat_messages())] == a.chat_messages()
    assert n_multi, "the corpus must contain multi-turn requests"


# ---------------------------------------------------------------- the rules

def test_tampered_trace_is_refused(tmp_path):
    _mini_corpus(tmp_path)
    load_manifest(tmp_path)
    p = tmp_path / "cold_start" / "seen.jsonl"
    p.write_text(p.read_text().replace("Case ", "case ", 1))
    with pytest.raises(CorpusError, match="new corpus version"):
        load_manifest(tmp_path)
    with pytest.raises(CorpusError):
        load_trace("cold_start", "heldout", tmp_path)   # any bad file poisons the whole corpus


def test_manifest_version_must_match_its_hashes(tmp_path):
    _mini_corpus(tmp_path)
    mp = tmp_path / "manifest.json"
    d = json.loads(mp.read_text())
    d["corpus_version"] = "0" * 64
    mp.write_text(json.dumps(d))
    with pytest.raises(CorpusError, match="corpus_version"):
        load_manifest(tmp_path)


def test_changed_class_slo_is_a_new_version(tmp_path):
    """The placeholder SLOs will be decided later; that decision must move the version, or
    pre- and post-SLO panels would compare as equals."""
    m = _mini_corpus(tmp_path)
    mp = tmp_path / "manifest.json"
    d = json.loads(mp.read_text())
    d["classes"]["cold_start"]["slo_ttft_ms"] = 150.0
    mp.write_text(json.dumps(d))
    with pytest.raises(CorpusError, match="class table"):
        load_manifest(tmp_path)
    tweaked = {k: WorkloadClass(**v) for k, v in d["classes"].items()}
    assert corpus_version(m.files, tweaked) != m.corpus_version
    assert corpus_version(m.files, m.classes) == m.corpus_version


def test_missing_trace_is_refused(tmp_path):
    _mini_corpus(tmp_path)
    (tmp_path / "long_context" / "heldout.jsonl").unlink()
    with pytest.raises(CorpusError, match="missing"):
        load_manifest(tmp_path)


def test_trace_round_trip(tmp_path):
    reqs = [TraceRequest(0.0, "s-0", 0, "hello", 16),
            TraceRequest(1.5, "s-0", 1, "hello\n\nhi!\n\nmore", 16, expected_output_hash="ab",
                         messages=[{"role": "user", "content": "hello"},
                                   {"role": "assistant", "content": "hi!"},
                                   {"role": "user", "content": "more"}]),
            TraceRequest(2.0, "s-1", 0, "bye", 16, expected_output_tokens=9)]
    write_trace(tmp_path / "t.jsonl", reqs)
    assert read_trace(tmp_path / "t.jsonl") == reqs


def test_unset_expected_output_tokens_is_not_serialised(tmp_path):
    """The field must be invisible until something measures it, or writing any trace would move
    the corpus_version and orphan every panel already measured against it."""
    plain = TraceRequest(0.0, "s-0", 0, "hello", 16)
    assert "expected_output_tokens" not in plain.to_dict()
    assert "expected_output_hash" in plain.to_dict(), "the older field still writes its null"
    write_trace(tmp_path / "t.jsonl", [plain])
    line = json.loads((tmp_path / "t.jsonl").read_text())
    assert set(line) == {"arrival_s", "expected_output_hash", "max_tokens", "prompt",
                         "sampling", "session_id", "turn_index"}
    assert read_trace(tmp_path / "t.jsonl")[0].expected_output_tokens is None


def test_regime_is_written_only_when_set(tmp_path):
    write_trace(tmp_path / "t.jsonl", [TraceRequest(0.0, "s", 0, "hi", 16),
                                       TraceRequest(1.0, "s", 0, "hi", 16, regime="spike")])
    a, b = [json.loads(x) for x in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "regime" not in a and b["regime"] == "spike"
    assert read_trace(tmp_path / "t.jsonl")[1].regime == "spike"


def test_a_trace_that_carries_the_field_writes_and_reads_it(tmp_path):
    write_trace(tmp_path / "t.jsonl", [TraceRequest(0.0, "s", 0, "hi", 16,
                                                    expected_output_tokens=3)])
    assert json.loads((tmp_path / "t.jsonl").read_text())["expected_output_tokens"] == 3


def test_class_slo_judgement():
    both = WorkloadClass("x", "", 200.0, 50.0, 4.0, "a", "b")
    assert both.within_slo(199.0, 49.0)
    assert not both.within_slo(200.0, 49.0)
    assert not both.within_slo(199.0, 50.0)
    assert not both.within_slo(199.0, None), "a TPOT ceiling with no TPOT measured is not met"
    ttft_only = WorkloadClass("y", "", 2000.0, None, 0.5, "a", "b")
    assert ttft_only.within_slo(1999.0, None)
    with pytest.raises(CorpusError, match="regime"):
        WorkloadClass("m", "", None, None, 1.1, "a", "b").within_slo(1.0, None)
    with pytest.raises(CorpusError):
        both.trace_file("test")


def test_dilation_is_hashed_and_old_manifests_without_it_still_load(tmp_path):
    m = _mini_corpus(tmp_path)
    assert "dilation" not in json.loads((tmp_path / "manifest.json").read_text())
    assert load_manifest(tmp_path).dilation == {}
    with_dil = corpus_version(m.files, m.classes, {"cold_start/seen.jsonl": 2.0})
    assert with_dil != m.corpus_version
    mp = tmp_path / "manifest.json"
    d = json.loads(mp.read_text())
    d["dilation"] = {"cold_start/seen.jsonl": 2.0}
    mp.write_text(json.dumps(d))
    with pytest.raises(CorpusError, match="corpus_version"):
        load_manifest(tmp_path)
    d["corpus_version"], d["dilation"] = with_dil, {"nope.jsonl": 2.0}
    mp.write_text(json.dumps(d))
    with pytest.raises(CorpusError, match="dilation"):
        load_manifest(tmp_path)


# ---------------------------------------------------------------- builder logic, synthetic

A = bc.Arrival
SPEC = {s.cls.name: s for s in bc._specs()}
MIXED = [40, 900, 150, 3000, 75, 400, 1200, 60, 2200, 300]     # CV well over 0.3


def test_varied_rejects_a_scripted_client_and_keeps_a_mix():
    assert not bc.varied([A(i, "", 360 + i % 20) for i in range(60)])
    assert not bc.varied([A(i, "", 20 + i % 5) for i in range(60)])
    assert bc.varied([A(i, "", MIXED[i % 10]) for i in range(60)])


def test_cold_start_window_follows_an_idle_gap_and_may_end_the_trace():
    head = [A(i * 10.0, "", 100) for i in range(5)]
    tail = [A(1000.0 + i * 5, "", MIXED[i % 10]) for i in range(60)]      # gap 960 s, 295 s
    wins = list(bc._windows_cold_start(head + tail, SPEC["cold_start"]))
    assert wins == [tail], "the last valid start is enumerated"
    scripted = [A(1000.0 + i * 5, "", 300) for i in range(60)]
    assert not list(bc._windows_cold_start(head + scripted, SPEC["cold_start"]))
    slow = [A(1000.0 + i * 30, "", MIXED[i % 10]) for i in range(60)]     # 1770 s > 20 min
    assert not list(bc._windows_cold_start(head + slow, SPEC["cold_start"]))


def test_long_context_keeps_only_long_requests_and_reaches_the_end():
    arr = []
    for i in range(60):
        arr += [A(i * 20.0, "", 4000), A(i * 20.0 + 1, "", 50)]
    wins = list(bc._windows_long(arr, SPEC["long_context"]))
    assert len(wins) == 1 and len(wins[0]) == 60
    assert all(a.tokens >= 3000 for a in wins[0])


def test_steady_window_is_busy_and_even():
    even = [A(i * 2.4, "", 100) for i in range(250)]                     # 250 in 600 s
    assert len(list(bc._windows_steady(even + [A(1200.0, "", 1)], SPEC["steady_interactive"]))) == 1
    lumpy = [A(i * 0.24, "", 100) for i in range(125)] + [A(300 + i * 2.4, "", 100)
                                                          for i in range(125)]
    assert not list(bc._windows_steady(lumpy + [A(1200.0, "", 1)], SPEC["steady_interactive"]))


def test_spike_window_is_a_burst_after_quiet_minutes():
    quiet = [A(m * 60.0 + 30, "", 100) for m in range(10)]              # 1/min for 10 min
    burst = [A(600.0 + i * 0.5, "", 100) for i in range(100)]            # 100 in minute 10
    tail = [A(660.0 + m * 60 + 30, "", 100) for m in range(3)]
    wins = list(bc._windows_spike(quiet + burst + tail, SPEC["spike"]))
    assert len(wins) == 1 and len(wins[0]) == 5 + 100 + 2


def test_pick_window_stays_inside_one_week_of_its_split():
    import random
    w0 = [A(10.0 + i, "", MIXED[i % 10]) for i in range(60)]
    w1 = [A(bc.WEEK_S + 10.0 + i, "", MIXED[i % 10]) for i in range(60)]
    arr = [A(0.0, "", 1)] + w0 + [A(bc.WEEK_S - 1000.0, "", 1)] + w1
    spec = SPEC["cold_start"]
    assert bc.pick_window(arr, spec, "heldout", random.Random(0)) == w1
    with pytest.raises(SystemExit):
        bc.pick_window(arr[:62], spec, "heldout", random.Random(0))


def _mixed_trace(per_min=7, spike=50, tokens=None):
    """Quiet, then 20 busy minutes from t=1200: post-idle requests, steady ones, a burst at
    minute 35, and a late request so a window can start at 1200."""
    tokens = tokens or (lambda i: MIXED[i % 10])
    arr, i = [A(0.0, "", 100)], 0
    for m in range(20, 40):
        n = spike if m == 35 else per_min
        for k in range(n):
            arr.append(A(m * 60.0 + k * 59.0 / n, "", tokens(i)))
            i += 1
    return arr + [A(3000.0, "", 100)]


def test_regimes_label_by_context_and_long_wins():
    arr = _mixed_trace()
    arr[5] = A(arr[5].t, "", 4000)
    lab = bc.regimes(arr)
    assert lab[5] == "long_context" and lab[1] == "cold_start" and lab[60] == "cold_start"
    assert lab[61] == "steady_interactive" and lab[0] == "steady_interactive"
    in_burst = {r for a, r in zip(arr, lab) if int(a.t // 60) == 35 and a.tokens < 3000}
    assert in_burst == {"spike"}


def test_mixed_window_needs_every_regime_spread_out_and_one_replica():
    spec = SPEC["mixed_1"]
    wins = list(bc._windows_mixed(_mixed_trace(), spec))
    assert wins and all(a.regime for w in wins for a in w)
    assert any(w[0].t == 1200.0 and len(w) == 19 * 7 + 50 for w in wins)
    assert not list(bc._windows_mixed(_mixed_trace(tokens=lambda i: 300), spec)), "scripted"
    assert not list(bc._windows_mixed(_mixed_trace(spike=200), spec)), "peak over one replica"
    assert not list(bc._windows_mixed(_mixed_trace(per_min=3), spec)), "too thin to mix"


def test_pick_window_avoids_days_already_taken():
    import random
    spec = SPEC["mixed_1"]
    arr = _mixed_trace()
    assert bc.pick_window(arr, spec, "seen", random.Random(0))
    with pytest.raises(SystemExit):
        bc.pick_window(arr, spec, "seen", random.Random(0), frozenset({0}))


def test_dilation_stretches_only_windows_faster_than_target():
    fast = [A(i * 0.5, "", 1) for i in range(61)]                         # 2 rps vs 0.5
    assert bc.dilation(fast, SPEC["cold_start"]) == 4.0
    assert bc.dilation([A(i * 10.0, "", 1) for i in range(61)], SPEC["cold_start"]) == 1.0
    peak = [A(i * 0.2, "", 1) for i in range(300)]                        # 300 in a minute
    assert bc.dilation(peak, SPEC["spike"]) == 2.0                        # 5 rps vs 2.5


def test_assign_turns_is_optimal_strictly_increasing_and_skips_short_turns():
    lens = [10, 100, 220, 400, 900]
    cost, turns = bc.assign_turns(lens, [100, 1000])
    assert turns == [1, 4]
    cost, turns = bc.assign_turns(lens, [12, 14])                        # 10 is below the floor
    assert turns == [1, 2] and turns[0] < turns[1]
    assert bc.assign_turns([5, 8], [10, 10])[0] == float("inf")


def test_pair_single_picks_the_nearest_unused_turn():
    index = sorted([(100, 0, 0), (110, 1, 0), (300, 1, 1), (1000, 2, 0)])
    assert bc.pair_single(index, 103, set()) == (0, 0)
    assert bc.pair_single(index, 103, {0}) == (1, 0)
    assert bc.pair_single(index, 280, {0}) == (1, 1)
    assert bc.pair_single(index, 5, set()) == (0, 0), "target floored at MIN_TARGET"


def _pool():
    def conv(tag, n):
        m = []
        for k in range(n):
            m += [{"role": "user", "content": f"{tag} q{k}"},
                  {"role": "assistant", "content": f"{tag} a{k}"}]
        return m
    convs = [conv("c0", 1), conv("c1", 4), conv("c2", 3)]
    lens = [[50], [40, 200, 600, 1500], [30, 300, 900]]
    pool = bc.Pool(convs, lens)
    pool.index = sorted((n, c, k) for c, ls in enumerate(lens) for k, n in enumerate(ls))
    return pool


def test_build_trace_pairs_sessions_with_real_growing_conversations():
    import random
    window = [A(100.0, "s", 200), A(101.0, "", 50), A(110.0, "s", 600), A(130.0, "s", 1500)]
    trace, lens = bc.build_trace(window, SPEC["steady_interactive"], "seen", 2.0, _pool(),
                                 set(), random.Random(0))
    assert [r.arrival_s for r in trace] == [0.0, 2.0, 20.0, 60.0], "offsets x dilation"
    sess = [r for r in trace if r.session_id.endswith("0000")]
    assert [r.turn_index for r in sess] == [1, 2, 3]
    for a, b in zip(sess, sess[1:]):
        assert b.messages[:len(a.messages)] == a.messages
    assert all(m["content"].startswith("c1") for m in sess[-1].messages)
    single = [r for r in trace if not r.session_id.endswith("0000")][0]
    assert single.messages is None and single.prompt == "c0 q0" and single.turn_index == 0
    assert [r.build_prompt_tokens for r in trace] == lens == [200, 50, 600, 1500]
    assert all(r.max_tokens == bc.MAX_TOKENS for r in trace)


def test_keep_rejects_a_conversation_that_contains_a_key():
    def row(text):
        return {"language": "English", "toxic": False, "flagged": False, "redacted": False,
                "messages": [{"role": "user", "content": "my code fails"},
                             {"role": "assistant", "content": text}]}
    assert bc._keep(row("Try checking the import path."))
    for secret in ("openai.api_key = 'sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2'",
                   "sk-" + "X" * 40, "AKIA" + "ABCDEFGHIJKLMNOP", "ghp_" + "a" * 36,
                   "-----BEGIN RSA " + "PRIVATE KEY-----", "hf_" + "b" * 34,
                   "M" + "T" * 25 + "." + "a" * 6 + "." + "b" * 30):
        assert not bc._keep(row(secret)), secret
