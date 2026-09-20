"""The chat-template fingerprint: the variable server-side templating introduces.

No transformers here — the loop CI lane installs pytest and ruff and nothing else, and the
module's tokenizer import is lazy for exactly that reason. A fake tokenizer is seeded into the
cache, which is also what proves the "no tokenizer -> None, never an exception" contract.
"""

from __future__ import annotations

import hashlib

import pytest

from inference_server.research import chat_template as CT
from inference_server.research.compare import comparable
from inference_server.research.schemas import Validity, Vitals

MODEL = "fake/instruct-model"
OVERHEAD = 7


class FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, *, enable_thinking=True, **kw):
        n = len(messages[0]["content"]) + OVERHEAD + (2 if enable_thinking else 0)
        return {"input_ids": list(range(n))}


@pytest.fixture(autouse=True)
def _clean_cache():
    CT._CACHE.clear()
    yield
    CT._CACHE.clear()


def _seed(tokenizer=None) -> None:
    CT._CACHE[MODEL] = FakeTokenizer() if tokenizer is None else tokenizer


def test_fingerprint_identifies_the_template_and_the_options_it_was_applied_with():
    _seed()
    fp = CT.fingerprint(MODEL)
    assert fp["tokenizer"] == MODEL
    assert fp["chat_template_sha256"] == \
        hashlib.sha256(FakeTokenizer.chat_template.encode()).hexdigest()
    assert fp["enable_thinking"] is False        # what the shim passes, recorded not assumed
    assert fp["probe_tokens"] == len("probe") + OVERHEAD
    assert fp["verified"] is None                # nothing has been checked yet


def test_enable_thinking_moves_the_fingerprint_because_it_moves_the_tokens():
    """On gemma-4 it injects a <|think|> system turn, which is the difference between a corpus
    that terminates early and one where every request runs to max_tokens."""
    _seed()
    a, b = CT.fingerprint(MODEL, enable_thinking=True), CT.fingerprint(MODEL)
    assert a["chat_template_sha256"] == b["chat_template_sha256"]   # same template string
    assert a["probe_tokens"] != b["probe_tokens"]                   # different rendered tokens
    assert not comparable(_panel(a), _panel(b), same_group_required=False)


def test_an_unreadable_tokenizer_is_None_not_an_exception():
    """A CPU-only lane, an offline box or a model with no chat template must not crash a
    replay — the same contract as harness.device_state_from_env()."""
    CT._CACHE[MODEL] = None
    assert CT.fingerprint(MODEL) is None
    assert CT.rendered_len(MODEL, "hi") is None

    class NoTemplate:
        chat_template = None
    _seed(NoTemplate())
    assert CT.fingerprint(MODEL) is None


def test_verify_compares_the_clients_rendering_with_what_the_server_encoded():
    _seed()
    fp = CT.fingerprint(MODEL)
    assert CT.verify(fp, "hello", len("hello") + OVERHEAD)["verified"] is True

    fp = CT.fingerprint(MODEL)
    bad = CT.verify(fp, "hello", 999)
    assert bad["verified"] is False and "999" in bad["verify_detail"]


def test_verify_leaves_None_when_either_side_is_unknown():
    """"not checked" and "checked and wrong" are different facts; only the second invalidates."""
    _seed()
    assert CT.verify(CT.fingerprint(MODEL), "hello", None)["verified"] is None
    assert CT.verify(None, "hello", 12) is None


def _panel(fp) -> Vitals:
    return Vitals(validity=Validity(
        engine_sha="abc", dirty=False, harness="replay_trace",
        harness_config={"prompt_format": "chat"}, workload_regime="synthetic", n_samples=3,
        run_group="grp-1", chat_template=fp))


def test_compare_ignores_the_fingerprint_when_only_one_panel_has_one():
    """Same rule as corpus_version: a pre-templating panel carries None and refuses nothing
    on this key alone."""
    _seed()
    fp = CT.fingerprint(MODEL)
    assert comparable(_panel(fp), _panel(None), same_group_required=False)
    assert comparable(_panel(fp), _panel(dict(fp)), same_group_required=False)


def test_a_revised_template_refuses_the_comparison_rather_than_drifting_silently():
    _seed()
    old = CT.fingerprint(MODEL)

    class Revised(FakeTokenizer):
        chat_template = "{{ messages }}{# revised #}"
    CT._CACHE.clear()
    _seed(Revised())
    new = CT.fingerprint(MODEL)

    cmp_ = comparable(_panel(old), _panel(new), same_group_required=False)
    assert not cmp_ and any("chat_template_sha256" in r for r in cmp_.reasons)
