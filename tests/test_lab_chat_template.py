"""The chat-template fingerprint: the variable server-side templating introduces.

No transformers here: the module's tokenizer import is lazy. A fake tokenizer is seeded into
the cache, which is also what proves the "no tokenizer -> None, never an exception" contract.
"""

from __future__ import annotations

import hashlib

import pytest

from lab import chat_template as CT

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


def test_an_unreadable_tokenizer_is_None_not_an_exception():
    """An offline box or a model with no chat template must not crash a replay."""
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
