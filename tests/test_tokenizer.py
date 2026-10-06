"""Tests for the tokenization pipeline."""

import pytest

from inference_server.tokenizer import Tokenizer


# Use GPT-2 tokenizer for tests — small, public, no auth required
TEST_MODEL = "gpt2"
CONTEXT_WINDOW = 100


@pytest.fixture
def tokenizer():
    return Tokenizer(TEST_MODEL, CONTEXT_WINDOW)


def test_encode_decode_roundtrip(tokenizer):
    text = "Tell me a joke"
    token_ids = tokenizer.encode(text)
    assert isinstance(token_ids, list)
    assert all(isinstance(t, int) for t in token_ids)
    decoded = tokenizer.decode(token_ids)
    assert "Tell me a joke" in decoded


def test_encode_empty_input(tokenizer):
    with pytest.raises(ValueError, match="empty"):
        tokenizer.encode("")


def test_encode_exceeds_context_window(tokenizer):
    long_text = "word " * 200  # way more than 100 tokens
    with pytest.raises(ValueError, match="exceeds context window"):
        tokenizer.encode(long_text)


def test_decode_single_token(tokenizer):
    token_ids = tokenizer.encode("hello")
    first_token = token_ids[0]
    result = tokenizer.decode_token(first_token)
    assert isinstance(result, str)
    assert len(result) > 0


def test_eos_token_id(tokenizer):
    assert isinstance(tokenizer.eos_token_id, int)


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size > 0


def test_context_window_explicit_wins_else_model_positions_capped(monkeypatch):
    from types import SimpleNamespace

    from transformers import AutoConfig

    from inference_server.config import resolve_context_window
    positions = {"gemma": 131072, "qwen": 40960, "small": 2048, "none": None}
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda name: SimpleNamespace(
        get_text_config=lambda: SimpleNamespace(max_position_embeddings=positions[name])))
    assert resolve_context_window("qwen", 4096) == 4096       # CONTEXT_WINDOW set: used as is
    assert resolve_context_window("gemma", 0) == 32768
    assert resolve_context_window("qwen", 0) == 32768
    assert resolve_context_window("small", 0) == 2048
    assert resolve_context_window("none", 0) == 8192
