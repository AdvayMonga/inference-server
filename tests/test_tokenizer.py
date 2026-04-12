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
