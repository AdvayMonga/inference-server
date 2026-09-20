"""Backend stop set: read from the model's generation config, the way HF generate() and vLLM do."""

import json
import logging
from types import SimpleNamespace

import pytest

from inference_server.backends.base import stop_token_ids

LOGGER = "inference_server.backends.base"

TOKENIZER = SimpleNamespace(eos_token_id=1)  # gemma-4-*-it's tokenizer knows only <eos>


def _repo(tmp_path, generation=None, model=None):
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": generation}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2", "eos_token_id": model}))
    return str(tmp_path)


def test_stop_set_includes_end_of_turn(tmp_path):
    """gemma-4-E2B-it's own eos fields: an answer ends on <turn|> (106), which the tokenizer omits."""
    repo = _repo(tmp_path, generation=[1, 106, 50], model=[1, 106])
    assert stop_token_ids(repo, TOKENIZER) == {1, 106, 50}


def test_preferred_source_does_not_warn(tmp_path, caplog):
    repo = _repo(tmp_path, generation=[1, 106, 50], model=[1, 106])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        stop_token_ids(repo, TOKENIZER)
    assert caplog.records == []


def test_falls_back_to_model_config(tmp_path, caplog):
    """Repo genuinely has no generation_config.json: fall back, and say so."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert stop_token_ids(_repo(tmp_path, model=[1, 106]), TOKENIZER) == {1, 106}
    assert "model config" in caplog.text and "generation_config.json" in caplog.text


def test_unreachable_repo_falls_back_and_names_the_cause(tmp_path, monkeypatch, caplog):
    """A network/offline failure is an OSError too, so it must not pass as a missing file."""
    from transformers import GenerationConfig

    from huggingface_hub.errors import LocalEntryNotFoundError

    assert issubclass(LocalEntryNotFoundError, OSError)  # why the old fallback was silent

    def unreachable(*args, **kwargs):
        raise LocalEntryNotFoundError("cannot reach the hub and nothing is cached")

    monkeypatch.setattr(GenerationConfig, "from_pretrained", unreachable)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert stop_token_ids(_repo(tmp_path, model=[1, 106]), TOKENIZER) == {1, 106}
    assert "LocalEntryNotFoundError" in caplog.text
    assert "cannot reach the hub" in caplog.text


def test_falls_back_to_tokenizer(tmp_path):
    assert stop_token_ids(_repo(tmp_path, model=None), TOKENIZER) == {1}


def test_real_gemma_generation_config():
    """Against the cached repo itself, so a changed upstream config shows up here."""
    from transformers import GenerationConfig

    try:
        GenerationConfig.from_pretrained("google/gemma-4-E2B-it", local_files_only=True)
    except OSError:
        pytest.skip("gemma-4-E2B-it generation config not in the local HF cache")
    assert 106 in stop_token_ids("google/gemma-4-E2B-it", TOKENIZER)
