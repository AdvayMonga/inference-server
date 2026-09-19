"""Backend stop set: read from the model's generation config, the way HF generate() and vLLM do."""

import json
from types import SimpleNamespace

import pytest

from inference_server.backends.base import stop_token_ids

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


def test_falls_back_to_model_config(tmp_path):
    assert stop_token_ids(_repo(tmp_path, model=[1, 106]), TOKENIZER) == {1, 106}


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
