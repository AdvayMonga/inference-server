"""Gemma 4 config reading across transformers layouts (5.17 made head_dim per-layer)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_server.models import gemma4
from inference_server.models.gemma4 import GemmaForCausalLM

# google/gemma-4-E2B-it's config.json, committed so this runs offline and in CI.
CONFIG_DIR = Path(__file__).parent / "fixtures" / "gemma-4-E2B-it"


class _Built(Exception):
    pass


def test_from_hf_builds_gemma_with_256_sliding_and_512_full_head_dim(monkeypatch):
    """from_hf must read the real config on the pinned transformers; stop before any weights load."""
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(gemma4, "GemmaModel", record)
    with pytest.raises(_Built):
        GemmaForCausalLM.from_hf(str(CONFIG_DIR))
    assert (seen["head_dim"], seen["global_head_dim"]) == (256, 512)
    assert seen["sliding_window"] == 512 and seen["num_kv_shared_layers"] == 20
    assert seen["layer_types"].count("full_attention") == 7


def test_head_dims_on_the_real_config():
    from transformers import AutoConfig

    from inference_server.models.gemma4 import head_dims
    assert head_dims(AutoConfig.from_pretrained(CONFIG_DIR).text_config) == (256, 512)


def test_head_dims_flat_layout():
    from inference_server.models.gemma4 import head_dims
    assert head_dims(SimpleNamespace(head_dim=256, global_head_dim=512)) == (256, 512)


def test_head_dims_per_layer_pairs_by_layer_type_not_index():
    from inference_server.models.gemma4 import head_dims

    cfg = SimpleNamespace(
        layer_types=["full_attention", "sliding_attention"],
        per_layer_config=[SimpleNamespace(head_dim=512), SimpleNamespace(head_dim=256)],
    )
    assert head_dims(cfg) == (256, 512)
