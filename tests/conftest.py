"""Shared test config — and the guard rails that keep the suite from eating the machine.

The full suite loads a 9.6 GB model repeatedly: test_gemma4_parity does 5 separate from_hf()
calls, test_sliding_window 2 more, and test_custom_backend_scheduler / test_prefill_batch each
hold their own module-scoped backend. On a 24 GB laptop that only survives because CPython frees
each one before the next allocates — there is no headroom, and two suite runs at once will take
the machine down.

So: model-heavy tests are marked `heavy` and DESELECTED by default. Run them deliberately:

    pytest                  # fast tests only, safe, seconds
    pytest -m heavy         # the model-loading ones, one process at a time
    pytest -m ''            # everything (what used to be the default)
"""

import pytest

# Any test module that constructs a real Gemma model or backend.
_HEAVY_MODULES = {
    "test_gemma4_parity",
    "test_sliding_window",
    "test_custom_backend_scheduler",
    "test_prefill_batch",
    "test_chunked_prefill",
    "test_scheduler_splice",
}


TINY_VOCAB, TINY_EOS = 1000, 997


def tiny_qwen3_moe_cfg(**over):
    from transformers import Qwen3MoeConfig
    kw = dict(vocab_size=TINY_VOCAB, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
              moe_intermediate_size=32, norm_topk_prob=True, tie_word_embeddings=False,
              rope_parameters={"rope_type": "default", "rope_theta": 1e6},
              max_position_embeddings=512, eos_token_id=TINY_EOS, pad_token_id=0, bos_token_id=1)
    kw.update(over)
    return Qwen3MoeConfig(**kw)


@pytest.fixture(scope="session")
def tiny_qwen3_moe_dir(tmp_path_factory):
    """Tiny HF Qwen3-MoE checkpoint + tokenizer on disk, loadable by from_pretrained and the backend."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3MoeForCausalLM as HF

    d = tmp_path_factory.mktemp("tiny_qwen3_moe")
    torch.manual_seed(0)
    HF._from_config(tiny_qwen3_moe_cfg(), dtype=torch.bfloat16).save_pretrained(d)
    vocab = {f"t{i}": i for i in range(TINY_VOCAB - 2)} | {"<think>": TINY_VOCAB - 2, "</think>": TINY_VOCAB - 1}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tk).save_pretrained(d)
    return str(d)


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: loads a real model (GBs of RAM); run alone")
    config.addinivalue_line(
        "markers", "needs_host: needs what the lab's jail forbids (held-out corpus, git, sockets); check skips it, CI runs it")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.module.__name__.rsplit(".", 1)[-1] in _HEAVY_MODULES:
            item.add_marker(pytest.mark.heavy)

    if config.getoption("-m"):
        return  # caller chose explicitly; respect it
    skip = pytest.mark.skip(reason="model-heavy; run with -m heavy (needs several GB, alone)")
    for item in items:
        if item.get_closest_marker("heavy"):
            item.add_marker(skip)
