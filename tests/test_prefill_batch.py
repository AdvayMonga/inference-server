"""prefill_batch (one padded forward over K prompts) matches per-row prefill.

K=1 is the strong check: same [1,L] forward, no padding → byte-identical first token AND identical
decode from the distributed paged cache (proves the per-row KV slice/append is correct). K>1 checks
the padded batching produces the same first tokens (allowing rare near-tie flips from batch
reduction-order noise — same caveat as the Triton kernel).
"""

import pytest
import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.paged_kv_cache import PrefixCache

MODEL = "google/gemma-4-E2B-it"
PROMPTS_TEXT = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The opposite of hot is",
]


@pytest.fixture(scope="module")
def backend():
    b = CustomTorchBackend(device="cpu")
    b.load_model(MODEL)
    return b


def _toks(backend, text):
    return backend.tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def _decode(backend, cache, first, n=5):
    """Greedy-decode n tokens from a seeded paged cache (single row), like generate()."""
    out, tok = [first], first
    for _ in range(n):
        logits = backend.model(torch.tensor([[tok]], device=backend.device), kv_cache=cache)
        tok = int(logits[:, -1, :].argmax(-1))
        out.append(tok)
    return out


def test_prefill_batch_k1_identical_to_single(backend):
    p = _toks(backend, PROMPTS_TEXT[0])

    backend.prefix_cache = PrefixCache(pools=backend.pools)  # fresh → full prefill both ways
    (cache_b, first_b, kvlen_b), = backend.prefill_batch([p])
    dec_b = _decode(backend, cache_b, first_b)

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    cache_s, first_s, kvlen_s = backend.prefill(p)
    dec_s = _decode(backend, cache_s, first_s)

    assert kvlen_b == kvlen_s == len(p)
    assert first_b == first_s                  # identical forward → identical token
    assert dec_b == dec_s                      # KV distributed correctly → identical decode


def test_prefill_batch_k3_first_tokens_match(backend):
    prompts = [_toks(backend, t) for t in PROMPTS_TEXT]

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    singles = [backend.prefill(p) for p in prompts]
    first_single = [s[1] for s in singles]
    kvlen_single = [s[2] for s in singles]

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    batched = backend.prefill_batch(prompts)
    first_batch = [b[1] for b in batched]
    kvlen_batch = [b[2] for b in batched]

    assert kvlen_batch == kvlen_single == [len(p) for p in prompts]
    # padded batch reorders reductions → allow rare near-tie flips, but the clear majority must match
    matches = sum(a == b for a, b in zip(first_batch, first_single))
    assert matches >= 2
