"""prefill_batch (one padded forward over K prompts) matches per-row prefill.

K=1 is the strong check: same [1,L] forward, no padding → byte-identical first token AND identical
decode from the distributed paged cache (proves the per-row KV slice/append is correct). K>1 checks
the padded batching produces the same first tokens (allowing rare near-tie flips from batch
reduction-order noise — same caveat as the Triton kernel). Tiny Qwen3-MoE checkpoint, CPU, bf16.
"""

import pytest
import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.paged_kv_cache import PrefixCache

# Odd lengths: a repeat is a partial prefix hit, never a full one (see test_qwen3_moe_parity).
PROMPTS = [
    [5, 17, 300, 42, 8, 99, 123, 7, 55],
    [12, 900, 31, 64, 2, 77, 410],
    [3, 3, 150, 151, 152, 9, 10, 11, 12, 13, 14],
]


@pytest.fixture(scope="module")
def backend(tiny_qwen3_moe_dir):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUSTOM_BACKEND_BLOCKS", "64")
        mp.setenv("CUSTOM_BACKEND_BLOCK_SIZE", "2")   # short prompts still fill blocks → prefix sharing
        b = CustomTorchBackend(device="cpu", model_name=tiny_qwen3_moe_dir)
        b.load_model(tiny_qwen3_moe_dir)
    return b


def _decode(backend, cache, first, n=5):
    """Greedy-decode n tokens from a seeded paged cache (single row), like generate()."""
    out, tok = [first], first
    for _ in range(n):
        logits = backend.model(torch.tensor([[tok]], device=backend.device), kv_cache=cache)
        tok = int(logits[:, -1, :].argmax(-1))
        out.append(tok)
    return out


def test_prefill_batch_k1_identical_to_single(backend):
    p = PROMPTS[0]

    backend.prefix_cache = PrefixCache(pools=backend.pools)  # fresh → full prefill both ways
    (cache_b, first_b, kvlen_b), = backend.prefill_batch([p])
    dec_b = _decode(backend, cache_b, first_b)

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    cache_s, first_s, kvlen_s = backend.prefill(p)
    dec_s = _decode(backend, cache_s, first_s)

    assert kvlen_b == kvlen_s == len(p)
    assert first_b == first_s                  # identical forward → identical token
    assert dec_b == dec_s                      # KV distributed correctly → identical decode


def test_prefill_batch_cache_hit_suffix_only_matches_full(backend):
    """With a warm prefix cache, prefill_batch forwards only the 1-token suffix against the seeded
    prefix — must still equal a cold full single prefill (first token + decode)."""
    p = PROMPTS[1]

    backend.prefix_cache = PrefixCache(pools=backend.pools)          # cold reference
    cache_s, first_s, _ = backend.prefill(p)
    dec_s = _decode(backend, cache_s, first_s)

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    backend.prefill(p)                                               # warm the cache
    (cache_b, first_b, kvlen_b), = backend.prefill_batch([p])        # → suffix-only path
    dec_b = _decode(backend, cache_b, first_b)

    assert kvlen_b == len(p)
    assert first_b == first_s
    assert dec_b == dec_s


def test_prefill_batch_k3_first_tokens_match(backend):
    backend.prefix_cache = PrefixCache(pools=backend.pools)
    singles = [backend.prefill(p) for p in PROMPTS]
    first_single = [s[1] for s in singles]
    kvlen_single = [s[2] for s in singles]

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    batched = backend.prefill_batch(PROMPTS)
    first_batch = [b[1] for b in batched]
    kvlen_batch = [b[2] for b in batched]

    assert kvlen_batch == kvlen_single == [len(p) for p in PROMPTS]
    # padded batch reorders reductions → allow rare near-tie flips, but the clear majority must match
    matches = sum(a == b for a, b in zip(first_batch, first_single))
    assert matches >= 2
