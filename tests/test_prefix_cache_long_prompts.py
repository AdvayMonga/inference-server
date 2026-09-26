"""Prefix caching for prompts longer than the sliding window.

Sliding layers keep only their last `window` tokens, so a long prompt used to be skipped by the
prefix cache entirely. The radix cache now stores full-attention blocks for the whole prefix and
sliding blocks only where they are still resident, and a lookup backs off to a depth whose window
is complete on every sliding layer. Tiny random model, CPU, no weights.
"""

from __future__ import annotations

import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.gemma4 import GemmaForCausalLM, GemmaModel
from inference_server.models.paged_kv_cache import (
    BlockPool,
    PagedKVCache,
    RadixPrefixCache,
    make_pools_for_gemma,
)

BS, W, V = 4, 16, 64          # window = 4 blocks


def _pools():
    dev = torch.device("cpu")
    full = BlockPool(64, BS, 1, 2, torch.float32, dev, window=None)
    sliding = BlockPool(64, BS, 1, 2, torch.float32, dev, window=W)
    return [full, sliding, None]    # None = a KV-shared layer


def _prefill(pools, pc, prompt):
    """Lookup, append the uncached suffix on both layers, store. Returns (matched, cache)."""
    m, shared = pc.lookup(prompt)
    c = PagedKVCache(pools=pools, shared_prefix=shared, shared_prefix_tokens=m)
    n = len(prompt) - m
    for L in (0, 1):
        c.append(L, torch.zeros(1, 1, n, 2), torch.zeros(1, 1, n, 2))
    pc.store(prompt, c.block_tables)
    return m, c


def test_prompt_longer_than_window_is_stored_and_hit():
    """The regression: a >window prompt was never stored, so its continuation matched nothing."""
    pools = _pools()
    pc = RadixPrefixCache(pools, max_block_fraction=1.0)
    p = list(range(50))                               # 12 full blocks; sliding keeps 8..11
    _, c = _prefill(pools, pc, p)
    c.free_all()

    m, shared = pc.lookup(p + [99] * 10)              # a later turn extending the prompt
    assert m == 48
    # Seeded exactly as a cold prefill to 48 tokens would leave it.
    c2 = PagedKVCache(pools=pools, shared_prefix=shared, shared_prefix_tokens=m)
    assert c2.evicted[1] == (48 - W) // BS == 8
    assert c2.block_tables[1] == [-1] * 8 + c2.block_tables[1][8:]
    assert len(c2.block_tables[1]) == 12 and all(b >= 0 for b in c2.block_tables[0])
    c2.free_all()


def test_shorter_match_is_refused_when_its_window_was_evicted():
    """Matching only part of a stored long prompt needs sliding blocks it no longer holds."""
    pools = _pools()
    pc = RadixPrefixCache(pools, max_block_fraction=1.0)
    p = list(range(50))
    _, c = _prefill(pools, pc, p)
    c.free_all()
    free = [pools[0].free_count, pools[1].free_count]

    # Shares 11 blocks: its window needs sliding blocks 7..10, but only 8..11 were kept.
    m, shared = pc.lookup(p[:44] + [99] * 8)
    assert (m, shared) == (0, {})
    assert [pools[0].free_count, pools[1].free_count] == free, "a refused match must acquire nothing"

    # Shares all 12 blocks: window 8..11 is resident, so it is usable.
    m, shared = pc.lookup(p[:48] + [99] * 8)
    assert m == 48
    PagedKVCache(pools=pools, shared_prefix=shared, shared_prefix_tokens=m).free_all()


def test_short_prompt_completes_a_path_a_long_prompt_stored_partially():
    """Without filling missing sliding blocks, a long prompt's partial path would shadow every
    short prompt sharing its start, and those used to hit before long prompts were stored."""
    pools = _pools()
    pc = RadixPrefixCache(pools, max_block_fraction=1.0)
    long_p = list(range(50))
    short_p = long_p[:24] + [99] * 4
    for p in (long_p, short_p):
        _, c = _prefill(pools, pc, p)
        c.free_all()
    m, shared = pc.lookup(short_p)
    assert m == 28
    PagedKVCache(pools=pools, shared_prefix=shared, shared_prefix_tokens=m).free_all()


def test_refcounts_return_to_zero():
    pools = _pools()
    pc = RadixPrefixCache(pools, max_block_fraction=1.0)
    p = list(range(50))
    for q in (p, p + [99] * 10, p[:24] + [98] * 4, p[:44] + [97] * 8):
        _, c = _prefill(pools, pc, q)
        c.free_all()
    pc.reclaim(10**9)
    assert pools[0].free_count == pools[0].num_blocks
    assert pools[1].free_count == pools[1].num_blocks


# ---------------------------------------------------------------- output equivalence

def _backend():
    torch.manual_seed(0)
    m = GemmaModel(
        vocab_size=V, hidden_size=32, num_layers=4, intermediate_size=64,
        num_q_heads=4, num_kv_heads=1, head_dim=8, global_head_dim=8, hidden_per_layer=8,
        layer_types=["sliding_attention", "sliding_attention", "full_attention",
                     "sliding_attention"],
        sliding_window=W, num_kv_shared_layers=1, use_double_wide_mlp=False,
        eps=1e-6, dtype=torch.float32,
    )
    b = CustomTorchBackend(device="cpu")
    b.model = GemmaForCausalLM(m, final_logit_softcapping=30.0).eval()
    b.pools = make_pools_for_gemma(b.model, num_blocks_per_pool=256, block_size=BS)
    b.prefix_cache = RadixPrefixCache(b.pools)
    return b


@torch.no_grad()
def _logits_after(b, prompt, forced):
    """Prefill `prompt`, then teacher-force `forced`; return the stacked next-token logits."""
    cache, _, _ = b.prefill(prompt)
    out = [b.model(torch.tensor([[t]]), kv_cache=cache)[0, -1] for t in forced]
    cache.free_all()
    return torch.stack(out)


def test_cached_long_prompt_matches_uncached():
    torch.manual_seed(1)
    p = torch.randint(0, V, (50,)).tolist()
    p2 = p + torch.randint(0, V, (7,)).tolist()       # suffix straddles the stored prompt's end
    forced = torch.randint(0, V, (12,)).tolist()

    b = _backend()
    cold = _logits_after(b, p2, forced)
    assert b.last_cache_hit_tokens == 0

    b.prefix_cache = RadixPrefixCache(b.pools)
    b.prefill(p)[0].free_all()                        # store the long prompt
    warm = _logits_after(b, p2, forced)
    assert b.last_cache_hit_tokens == 48

    assert torch.equal(cold.argmax(-1), warm.argmax(-1))
    assert (cold - warm).abs().max().item() < 1e-4
