"""KV pools sized from a memory budget: every layer that owns K/V, K and V, the model's dtype."""

from types import SimpleNamespace

import torch

from inference_server.models.paged_kv_cache import blocks_for_budget


def _model(layers, dtype=torch.bfloat16):
    attn = [SimpleNamespace(self_attn=SimpleNamespace(num_kv_heads=h, head_dim=d, is_kv_shared=shared))
            for h, d, shared in layers]
    return SimpleNamespace(model=SimpleNamespace(layers=attn),
                           parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1, dtype=dtype))]))


def test_qwen3_30b_a3b_holds_about_96_kib_of_kv_per_token():
    m = _model([(4, 128, False)] * 48)                 # 48 layers, 4 KV heads of 128, bf16
    per_block = 2 * 4 * 16 * 128 * 2 * 48              # 16-token blocks
    assert blocks_for_budget(m, 70 * 2**30, 16) == 70 * 2**30 // per_block
    assert blocks_for_budget(m, 70 * 2**30, 16) * 16 > 700_000      # ~700k tokens in 70 GiB, not 8k


def test_layers_sharing_kv_cost_nothing_and_a_tiny_budget_still_gives_one_block():
    shared = _model([(4, 128, False), (4, 128, True)])
    alone = _model([(4, 128, False)])
    assert blocks_for_budget(shared, 10**9, 16) == blocks_for_budget(alone, 10**9, 16)
    assert blocks_for_budget(alone, 1, 16) == 1
