"""The fused q/k/v projection matches the three separate ones."""

import copy

import torch

from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM


def test_fused_qkv_matches_separate_projections(tiny_qwen3_moe_dir):
    ours = Qwen3MoeForCausalLM.from_hf(tiny_qwen3_moe_dir).eval()
    fz = copy.deepcopy(ours)
    for layer in fz.model.layers:
        layer.self_attn.fuse_qkv()
    attn = fz.model.layers[0].self_attn
    assert not hasattr(attn, "q_proj") and attn.qkv_weight.shape[0] == (attn.num_q_heads + 2 * attn.num_kv_heads) * attn.head_dim
    ids = torch.tensor([[5, 17, 300, 42, 8, 99, 123, 7, 55]])
    with torch.no_grad():
        torch.testing.assert_close(fz(ids), ours(ids), atol=2e-2, rtol=2e-2)
