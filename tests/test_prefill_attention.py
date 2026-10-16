"""SDPA prefill over a paged prefix + suffix matches the Triton paged prefill kernel (CUDA)."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("prefix,suffix", [(0, 300), (40, 7), (160, 500), (33, 1)])
def test_sdpa_prefill_matches_the_paged_kernel(prefix, suffix):
    from inference_server.models.paged_attention_kernel import paged_prefill_attention
    from inference_server.models.prefill_attention import sdpa_prefill
    torch.manual_seed(0)
    Hq, Hkv, D, bs = 32, 4, 128, 16
    total = prefix + suffix
    nb = (total + bs - 1) // bs
    kall = torch.randn(1, Hkv, nb * bs, D, dtype=torch.bfloat16, device="cuda")
    vall = torch.randn(1, Hkv, nb * bs, D, dtype=torch.bfloat16, device="cuda")
    ids = torch.randperm(nb * 3, device="cuda")[:nb]                       # scattered physical blocks
    pool_k = torch.zeros(nb * 3, Hkv, bs, D, dtype=torch.bfloat16, device="cuda")
    pool_v = torch.zeros_like(pool_k)
    pool_k[ids] = kall[0].view(Hkv, nb, bs, D).permute(1, 0, 2, 3)
    pool_v[ids] = vall[0].view(Hkv, nb, bs, D).permute(1, 0, 2, 3)
    q = torch.randn(1, Hq, suffix, D, dtype=torch.bfloat16, device="cuda")
    k_suf, v_suf = kall[:, :, prefix:total], vall[:, :, prefix:total]
    ctx = SimpleNamespace(pools=[SimpleNamespace(k=pool_k, v=pool_v)], _prefix=[prefix], _suffix=[suffix],
                          caches=[SimpleNamespace(block_tables=[ids.tolist()])])
    got = sdpa_prefill(q, k_suf, v_suf, ctx, 0, D ** -0.5)
    bt = ids.to(torch.int32)[None]
    ref = paged_prefill_attention(q.transpose(1, 2), pool_k, pool_v, bt,
                                  torch.tensor([prefix], dtype=torch.int32, device="cuda"),
                                  torch.tensor([suffix], dtype=torch.int32, device="cuda"), scale=D ** -0.5)
    torch.testing.assert_close(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
