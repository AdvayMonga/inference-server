"""Fused Triton kernels match the torch code they replace (CUDA)."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _norm(x, w, eps):
    x32 = x.float()
    return w * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def test_rmsnorm_and_add_rmsnorm():
    from inference_server.models import fused
    torch.manual_seed(0)
    x = torch.randn(3, 5, 2048, dtype=torch.bfloat16, device="cuda") * 3
    a = torch.randn_like(x)
    w = (torch.rand(2048, device="cuda") + 0.5).to(torch.bfloat16)
    torch.testing.assert_close(fused.rmsnorm(x, w, 1e-6), _norm(x, w, 1e-6), atol=2e-2, rtol=2e-2)
    h, y = fused.add_rmsnorm(x, a, w, 1e-6)
    torch.testing.assert_close(h, x + a)
    torch.testing.assert_close(y, _norm(x + a, w, 1e-6), atol=2e-2, rtol=2e-2)


def test_rmsnorm_rope_matches_norm_then_rotary():
    from inference_server.models import fused
    from inference_server.models.gemma4 import GemmaRotaryEmbedding, apply_rotary
    torch.manual_seed(0)
    B, S, H, D = 2, 7, 4, 128
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    pos = torch.arange(S, device="cuda")[None].repeat(B, 1) + torch.tensor([[0], [1000]], device="cuda")
    cos, sin = (t.to(torch.bfloat16) for t in GemmaRotaryEmbedding(D, 1e6).cuda()(pos))
    ref = apply_rotary(_norm(x, w, 1e-6), cos, sin, unsqueeze_dim=2)
    torch.testing.assert_close(fused.rmsnorm_rope(x, w, cos, sin, 1e-6), ref, atol=3e-2, rtol=3e-2)


def test_silu_mul_and_topk_softmax():
    from inference_server.models import fused
    from inference_server.models.qwen3_moe import route
    torch.manual_seed(0)
    h = torch.randn(37, 2 * 768, dtype=torch.bfloat16, device="cuda")
    g, u = h.chunk(2, dim=-1)
    torch.testing.assert_close(fused.silu_mul(h), F.silu(g) * u, atol=2e-2, rtol=2e-2)
    logits = torch.randn(37, 128, dtype=torch.bfloat16, device="cuda") * 2
    for renorm in (True, False):
        w, ids = fused.topk_softmax(logits, 8, renorm)
        rw, rids = route(logits, 8, renorm)
        assert ids.dtype == torch.int64 and ids.shape == (37, 8)
        assert torch.equal(ids.sort(dim=-1).values, rids.sort(dim=-1).values)
        torch.testing.assert_close(w.sort(dim=-1).values, rw.sort(dim=-1).values, atol=2e-3, rtol=2e-2)


def test_add_cast_and_fp32_topk_weights():
    from inference_server.models import fused
    torch.manual_seed(0)
    h = torch.randn(3, 5, 2048, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(3, 5, 2048, device="cuda")
    torch.testing.assert_close(fused.add_cast(h, y), (h.float() + y).to(torch.bfloat16))
    logits = torch.randn(9, 128, dtype=torch.bfloat16, device="cuda")
    w32, ids32 = fused.topk_softmax(logits, 8, True, fp32=True)
    w16, ids16 = fused.topk_softmax(logits, 8, True)
    assert w32.dtype == torch.float32 and torch.equal(ids32, ids16) and torch.equal(w32.to(w16.dtype), w16)


def test_kv_store_matches_the_index_put_scatter():
    from inference_server.models import fused
    torch.manual_seed(0)
    R, H, bs, D = 5, 4, 16, 128
    pk = torch.randn(64, H, bs, D, dtype=torch.bfloat16, device="cuda")
    pv = torch.randn_like(pk)
    rk, rv = pk.clone(), pv.clone()
    bid = torch.tensor([3, 17, 0, 63, 17], device="cuda")
    slot = torch.tensor([0, 15, 7, 1, 2], device="cuda")
    k = torch.randn(R, 1, H, D, dtype=torch.bfloat16, device="cuda").transpose(1, 2)        # rope's layout
    v = torch.randn(R, 1, 3 * H * D, dtype=torch.bfloat16, device="cuda")[..., 2 * H * D:].view(R, 1, H, D).transpose(1, 2)
    fused.kv_store(pk, pv, bid, slot, k, v)
    rk[bid, :, slot, :] = k[:, :, 0, :]
    rv[bid, :, slot, :] = v[:, :, 0, :]
    assert torch.equal(pk, rk) and torch.equal(pv, rv)


def test_topk_softmax_keeps_ids_in_range_on_nan_rows():
    from inference_server.models import fused
    logits = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    logits[1] = float("nan")
    w, ids = fused.topk_softmax(logits, 8, True)
    assert ids.min() >= 0 and ids.max() < 128
