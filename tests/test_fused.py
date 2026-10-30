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
    from inference_server.models.common import RotaryEmbedding, apply_rotary
    torch.manual_seed(0)
    B, S, H, D = 2, 7, 4, 128
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    pos = torch.arange(S, device="cuda")[None].repeat(B, 1) + torch.tensor([[0], [1000]], device="cuda")
    cos, sin = (t.to(torch.bfloat16) for t in RotaryEmbedding(D, 1e6).cuda()(pos))
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


@pytest.mark.parametrize("pool_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_kv_store_matches_the_index_put_scatter(pool_dtype):
    from inference_server.models import fused
    torch.manual_seed(0)
    R, H, bs, D = 5, 4, 16, 128
    pk = torch.randn(64, H, bs, D, dtype=torch.bfloat16, device="cuda").to(pool_dtype)
    pv = torch.randn(64, H, bs, D, dtype=torch.bfloat16, device="cuda").to(pool_dtype)
    rk, rv = pk.clone(), pv.clone()
    bid = torch.tensor([3, 17, 0, 63, 17], device="cuda")
    lb = torch.tensor([0, 2, 1, 3, 0], device="cuda")
    bt = torch.zeros(R, 4, dtype=torch.long, device="cuda")
    bt[torch.arange(R, device="cuda"), lb] = bid                                             # the row's current block
    slot = torch.tensor([0, 15, 7, 1, 2], device="cuda")
    k = torch.randn(R, 1, H, D, dtype=torch.bfloat16, device="cuda").transpose(1, 2)        # rope's layout
    v = torch.randn(R, 1, 3 * H * D, dtype=torch.bfloat16, device="cuda")[..., 2 * H * D:].view(R, 1, H, D).transpose(1, 2)
    fused.kv_store(pk, pv, bt, lb, slot, k, v)
    rk[bid, :, slot, :] = k[:, :, 0, :].to(pool_dtype)
    rv[bid, :, slot, :] = v[:, :, 0, :].to(pool_dtype)
    bits = lambda t: t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)
    assert torch.equal(bits(pk), bits(rk)) and torch.equal(bits(pv), bits(rv))


def test_topk_softmax_keeps_ids_in_range_on_nan_rows():
    from inference_server.models import fused
    logits = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
    logits[1] = float("nan")
    w, ids = fused.topk_softmax(logits, 8, True)
    assert ids.min() >= 0 and ids.max() < 128


def test_embed_rope_matches_embedding_and_rotary():
    from inference_server.models import fused
    from inference_server.models.common import RotaryEmbedding
    torch.manual_seed(0)
    w = torch.randn(1000, 2048, dtype=torch.bfloat16, device="cuda")
    rope = RotaryEmbedding(128, 1_000_000.0).cuda()
    ids = torch.randint(0, 1000, (5, 3), device="cuda")
    pos = torch.tensor([[0, 1, 2], [7, 8, 9], [4095, 4096, 4097], [30000, 30001, 30002], [17, 18, 19]], device="cuda")
    h, cos, sin = fused.embed_rope(ids, w, rope.inv_freq, pos)
    rc, rs = (t.to(torch.bfloat16) for t in rope(pos))
    assert torch.equal(h, F.embedding(ids, w))
    torch.testing.assert_close(cos, rc, atol=8e-3, rtol=0)
    torch.testing.assert_close(sin, rs, atol=8e-3, rtol=0)
    assert (cos != rc).float().mean() < 0.01 and (sin != rs).float().mean() < 0.01      # one bf16 ulp at most, rarely


def test_step_inputs_match_the_torch_ones():
    from inference_server.models import fused
    seq = torch.tensor([0, 15, 16, 17, 4095, 31], device="cuda")
    kb, ks, sl = fused.step_inputs(seq, 16)
    assert torch.equal(kb, seq // 16) and torch.equal(ks, seq % 16) and torch.equal(sl, (seq + 1).int())


@pytest.mark.parametrize("new", [False, True])
def test_stage_decode_matches_the_copies_it_replaces(new):
    from inference_server.models import fused
    torch.manual_seed(0)
    bucket, cols, n, scratch = 8, 2048, 5, 3
    g = {"tokens": torch.full((bucket, 1), 9, dtype=torch.long, device="cuda"),
         "pos": torch.full((bucket, 1), 9, dtype=torch.long, device="cuda"),
         "seqlens": torch.full((bucket,), 9, dtype=torch.long, device="cuda"),
         "bt": torch.full((bucket, cols), 7, dtype=torch.int32, device="cuda")}
    cur = torch.randint(0, 100, (n, 1), device="cuda")
    seq = torch.randint(0, 500, (n,), device="cuda")
    bt = torch.randint(0, 900, (n, 40), device="cuda")
    bt[1, 39] = -1
    ref_bt = bt.clone()
    blk = torch.tensor([-1, 77, -1, 88, -1], device="cuda")
    col = torch.tensor([-1, 12, -1, 39, -1], device="cuda")
    if new:
        ref_bt[1, 12], ref_bt[3, 39] = 77, 88
    nxt = torch.empty_like(seq)
    fused.stage_decode(g, cur, seq.unsqueeze(1), seq, bt, n, scratch, new=(blk, col) if new else None, seq_next=nxt)
    want = torch.full((bucket, cols), scratch, dtype=torch.int32, device="cuda")
    want[:n, :40] = ref_bt.clamp_min(0).int()
    assert torch.equal(g["bt"], want) and torch.equal(bt, ref_bt)
    assert torch.equal(g["tokens"][:n], cur) and torch.equal(g["tokens"][n:], torch.zeros_like(g["tokens"][n:]))
    assert torch.equal(g["pos"][:n, 0], seq) and torch.equal(g["seqlens"][:n], seq)
    assert torch.equal(g["seqlens"][n:], torch.ones_like(g["seqlens"][n:])) and torch.equal(nxt, seq + 1)


@pytest.mark.parametrize("N", [1, 7, 256])
def test_argmax_matches_torch_ties_included(N):
    from inference_server.models import fused
    torch.manual_seed(N)
    x = torch.randn(N, 151936, dtype=torch.bfloat16, device="cuda")
    x[0, 5000] = x[0, 140000] = 50.0                    # a tie across chunks: the first index, as torch
    if N > 1:
        x[1, 8191] = x[1, 8192] = 60.0                   # a tie across a chunk edge
    assert torch.equal(fused.argmax(x), x.argmax(-1))
    view = torch.randn(N, 2, 151936, dtype=torch.bfloat16, device="cuda")[:, -1, :]      # a strided row view
    assert torch.equal(fused.argmax(view), view.argmax(-1))
