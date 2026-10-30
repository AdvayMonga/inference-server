"""The Triton MoE grouped GEMM matches torch._grouped_mm on expert-sorted rows (CUDA)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("T,top_k,E", [(1, 8, 128), (7, 8, 128), (64, 8, 128), (5, 2, 8)])
def test_moe_gemm_matches_grouped_mm(T, top_k, E):
    from inference_server.models import moe_kernel
    torch.manual_seed(T)
    K, N = 256, 384
    flat = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    experts = torch.stack([torch.randperm(E, device="cuda")[:top_k] for _ in range(T)])      # distinct per token
    sorted_e, order = torch.sort(experts.view(-1), stable=True)
    tok = order // top_k
    M = tok.numel()
    offs = torch.searchsorted(sorted_e, torch.arange(E, device="cuda"), right=True).to(torch.int32)
    ref = torch._grouped_mm(flat[tok], w.transpose(-2, -1), offs=offs)
    sorted_ids, expert_ids = moe_kernel.align(sorted_e, E, 16)
    assert sorted_ids.numel() == M + E * 15 and (sorted_ids < M).sum() == M
    got = moe_kernel.moe_gemm(flat, tok, w, sorted_ids, expert_ids, M)                     # auto split-K (fp32 when split)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=2e-2)
    got2 = moe_kernel.moe_gemm(flat[tok].contiguous(), None, w, sorted_ids, expert_ids, M, split_k=1)   # rows direct, bf16
    assert got2.dtype == torch.bfloat16
    torch.testing.assert_close(got2.float(), ref.float(), atol=3e-2, rtol=2e-2)
    got4 = moe_kernel.moe_gemm(flat, tok, w, sorted_ids, expert_ids, M, split_k=4)
    assert got4.dtype == torch.float32
    torch.testing.assert_close(got4, ref.float(), atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("T,top_k,E", [(1, 8, 128), (7, 8, 128), (64, 8, 128), (256, 8, 128), (5, 2, 8), (3, 1, 4)])
def test_align_fused_matches_align_and_the_scatter_gemm_matches_index_add(T, top_k, E):
    from inference_server.models import moe_kernel
    torch.manual_seed(T + 100)
    K, N = 256, 384
    experts = torch.stack([torch.randperm(E, device="cuda")[:top_k] for _ in range(T)])
    weights = torch.rand(T, top_k, device="cuda")
    sorted_e, order = torch.sort(experts.view(-1), stable=True)
    M = sorted_e.numel()
    ref_ids, ref_experts = moe_kernel.align(sorted_e.clone(), E, 16)
    sorted_ids, expert_ids, tok, pair_w = moe_kernel.align_fused(sorted_e, order, weights, E, top_k, 16)
    # align sizes its ids M + E * 15 (not always a multiple of 16); the fused one stops at the last whole block.
    assert sorted_ids.numel() == expert_ids.numel() * 16 and sorted_ids.numel() <= ref_ids.numel()
    assert torch.equal(sorted_ids, ref_ids[:sorted_ids.numel()]) and torch.equal(expert_ids, ref_experts)
    assert torch.equal(tok.long(), order // top_k) and torch.equal(pair_w, weights.view(-1)[order])
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    offs = torch.searchsorted(sorted_e, torch.arange(E, device="cuda"), right=True).to(torch.int32)
    y = torch._grouped_mm(act, w.transpose(-2, -1), offs=offs).float() * pair_w[:, None]
    ref = torch.zeros(T, N, device="cuda").index_add_(0, order // top_k, y)
    out = torch.zeros(T, N, device="cuda")
    moe_kernel.moe_gemm_scatter(act, w, sorted_ids, expert_ids, M, tok, pair_w, out)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("router,T,top_k,E,renorm", [("route_fused",) + c for c in [(1, 8, 128, True), (7, 8, 128, True), (32, 8, 128, False), (5, 2, 8, True),
                                               (3, 1, 4, True)]]
                         + [("route_multi", 1, 8, 128, True), ("route_multi", 33, 8, 128, False),
                            ("route_multi", 200, 8, 128, True), ("route_multi", 5, 2, 8, True),
                            ("route_multi", 256, 8, 128, True), ("route_multi", 4096, 8, 128, True)])
def test_router_matches_the_four_launch_routing(router, T, top_k, E, renorm):
    """Same grouping (every pair once, each block one expert), same weights, same scattered output."""
    from inference_server.models import moe_kernel
    from inference_server.models.qwen3_moe import route
    torch.manual_seed(T + 7)
    logits = torch.randn(T, E, dtype=torch.bfloat16, device="cuda") * 2
    rw, rids = route(logits, top_k, renorm)
    K_, N = 256, 384
    out = torch.full((T, N), float("nan"), device="cuda")              # the router zeroes the down GEMM's accumulator
    sorted_ids, expert_ids, tok, pair_w = getattr(moe_kernel, router)(logits, E, top_k, renorm, 16, out=out)
    assert torch.equal(out, torch.zeros_like(out))
    M = T * top_k
    assert sorted_ids.numel() == expert_ids.numel() * 16 and (sorted_ids < M).sum() == M
    real = sorted_ids[sorted_ids < M]
    assert torch.equal(real.sort().values, torch.arange(M, device="cuda", dtype=torch.int32))   # every pair once
    # bf16 logits tie often, and the kernel breaks ties by lowest index while torch.topk's order is unspecified: compare
    # each token's expert SET and each (token, expert) weight, and the scattered output (order-free).
    pair_e = torch.full((M,), -1, dtype=torch.int32, device="cuda")
    for b in range(expert_ids.numel()):
        ids = sorted_ids[b * 16:(b + 1) * 16]
        ids = ids[ids < M].long()
        if ids.numel():
            assert expert_ids[b] < E
            pair_e[ids] = expert_ids[b]
    assert (pair_e >= 0).all()
    assert torch.equal(tok.long(), torch.arange(M, device="cuda") // top_k)
    ref_w = torch.zeros(T, E, device="cuda").scatter_(1, rids, rw.float())          # weight per (token, expert)
    for t in range(T):
        assert set(pair_e[t * top_k:(t + 1) * top_k].tolist()) == set(rids[t].tolist())
    torch.testing.assert_close(pair_w, ref_w[tok.long(), pair_e.long()], atol=2e-3, rtol=2e-2)
    act = torch.randn(M, K_, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(E, N, K_, dtype=torch.bfloat16, device="cuda") * 0.05
    moe_kernel.moe_gemm_scatter(act, w, sorted_ids, expert_ids, M, tok, pair_w, out)
    ref = torch.zeros(T, N, device="cuda")
    for i in range(M):
        ref[i // top_k] += (act[i].float() @ w[pair_e[i]].float().T) * ref_w[i // top_k, pair_e[i]]
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=3e-2)


@pytest.mark.parametrize("T,top_k,N", [(1, 8, 2048), (300, 8, 2048), (1000, 8, 384)])
def test_gather_weighted_matches_index_add(T, top_k, N):
    from inference_server.models import moe_kernel
    torch.manual_seed(T + 1)
    M = T * top_k
    y = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    order = torch.randperm(M, device="cuda")                       # sorted position -> pair (token * K + slot)
    weights = torch.rand(T, top_k, device="cuda")
    pos = torch.empty_like(order).scatter_(0, order, torch.arange(M, device="cuda"))   # pair -> sorted position
    ref = torch.zeros(T, N, device="cuda").index_add_(0, order // top_k, y.float() * weights.view(-1)[order, None])
    got = moe_kernel.gather_weighted(y, pos.view(T, top_k), weights)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("T,top_k,N", [(1, 8, 2048), (300, 8, 2048), (1000, 8, 384)])
def test_scatter_add_weighted_matches_index_add(T, top_k, N):
    from inference_server.models import moe_kernel
    torch.manual_seed(T)
    M = T * top_k
    y = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    order = torch.randperm(M, device="cuda")
    tok = (order // top_k).to(torch.int64)
    pair_w = torch.rand(M, device="cuda")
    ref = torch.zeros(T, N, device="cuda").index_add_(0, tok, y.float() * pair_w[:, None])
    out = torch.zeros(T, N, device="cuda")
    moe_kernel.scatter_add_weighted(y, tok, pair_w, out)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("T,top_k,E", [(1, 8, 128), (64, 8, 128)])
def test_fp8_weights_match_the_dequantized_bf16_gemms(T, top_k, E):
    """Both Triton GEMMs with fp8 e4m3 weights and a per-channel scale, against the same GEMMs on the dequantized
    weights (the numerics the megakernel uses too: W8A16, scale on the fp32 accumulator)."""
    from inference_server.models import moe_kernel
    from inference_server.models.qwen3_moe import quantize_rowwise
    torch.manual_seed(T)
    K, N = 256, 384
    flat = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    q, s = quantize_rowwise(w)
    deq = (q.float() * s[..., None]).to(torch.bfloat16)
    experts = torch.stack([torch.randperm(E, device="cuda")[:top_k] for _ in range(T)])
    sorted_e, order = torch.sort(experts.view(-1), stable=True)
    tok = (order // top_k).to(torch.int32)
    M = tok.numel()
    sorted_ids, expert_ids = moe_kernel.align(sorted_e, E, 16)
    ref = moe_kernel.moe_gemm(flat, tok, deq, sorted_ids, expert_ids, M)
    got = moe_kernel.moe_gemm(flat, tok, q, sorted_ids, expert_ids, M, w_scale=s.contiguous())
    torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=3e-2)
    pair_w = torch.rand(M, device="cuda")
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wd = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    qd, sd = quantize_rowwise(wd)
    deqd = (qd.float() * sd[..., None]).to(torch.bfloat16)
    out_ref = torch.zeros(T, N, dtype=torch.float32, device="cuda")
    out_got = torch.zeros(T, N, dtype=torch.float32, device="cuda")
    moe_kernel.moe_gemm_scatter(act, deqd, sorted_ids, expert_ids, M, tok, pair_w, out_ref)
    moe_kernel.moe_gemm_scatter(act, qd, sorted_ids, expert_ids, M, tok, pair_w, out_got, w_scale=sd.contiguous())
    torch.testing.assert_close(out_got, out_ref, atol=5e-2, rtol=3e-2)


@pytest.mark.parametrize("M,K", [(3, 2048), (1000, 768)])
def test_quant_rows_matches_quantize_rowwise(M, K):
    from inference_server.models import moe_kernel
    from inference_server.models.qwen3_moe import quantize_rowwise
    torch.manual_seed(M)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 3
    q, s = moe_kernel.quant_rows(x)
    q_ref, s_ref = quantize_rowwise(x)
    torch.testing.assert_close(s, s_ref)
    assert (q.float() != q_ref.float()).float().mean() < 1e-3          # a rounding-boundary element or two at most


@pytest.mark.parametrize("T,top_k,E", [(400, 8, 128), (1024, 8, 128), (2048, 8, 128)])
def test_tall_tiles_match_the_torch_grouped_gemms(T, top_k, E):
    """Past 2048 pairs the GEMMs run 32- and 64-row tiles (qwen3_moe.moe_block_m): the gate_up GEMM over the routed
    rows and the scattered down GEMM match the torch grouped GEMMs on the same routing."""
    from inference_server.models import moe_kernel
    from inference_server.models.qwen3_moe import moe_block_m, route
    torch.manual_seed(T)
    K, N = 256, 384
    M = T * top_k
    bm = moe_block_m(M)
    assert bm == (32 if M <= 4096 else 64)
    logits = torch.randn(T, E, dtype=torch.bfloat16, device="cuda") * 2
    rw, rids = route(logits, top_k, True)
    sorted_ids, expert_ids, tok, pair_w = moe_kernel.route_multi(logits, E, top_k, True, bm)
    assert sorted_ids.numel() == expert_ids.numel() * bm and (sorted_ids < M).sum() == M
    flat = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda") * 0.05
    h = moe_kernel.moe_gemm(flat, tok, w1, sorted_ids, expert_ids, M, block_m=bm)
    out = torch.zeros(T, K, device="cuda")
    moe_kernel.moe_gemm_scatter(h, w2, sorted_ids, expert_ids, M, tok, pair_w, out, block_m=bm)
    # reference on the kernel's own routing (pair i -> expert of its block), through torch's grouped GEMMs
    pair_e = torch.full((M,), -1, dtype=torch.long, device="cuda")
    for b in range(expert_ids.numel()):
        ids = sorted_ids[b * bm:(b + 1) * bm]
        ids = ids[ids < M].long()
        pair_e[ids] = expert_ids[b].long()
    assert (pair_e >= 0).all()
    order = torch.argsort(pair_e, stable=True)
    sorted_e = pair_e[order]
    offs = torch.searchsorted(sorted_e, torch.arange(E, device="cuda"), right=True).to(torch.int32)
    h_ref = torch._grouped_mm(flat[tok.long()[order]], w1.transpose(-2, -1), offs=offs)
    torch.testing.assert_close(h[order].float(), h_ref.float(), atol=3e-2, rtol=2e-2)
    y = torch._grouped_mm(h_ref, w2.transpose(-2, -1), offs=offs).float() * pair_w[order, None]
    ref = torch.zeros(T, K, device="cuda").index_add_(0, tok.long()[order], y)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=3e-2)


@pytest.mark.parametrize("T,top_k,E,bm", [(64, 8, 128, 16), (1024, 8, 128, 64)])
def test_fp8_activations_match_torch_scaled_grouped_mm(T, top_k, E, bm):
    """W8A8: fp8 activations (per-row scale) x fp8 weights (per-channel scale) on the tensor cores equal torch's
    _scaled_grouped_mm on the same rows and scales, for the gate_up GEMM and the scattered down GEMM."""
    from inference_server.models import moe_kernel
    torch.manual_seed(T + 3)
    K, N = 256, 384
    M = T * top_k
    logits = torch.randn(T, E, dtype=torch.bfloat16, device="cuda") * 2
    sorted_ids, expert_ids, tok, pair_w = moe_kernel.route_multi(logits, E, top_k, True, bm)
    flat = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    s1 = w1.abs().amax(-1).float() / 448.0                                  # per output channel
    w1q = (w1.float() / s1[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    a8, sa = moe_kernel.quant_rows(flat)
    h = moe_kernel.moe_gemm(a8, tok, w1q, sorted_ids, expert_ids, M, block_m=bm, w_scale=s1, a_scale=sa)
    pair_e = torch.full((M,), -1, dtype=torch.long, device="cuda")
    for b in range(expert_ids.numel()):
        ids = sorted_ids[b * bm:(b + 1) * bm]
        ids = ids[ids < M].long()
        pair_e[ids] = expert_ids[b].long()
    order = torch.argsort(pair_e, stable=True)
    offs = torch.searchsorted(pair_e[order], torch.arange(E, device="cuda"), right=True).to(torch.int32)
    rows = tok.long()[order]
    h_ref = torch._scaled_grouped_mm(a8[rows], w1q.transpose(-2, -1), sa[rows], s1, offs=offs, out_dtype=torch.bfloat16)
    torch.testing.assert_close(h[order].float(), h_ref.float(), atol=3e-2, rtol=2e-2)
    act = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda") * 0.05
    s2 = w2.abs().amax(-1).float() / 448.0
    w2q = (w2.float() / s2[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    act8, sact = moe_kernel.quant_rows(act)
    out = torch.zeros(T, K, device="cuda")
    moe_kernel.moe_gemm_scatter(act8, w2q, sorted_ids, expert_ids, M, tok, pair_w, out, block_m=bm, w_scale=s2,
                                a_scale=sact)
    y = torch._scaled_grouped_mm(act8[order], w2q.transpose(-2, -1), sact[order], s2, offs=offs,
                                 out_dtype=torch.bfloat16).float() * pair_w[order, None]
    ref = torch.zeros(T, K, device="cuda").index_add_(0, rows, y)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=3e-2)
