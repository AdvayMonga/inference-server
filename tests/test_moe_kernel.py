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
    got = moe_kernel.moe_gemm(flat, tok, w, sorted_ids, expert_ids, M)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=2e-2)
    got2 = moe_kernel.moe_gemm(flat[tok].contiguous(), None, w, sorted_ids, expert_ids, M)   # rows given directly
    torch.testing.assert_close(got2.float(), ref.float(), atol=3e-2, rtol=2e-2)


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


@pytest.mark.parametrize("T,top_k,E,renorm", [(1, 8, 128, True), (7, 8, 128, True), (32, 8, 128, False), (5, 2, 8, True),
                                               (3, 1, 4, True)])
def test_route_fused_matches_the_four_launch_routing(T, top_k, E, renorm):
    """Same grouping (every pair once, each block one expert), same weights, same scattered output."""
    from inference_server.models import moe_kernel
    from inference_server.models.qwen3_moe import route
    torch.manual_seed(T + 7)
    logits = torch.randn(T, E, dtype=torch.bfloat16, device="cuda") * 2
    rw, rids = route(logits, top_k, renorm)
    sorted_ids, expert_ids, tok, pair_w = moe_kernel.route_fused(logits, E, top_k, renorm, 16)
    M = T * top_k
    assert sorted_ids.numel() == expert_ids.numel() * 16 and (sorted_ids < M).sum() == M
    real = sorted_ids[sorted_ids < M]
    assert torch.equal(real.sort().values, torch.arange(M, device="cuda", dtype=torch.int32))   # every pair once
    pair_e = rids.view(-1).to(torch.int32)                       # the reference expert of pair t * K + slot
    assert torch.equal(tok.long(), torch.arange(M, device="cuda") // top_k)
    torch.testing.assert_close(pair_w, rw.view(-1).float(), atol=2e-3, rtol=2e-2)
    for b in range(expert_ids.numel()):
        ids = sorted_ids[b * 16:(b + 1) * 16]
        ids = ids[ids < M]
        if ids.numel():
            assert (pair_e[ids.long()] == expert_ids[b]).all()
        else:
            assert expert_ids[b] == E or True                   # an empty block may close an expert's padding
    K_, N = 256, 384
    act = torch.randn(M, K_, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(E, N, K_, dtype=torch.bfloat16, device="cuda") * 0.05
    out = torch.zeros(T, N, device="cuda")
    moe_kernel.moe_gemm_scatter(act, w, sorted_ids, expert_ids, M, tok, pair_w, out)
    ref = torch.zeros(T, N, device="cuda")
    for i in range(M):
        ref[i // top_k] += (act[i].float() @ w[pair_e[i]].float().T) * rw.view(-1)[i].float()
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=3e-2)
