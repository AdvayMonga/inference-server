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
    assert torch.equal(sorted_ids, ref_ids) and torch.equal(expert_ids, ref_experts)
    assert torch.equal(tok.long(), order // top_k) and torch.equal(pair_w, weights.view(-1)[order])
    act = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.05
    offs = torch.searchsorted(sorted_e, torch.arange(E, device="cuda"), right=True).to(torch.int32)
    y = torch._grouped_mm(act, w.transpose(-2, -1), offs=offs).float() * pair_w[:, None]
    ref = torch.zeros(T, N, device="cuda").index_add_(0, order // top_k, y)
    out = torch.zeros(T, N, device="cuda")
    moe_kernel.moe_gemm_scatter(act, w, sorted_ids, expert_ids, M, tok, pair_w, out)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=2e-2)
