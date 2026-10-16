"""fp8 experts: quantization error bounded, the MoE block stays close to bf16, and on CUDA the fp8 grouped GEMM
matches the dequantized reference path."""

from __future__ import annotations

import copy

import pytest
import torch

from inference_server.models.qwen3_moe import FP8, Qwen3MoeForCausalLM, quantize_rowwise


def test_rowwise_fp8_round_trip_is_within_e4m3_precision():
    x = torch.randn(6, 300, dtype=torch.bfloat16) * torch.tensor([[1.0], [10.0], [0.01], [100.0], [1e-3], [3.0]])
    q, s = quantize_rowwise(x)
    assert q.dtype == FP8 and s.dtype == torch.float32 and s.shape == (6,)
    back = q.to(torch.float32) * s[:, None]
    err = (back - x.float()).abs().amax(dim=1)
    assert torch.all(err <= x.float().abs().amax(dim=1) * 2 ** -4)        # 3 mantissa bits: half a step at the max


def test_fp8_experts_free_the_bf16_weights_and_stay_close(tiny_qwen3_moe_dir):
    ours = Qwen3MoeForCausalLM.from_hf(tiny_qwen3_moe_dir).eval()
    fp8 = copy.deepcopy(ours)
    for layer in fp8.model.layers:
        layer.mlp.experts.quantize_fp8()
    ex = fp8.model.layers[0].mlp.experts
    assert ex.fp8 and "gate_up_proj" not in dict(ex.named_parameters()) and ex.gate_up_proj_fp8.dtype == FP8
    assert ex.gate_up_proj_scale.shape == ex.gate_up_proj_fp8.shape[:2]
    x = torch.randn(2, 9, 64, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref, got = ours.model.layers[0].mlp(x), fp8.model.layers[0].mlp(x)
    rel = (got.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.08, rel
    ids = torch.tensor([[5, 17, 300, 42, 8, 99, 123, 7, 55]])
    with torch.no_grad():
        a, b = ours(ids).argmax(-1), fp8(ids).argmax(-1)
    assert (a == b).float().mean() >= 0.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 grouped GEMM needs CUDA (sm89+)")
def test_cuda_fp8_grouped_gemm_matches_the_reference_path():
    from inference_server.models.qwen3_moe import Qwen3Experts
    torch.manual_seed(0)
    ex = Qwen3Experts(4, 256, 128).cuda()
    with torch.no_grad():
        ex.gate_up_proj.normal_(); ex.down_proj.normal_()
    ex.quantize_fp8()
    a = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([16, 40, 40, 64], dtype=torch.int32, device="cuda")   # an empty group too
    got = ex.gate_up(a, offs)
    cpu = copy.deepcopy(ex).cpu()
    ref = cpu.gate_up(a.cpu(), offs.cpu())
    torch.testing.assert_close(got.cpu().float(), ref.float(), atol=0.1 * ref.abs().max().item(), rtol=0.05)
