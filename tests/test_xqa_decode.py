"""XQA decode attention (fp8 K/V pools, Hopper) against the reference over rows of very different lengths."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
                                reason="needs a Hopper GPU")


def test_xqa_matches_reference_over_mixed_lengths():
    from inference_server.models import paged_attention_kernel as pak
    from inference_server.models.stations import _ref_paged_decode_attention
    torch.manual_seed(0)
    lens = [11883, 3644, 3276, 1, 6243, 17, 3445, 129]
    hkv, hq, d, bs, cols = 4, 32, 128, 16, 2048
    n_blocks = sum((L + bs - 1) // bs for L in lens) + 1
    k_pool = (torch.randn(n_blocks, hkv, bs, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    v_pool = (torch.randn(n_blocks, hkv, bs, d, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    bt = torch.zeros(len(lens), cols, dtype=torch.int32, device="cuda")
    perm, i = torch.randperm(n_blocks - 1, device="cuda").int() + 1, 0
    for r, L in enumerate(lens):
        nb = (L + bs - 1) // bs
        bt[r, :nb] = perm[i:i + nb]
        i += nb
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(len(lens), hq, d, dtype=torch.bfloat16, device="cuda")
    scale = d ** -0.5
    assert pak._xqa_applies(q, k_pool, bt, seq_lens)
    out = pak.paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale)
    ref = _ref_paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
