"""The decode-step megakernel with fp8 KV pools (KV_CACHE_DTYPE=fp8) against the torch decode forward reading the
same pools (CUDA, nvcc, the real model: MEGAKERNEL_TEST=1; its own process, it loads the model with fp8 pools)."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("MEGAKERNEL_TEST") != "1",
                                reason="needs CUDA, nvcc and MEGAKERNEL_TEST=1 (loads the real model)")


@pytest.fixture(scope="module")
def backend():
    os.environ.update(CUSTOM_BACKEND_CUDA_GRAPH="0", CUSTOM_BACKEND_QUANT="bf16", MEGAKERNEL_FP8="0",
                      CUSTOM_BACKEND_BLOCKS="8192", KV_CACHE_DTYPE="fp8")
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = CustomTorchBackend("cuda", name)
    b.load_model(name)
    assert b.pools[0].k.dtype == torch.float8_e4m3fn and b._mega.kv8
    return b


@pytest.mark.parametrize("lens", [(37,), (37, 16, 300), (1000,)])
def test_fp8_kv_step_matches_torch_on_the_same_pools(backend, lens):
    from tests.test_megakernel import _step, picks_agree
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    got, ref, sb, sa = _step(backend, backend._mega, prompts, seed=20 + len(lens))
    ok, deficit = picks_agree(got, ref)                         # near-ties as with bf16 pools (test_megakernel)
    assert ok, deficit
    d, mag = (got.float() - ref.float()).abs(), ref.float().abs().mean()
    assert d.mean() < 0.1 * mag, (d.mean(), d.max(), mag)
    for r in range(len(lens)):                                  # layer 0's new K/V, as stored, agree to fp8 rounding
        p = int(sa.seq_lens[r])
        ba, bb = sa.bt[r, p // 16].item(), sb.bt[r, p // 16].item()
        for pool in (backend.pools[0].k, backend.pools[0].v):
            torch.testing.assert_close(pool[bb, :, p % 16].float(), pool[ba, :, p % 16].float(), atol=0.1, rtol=0.13)
