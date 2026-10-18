"""The decode-step megakernel against the torch decode forward on the real model (CUDA, nvcc, the 60 GB weights:
MEGAKERNEL_TEST=1). One whole step, B in (1, 3), contexts under and over one 256-position chunk; the logits' argmax
must agree on every row, and the new token's K/V written into every layer's pool must match. bf16 experts are
compared tightly; fp8 experts (the default) are a different quantisation, so only the argmax is required there."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("MEGAKERNEL_TEST") != "1",
                                reason="needs CUDA, nvcc and MEGAKERNEL_TEST=1 (loads the real model)")


@pytest.fixture(scope="module")
def backend():
    os.environ["CUSTOM_BACKEND_CUDA_GRAPH"] = "0"
    os.environ["MEGAKERNEL_FP8"] = "1"                 # the fp8 tables are built before the pools take the free memory
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = CustomTorchBackend("cuda", name)
    b.load_model(name)
    return b


@pytest.fixture(scope="module")
def bf16_model(backend):
    from inference_server.models import megakernel as mk
    m = mk.Model(backend.model, backend.device, fp8=False)               # points at the bf16 weights, no copy
    m.bind_pools(backend.pools)
    return m


def _state(b, prompts):
    from inference_server.models.paged_kv_cache import BatchedDecodeState
    from inference_server.sampling import GREEDY
    rows = [b.prefill_chunk_batch([(None, p, True, GREEDY)])[0] for p in prompts]
    st = BatchedDecodeState(b.pools, b.device)
    for kv, _, _ in rows:
        st.add_row(kv)
    st.prepare_step()
    return st


def _step(backend, mdl, prompts, seed):
    """(kernel logits, torch logits, kernel state, torch state) for one decode step after the prompts."""
    from inference_server.models import megakernel as mk
    torch.manual_seed(seed)
    tokens = torch.randint(1000, 30000, (len(prompts),), device="cuda")
    sa = _state(backend, prompts)
    ref = backend.model(tokens.view(-1, 1), position_ids=sa.seq_lens.view(-1, 1), paged_ctx=sa)[:, 0]
    sb = _state(backend, prompts)
    got = mk.decode_step(mdl, tokens, sb.seq_lens.to(torch.int32), sb.bt.to(torch.int32))
    torch.cuda.synchronize()
    return got, ref, sb, sa


def _kv_match(backend, sb, sa, B):
    for L in (0, 23, 47):
        for r in range(B):
            p = int(sa.seq_lens[r])
            ba, bb = sa.bt[sa._li[L]][r, p // 16].item(), sb.bt[sb._li[L]][r, p // 16].item()
            pool = backend.pools[L]
            torch.testing.assert_close(pool.k[bb, :, p % 16].float(), pool.k[ba, :, p % 16].float(), atol=2e-2, rtol=2e-2)
            assert torch.equal(pool.v[bb, :, p % 16], pool.v[ba, :, p % 16])


@pytest.mark.parametrize("lens", [(37,), (37, 16, 300), (1000,)])
def test_bf16_step_matches_torch(backend, bf16_model, lens):
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    got, ref, sb, sa = _step(backend, bf16_model, prompts, seed=len(lens))
    assert torch.equal(got.float().argmax(-1), ref.float().argmax(-1))
    assert (got.float() - ref.float()).abs().mean() < 0.05, (got.float() - ref.float()).abs().max()
    _kv_match(backend, sb, sa, len(lens))


@pytest.mark.parametrize("lens", [(37,), (37, 16, 300)])
def test_fp8_step_agrees_on_the_argmax(backend, lens):
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    got, ref, sb, sa = _step(backend, backend._mega, prompts, seed=10 + len(lens))
    assert torch.equal(got.float().argmax(-1), ref.float().argmax(-1))
    _kv_match(backend, sb, sa, len(lens))
