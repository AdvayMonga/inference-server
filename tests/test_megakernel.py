"""The decode-layer megakernel against the torch decode layer on the real model (CUDA, nvcc, the 60 GB weights:
MEGAKERNEL_TEST=1). One decode step per layer tested, B in (1, 3); the new token's K/V written into the pool too."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("MEGAKERNEL_TEST") != "1",
                                reason="needs CUDA, nvcc and MEGAKERNEL_TEST=1 (loads the real model)")


@pytest.fixture(scope="module")
def backend():
    os.environ["CUSTOM_BACKEND_CUDA_GRAPH"] = "0"
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = CustomTorchBackend("cuda", name)
    b.load_model(name)
    return b


def _state(b, prompts):
    from inference_server.models.paged_kv_cache import BatchedDecodeState
    from inference_server.sampling import GREEDY
    rows = [b.prefill_chunk_batch([(None, p, True, GREEDY)])[0] for p in prompts]
    st = BatchedDecodeState(b.pools, b.device)
    for kv, _, _ in rows:
        st.add_row(kv)
    st.prepare_step()
    return st


@pytest.mark.parametrize("L", [0, 24, 47])
@pytest.mark.parametrize("lens", [(37,), (37, 16, 300)])
def test_layer_matches_torch_and_writes_the_kv(backend, L, lens):
    from inference_server.models import megakernel as mk
    assert mk.available()
    model = backend.model.model
    torch.manual_seed(L)
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    B = len(prompts)
    h = torch.randn(B, 2048, dtype=torch.bfloat16, device="cuda")
    layer = model.layers[L]
    sa = _state(backend, prompts)
    pos = sa.seq_lens.clone()
    cos, sin = (t.to(h.dtype) for t in model.rope(pos.unsqueeze(1)))
    ref, _ = layer(h.unsqueeze(1), cos, sin, paged_ctx=sa, kv_layer_idx=L)
    sb = _state(backend, prompts)
    got = mk.decode_layer(h, sb.seq_lens.to(torch.int32), sb.bt[sb._li[L]].to(torch.int32), backend.pools[L].k,
                          backend.pools[L].v, mk.LayerWeights(layer, model.rope.inv_freq), mk.Scratch(backend.device))
    torch.testing.assert_close(got.float(), ref[:, 0].float(), atol=6e-2, rtol=3e-2)
    for r in range(B):
        p = int(pos[r])
        ba, bb = sa.bt[sa._li[L]][r, p // 16].item(), sb.bt[sb._li[L]][r, p // 16].item()
        torch.testing.assert_close(backend.pools[L].k[bb, :, p % 16].float(), backend.pools[L].k[ba, :, p % 16].float(),
                                   atol=2e-2, rtol=2e-2)
        assert torch.equal(backend.pools[L].v[bb, :, p % 16], backend.pools[L].v[ba, :, p % 16])
