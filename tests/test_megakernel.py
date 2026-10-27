"""The decode-step megakernel against the torch decode forward on the real model (CUDA, nvcc, the 60 GB weights:
MEGAKERNEL_TEST=1). One whole step, B in (1, 3), contexts under and over one 256-position chunk; the logits' argmax
must agree on every row, and the new token's K/V written into every layer's pool must match. bf16 experts are
compared tightly; fp8 experts (the default) are a different quantisation, so only the argmax is required there.
Measured on the H200 (2026-10-20), main's kernel and this one alike: the torch forward and the kernel round every
GEMM differently, so the argmax flips on near-ties (the kernel's pick within 0.125 of the top logit, 1-2 bf16 steps
at |logit| ~10), the mean logit error reaches 5% of the mean |logit|, and K/V deeper than layer 0 differ by up to
~25% of their norm; the assertions are those bounds, which a wrong head, position or block still breaks."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("MEGAKERNEL_TEST") != "1",
                                reason="needs CUDA, nvcc and MEGAKERNEL_TEST=1 (loads the real model)")


@pytest.fixture(scope="module")
def backend():
    os.environ["CUSTOM_BACKEND_CUDA_GRAPH"] = "0"
    os.environ["CUSTOM_BACKEND_QUANT"] = "bf16"        # the torch reference forward runs bf16 experts
    os.environ["MEGAKERNEL_FP8"] = "0"
    os.environ["CUSTOM_BACKEND_BLOCKS"] = "8192"       # small pools: room for the fp8 copies built below
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = CustomTorchBackend("cuda", name)
    b.load_model(name)
    return b


@pytest.fixture(scope="module")
def fp8_model(backend):
    from inference_server.models import megakernel as mk
    m = mk.Model(backend.model, backend.device, fp8=True)                # fp8 copies of the experts
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
    got, nxt = mk.decode_step(mdl, tokens, sb.seq_lens, sb.bt)
    torch.cuda.synchronize()
    assert torch.equal(nxt, got.float().argmax(-1)), "the kernel's greedy token is its own logits' argmax"
    return got, ref, sb, sa


def picks_agree(got, ref, slack=0.25):
    """Each row's kernel token is the reference's argmax or within `slack` of its top logit (a near-tie)."""
    got, ref = got.float(), ref.float()
    deficit = ref.max(-1).values - ref.gather(-1, got.argmax(-1, keepdim=True))[:, 0]
    return bool((deficit <= slack).all()), deficit.tolist()


def _kv_match(backend, sb, sa, B):
    """Layer 0's new K/V depend on the embedding alone and agree to rounding; deeper layers carry every earlier
    layer's rounding, so they are compared by relative norm."""
    for L in (0, 23, 47):
        for r in range(B):
            p = int(sa.seq_lens[r])
            ba, bb = sa.bt[r, p // 16].item(), sb.bt[r, p // 16].item()
            pool = backend.pools[L]
            for x in (pool.k, pool.v):
                a, b = x[ba, :, p % 16].float(), x[bb, :, p % 16].float()
                if L == 0:
                    torch.testing.assert_close(b, a, atol=2e-2, rtol=2e-2)
                else:
                    err = ((b - a).norm(dim=-1) / a.norm(dim=-1)).max().item()
                    assert err < 0.35, (L, r, err)


@pytest.mark.parametrize("lens", [(37,), (37, 16, 300), (1000,)])
def test_bf16_step_matches_torch(backend, lens):
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    got, ref, sb, sa = _step(backend, backend._mega, prompts, seed=len(lens))
    ok, deficit = picks_agree(got, ref)
    assert ok, deficit
    d, mag = (got.float() - ref.float()).abs(), ref.float().abs().mean()
    assert d.mean() < 0.1 * mag, (d.mean(), d.max(), mag)
    _kv_match(backend, sb, sa, len(lens))


@pytest.mark.parametrize("lens", [(37,), (37, 16, 300)])
def test_fp8_step_agrees_on_the_argmax(backend, fp8_model, lens):
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    got, ref, sb, sa = _step(backend, fp8_model, prompts, seed=10 + len(lens))
    ok, deficit = picks_agree(got, ref, slack=0.5)
    assert ok, deficit
    _kv_match(backend, sb, sa, len(lens))


@pytest.mark.parametrize("lens", [(37,), (37, 300), (37, 16, 300, 1000), (37, 16, 300, 1000, 5, 64, 200, 129)])
@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_tma_streamed_weights_give_the_register_paths_tokens(backend, fp8_model, lens, which):
    """MEGAKERNEL_TMA: the same step with every GEMV phase's weights streamed through the shared-memory ring (batch
    <= MEGAKERNEL_TMA_ROWS); only the summation order differs, so the picks agree to a near-tie and the logits are close."""
    from inference_server.models import megakernel as mk
    mdl = backend._mega if which == "bf16" else fp8_model
    prompts = [torch.randint(1000, 30000, (n,)).tolist() for n in lens]
    outs = []
    for tma in (False, True):
        mk.TMA = tma
        try:
            got, _, _, _ = _step(backend, mdl, prompts, seed=40 + len(lens))
        finally:
            mk.TMA = False
        outs.append(got.float())
    ok, deficit = picks_agree(outs[1], outs[0])
    assert ok, deficit
    assert (outs[0] - outs[1]).abs().mean() < 0.1 * outs[0].abs().mean()


def test_verify_rows_of_one_sequence_match_the_torch_forward(backend):
    """A verify step: rows = consecutive positions of one sequence (one block table). Each row reads the K/V the rows
    before it write in the same launch (`shared_kv`), and `aux` holds each row's residual entering the 3 layers."""
    from inference_server.models import megakernel as mk
    from inference_server.sampling import GREEDY
    torch.manual_seed(7)
    n, S, layers = 300, 4, (2, 24, 45)
    prompt = torch.randint(1000, 30000, (n,)).tolist()
    toks = torch.randint(1000, 30000, (S,), device="cuda")
    m = backend.model.model
    kv_ref = backend.prefill_chunk_batch([(None, prompt, True, GREEDY)])[0][0]
    m.aux_layers = layers
    try:
        ref = backend.model(toks.view(1, -1), kv_cache=kv_ref)[0]
        aux_ref = m.aux_out[0].view(S, 3, -1).float()
    finally:
        m.aux_layers = ()
    kv_ref.free_all()
    st = _state(backend, [prompt])
    st.reserve(0, S)
    bt = st.bt[0:1].expand(S, -1).contiguous()
    aux = torch.empty(S, 3, 2048, dtype=torch.bfloat16, device="cuda")
    got, _ = mk.decode_step(backend._mega, toks, torch.arange(n, n + S, device="cuda"), bt, aux=aux, aux_layers=layers,
                            shared_kv=True)
    torch.cuda.synchronize()
    ok, deficit = picks_agree(got, ref)
    assert ok, deficit
    d, mag = (got.float() - ref.float()).abs(), ref.float().abs().mean()
    assert d.mean() < 0.1 * mag, (d.mean(), d.max(), mag)
    err = (aux.float() - aux_ref).norm(dim=-1) / aux_ref.norm(dim=-1)
    assert err[:, 0].max() < 0.05 and err.max() < 0.35, err.tolist()
