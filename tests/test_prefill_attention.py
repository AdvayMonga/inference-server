"""SDPA prefill over a paged prefix + suffix matches the Triton paged prefill kernel, one call for a whole wave (CUDA)."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
Hq, Hkv, D, BS = 32, 4, 128, 16


def _wave(rows):
    """rows: [(prefix, suffix)] -> (q, k_suf, v_suf, ctx, reference via the Triton kernel)."""
    from inference_server.models.paged_attention_kernel import paged_prefill_attention
    torch.manual_seed(len(rows))
    B = len(rows)
    S = max(s for _, s in rows)
    nbs = [(p + s + BS - 1) // BS for p, s in rows]
    nb_tot = sum(nbs)
    pool_k = torch.zeros(nb_tot + 1, Hkv, BS, D, dtype=torch.bfloat16, device="cuda")
    pool_v = torch.zeros_like(pool_k)
    ids_all = torch.randperm(nb_tot, device="cuda") + 1                       # block 0 is scratch
    q = torch.zeros(B, Hq, S, D, dtype=torch.bfloat16, device="cuda")
    k_suf = torch.zeros(B, Hkv, S, D, dtype=torch.bfloat16, device="cuda")
    v_suf = torch.zeros_like(k_suf)
    tables, off = [], 0
    maxb = max(nbs)
    for r, (p, s) in enumerate(rows):
        ids = ids_all[off:off + nbs[r]]; off += nbs[r]
        kall = torch.randn(Hkv, nbs[r] * BS, D, dtype=torch.bfloat16, device="cuda")
        vall = torch.randn_like(kall)
        pool_k[ids] = kall.view(Hkv, nbs[r], BS, D).permute(1, 0, 2, 3)
        pool_v[ids] = vall.view(Hkv, nbs[r], BS, D).permute(1, 0, 2, 3)
        q[r, :, :s] = torch.randn(Hq, s, D, dtype=torch.bfloat16, device="cuda")
        k_suf[r, :, :s], v_suf[r, :, :s] = kall[:, p:p + s], vall[:, p:p + s]
        tables.append(torch.cat([ids, torch.zeros(maxb - nbs[r], dtype=ids.dtype, device="cuda")]))
    bt = torch.stack(tables).to(torch.int32)
    pre = torch.tensor([p for p, _ in rows], dtype=torch.int32, device="cuda")
    suf = torch.tensor([s for _, s in rows], dtype=torch.int32, device="cuda")
    ctx = SimpleNamespace(pools=[SimpleNamespace(k=pool_k, v=pool_v)], _prefix=[p for p, _ in rows],
                          _suffix=[s for _, s in rows], prefix_lens=pre, suffix_lens=suf,
                          block_table_tensor=lambda layer: bt)
    ref = paged_prefill_attention(q.transpose(1, 2), pool_k, pool_v, bt, pre, suf, scale=D ** -0.5)
    return q, k_suf, v_suf, ctx, ref


@pytest.mark.parametrize("rows", [[(0, 300)], [(40, 7)], [(160, 500)], [(33, 1)],
                                  [(0, 300), (0, 120), (0, 7)],                       # a cache-miss wave
                                  [(48, 200), (0, 333), (17, 1), (160, 64)]])          # mixed prefixes, ragged
@pytest.mark.parametrize("batched_max", [10 ** 9, 0])                           # the wave path, then per row
def test_sdpa_prefill_matches_the_paged_kernel(rows, batched_max, monkeypatch):
    from inference_server.models import prefill_attention
    monkeypatch.setattr(prefill_attention, "BATCHED_MAX_TOKENS", batched_max)
    q, k, v, ctx, ref = _wave(rows)
    got = prefill_attention.sdpa_prefill(q, k, v, ctx, 0, D ** -0.5)
    for r, (_, s) in enumerate(rows):                                            # padding rows are unspecified
        torch.testing.assert_close(got[r, :s].float(), ref[r, :s].float(), atol=2e-2, rtol=2e-2)
