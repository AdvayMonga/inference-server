"""The packed prefill forward: K/V land where the padded wave put them, and the packed attention matches the padded one."""

from types import SimpleNamespace

import torch

from inference_server.backends.custom_torch_backend import _PackedCtx, _PrefillCtx
from inference_server.models.paged_kv_cache import BlockPool, PagedKVCache
from inference_server.models.prefill_attention import sdpa_prefill, sdpa_prefill_packed

DEV = torch.device("cpu")
BS, H, D = 4, 2, 8


def _pools(n=2):
    return [BlockPool(64, BS, H, D, torch.float32, DEV) for _ in range(n)]


def _rows(pools, prefixes):
    caches = []
    for P in prefixes:
        kv = PagedKVCache(pools=pools)
        for L in range(len(pools)):
            for pos in range(P):
                kv.append(L, torch.full((1, H, 1, D), float(pos)), torch.full((1, H, 1, D), -float(pos)))
        caches.append(kv)
    return caches


def test_packed_context_writes_the_same_kv_as_the_padded_wave():
    torch.manual_seed(0)
    prefixes, suffixes = [6, 0, 11], [5, 3, 1]
    pools_a, pools_b = _pools(), _pools()
    ca, cb = _rows(pools_a, prefixes), _rows(pools_b, prefixes)
    k_rows = [torch.randn(H, n, D) for n in suffixes]
    v_rows = [torch.randn(H, n, D) for n in suffixes]
    S = max(suffixes)
    k_pad = torch.zeros(3, H, S, D)
    v_pad = torch.zeros(3, H, S, D)
    for r, n in enumerate(suffixes):
        k_pad[r, :, :n], v_pad[r, :, :n] = k_rows[r], v_rows[r]
    k_packed = torch.cat(k_rows, dim=1).unsqueeze(0)              # [1, H, T, D]
    v_packed = torch.cat(v_rows, dim=1).unsqueeze(0)
    scratch_a = [p.alloc() for p in pools_a]                   # the padded wave parks its padding tokens here
    scratch_b = [p.alloc() for p in pools_b]
    padded = _PrefillCtx(ca, pools_a, prefixes, suffixes, DEV, scratch=scratch_a)
    packed = _PackedCtx(cb, pools_b, prefixes, suffixes, DEV, scratch=scratch_b)
    for L in range(2):
        padded.append(L, k_pad, v_pad)
        packed.append(L, k_packed, v_packed)
    for L in range(2):
        real = torch.ones(64, dtype=torch.bool)
        real[scratch_a[L]] = False
        assert torch.equal(pools_a[L].k[real], pools_b[L].k[real]) and torch.equal(pools_a[L].v[real], pools_b[L].v[real])
        assert torch.equal(pools_b[L].k[scratch_b[L]], torch.zeros(H, BS, D))   # packed never touches the scratch block
        assert [c.seq_lens[L] for c in ca] == [c.seq_lens[L] for c in cb] == [11, 3, 12]
    assert packed.tokens is None and packed._pos.tolist() == [6, 7, 8, 9, 10, 0, 1, 2, 11]


def test_packed_attention_matches_the_padded_reference():
    torch.manual_seed(1)
    Hq = 4
    prefixes, suffixes = [6, 0, 11], [5, 3, 1]
    pools = _pools(1)
    caches = _rows(pools, prefixes)
    ctx_pad = _PrefillCtx(caches, pools, prefixes, suffixes, DEV)
    S = max(suffixes)
    q_pad = torch.zeros(3, Hq, S, D)
    k_pad = torch.zeros(3, H, S, D)
    v_pad = torch.zeros(3, H, S, D)
    q_rows, k_rows, v_rows = [], [], []
    for r, n in enumerate(suffixes):
        q_rows.append(torch.randn(Hq, n, D)); k_rows.append(torch.randn(H, n, D)); v_rows.append(torch.randn(H, n, D))
        q_pad[r, :, :n], k_pad[r, :, :n], v_pad[r, :, :n] = q_rows[r], k_rows[r], v_rows[r]
    ctx_pad.append(0, k_pad, v_pad)                                 # the suffix K/V must be in the pool for both paths
    ref = sdpa_prefill(q_pad, k_pad, v_pad, ctx_pad, 0, D ** -0.5)  # [3, S, Hq, D]
    ctx_packed = SimpleNamespace(pools=pools, _prefix=prefixes, _suffix=suffixes,
                                 block_table_tensor=lambda layer: ctx_pad.block_table_tensor(layer))
    got = sdpa_prefill_packed(torch.cat(q_rows, 1).unsqueeze(0), torch.cat(k_rows, 1).unsqueeze(0),
                              torch.cat(v_rows, 1).unsqueeze(0), ctx_packed, 0, D ** -0.5)[0]   # [T, Hq, D]
    s = 0
    for r, n in enumerate(suffixes):
        torch.testing.assert_close(got[s:s + n], ref[r, :n], atol=1e-5, rtol=1e-5)
        s += n
