"""The prefill stations (models/stations.py PREFILL): on CPU, their reads and writes chain, overrides go to the plan
of their own station set, the capture of a "p_" stretch reads PREFILL, and the reference attention is causal attention
at the end of each row's keys; on a GPU with the real model (PREFILL_STATIONS_TEST=1), a prefill wave through the
stations is bit-identical to the module path (K/V written and the next token), padded and packed."""

import os

import pytest
import torch

from inference_server.models import region_capture, stations


def test_prefill_reads_and_writes_chain():
    given = {"h", "cos", "sin", "tokens", "paged_ctx", "layer", "cfg", "k_cache", "v_cache"}
    have = set(given)
    for s in stations.PREFILL:
        missing = [r for r in s.reads if r not in have and not r.startswith("w.")]
        assert not missing, (s.name, missing)
        have |= set(s.writes)
    assert [s.name for s in stations.reference(stations.PREFILL)] == [s.name for s in stations.PREFILL]
    assert all(s.name.startswith("p_") for s in stations.PREFILL)
    assert stations.stations_of("p_moe") is stations.PREFILL and stations.stations_of("router") is stations.DECODE


def test_overrides_go_to_their_own_station_set(monkeypatch):
    fn = lambda c: {"h": c["h"]}                     # noqa: E731
    monkeypatch.setattr(stations, "overrides", lambda package="": {("p_moe", "p_moe"): fn})
    stations.plan.cache_clear()
    stations.plan_prefill.cache_clear()
    try:
        assert stations.plan() == tuple((s.name, s.name, s.fn) for s in stations.DECODE)
        assert [k[:2] for k in stations.plan_prefill()][-1] == ("p_moe", "p_moe")
        assert stations._overrides_of(stations.PREFILL) == {("p_moe", "p_moe"): fn}
    finally:
        stations.plan.cache_clear()
        stations.plan_prefill.cache_clear()


def test_a_prefill_stretch_is_captured_from_the_prefill_stations(tmp_path):
    cap = region_capture.Capture("p_attention", "p_attention", None, tmp_path)
    assert cap.prefill and cap.inputs == ("q", "k_cache", "v_cache", *stations.FI, "cfg") and cap.outputs == ("attn",)
    assert not region_capture.Capture("attention", "attention", None, tmp_path).prefill
    blocks = region_capture._touched({"fi_kv_indices": torch.tensor([7, 3, 3, 9])})
    assert blocks.tolist() == [3, 7, 9]


def test_the_reference_prefill_attention_is_causal_at_the_end_of_each_rows_keys():
    torch.manual_seed(0)
    Hq, Hkv, D, bs = 4, 2, 8, 4
    prefix, suffix = [3, 0], [2, 5]                  # row 0: 3 cached keys then 2 new; row 1: 5 new
    blocks = [[5, 1], [0, 2]]
    pool_k, pool_v = (torch.randn(8, Hkv, bs, D) for _ in range(2))
    q = torch.randn(1, Hq, sum(suffix), D)           # a packed wave: one row of every new token
    qo, kvp, last = [0, 2, 7], [0, 2, 4], [1, 1]
    out = stations._ref_prefill_attention(q, pool_k, pool_v, torch.tensor(qo), torch.tensor(kvp),
                                          torch.tensor([b for row in blocks for b in row]), torch.tensor(last), None,
                                          0.5)
    for r, (p, s) in enumerate(zip(prefix, suffix)):
        L = p + s
        k = torch.cat([pool_k[b] for b in blocks[r]], dim=1)[:, :L].repeat_interleave(2, 0)
        v = torch.cat([pool_v[b] for b in blocks[r]], dim=1)[:, :L].repeat_interleave(2, 0)
        for i in range(s):                           # the i-th new token sees keys 0 .. p + i
            qi = q[0, :, qo[r] + i]
            w = torch.softmax((k[:, :p + i + 1] @ qi[:, :, None])[..., 0] * 0.5, -1)
            want = (w[:, :, None] * v[:, :p + i + 1]).sum(1)
            torch.testing.assert_close(out[0, qo[r] + i], want, atol=1e-5, rtol=1e-5)


GPU = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("PREFILL_STATIONS_TEST") != "1",
                         reason="needs CUDA and PREFILL_STATIONS_TEST=1 (loads the real model)")


@pytest.fixture(scope="module")
def backend():
    os.environ.update(CUSTOM_BACKEND_CUDA_GRAPH="0", CUSTOM_BACKEND_BLOCKS="8192")
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = CustomTorchBackend("cuda", name)
    b.load_model(name)
    return b


def _wave(b, prompts):
    """One prefill wave: each row's next token and every layer's K/V of its blocks."""
    from inference_server.sampling import GREEDY
    rows = b.prefill_chunk_batch([(None, p, True, GREEDY) for p in prompts])
    torch.cuda.synchronize()
    kv = [torch.cat([torch.stack([pool.k[c.blocks], pool.v[c.blocks]]) for pool in b.pools]) for c, _, _ in rows]
    return [int(t) for _, t, _ in rows], kv


@GPU
@pytest.mark.parametrize("lengths", [(37, 37), (40, 300, 2049)])
def test_a_wave_through_the_stations_is_bit_identical(backend, lengths, monkeypatch):
    g = torch.Generator().manual_seed(len(lengths))
    prompts = [torch.randint(1000, 30000, (n,), generator=g).tolist() for n in lengths]
    want = _wave(backend, prompts)
    monkeypatch.setattr(stations, "prefill_applies", lambda *a: True)
    got = _wave(backend, prompts)
    assert got[0] == want[0]
    for a, b in zip(got[1], want[1]):
        assert torch.equal(a, b)


@GPU
def test_the_reference_moe_is_close_to_the_station(backend):
    """The fp32 reference of p_moe at the engine's experts against the station, on one layer's weights."""
    torch.manual_seed(0)
    layer = backend.model.model.layers[0]
    c = {**stations._weights(layer), "cfg": stations.layer_config(backend.model.model), "tokens": None}
    H = backend.model.model.hidden_size
    for T in (16, 300, 1500):                        # Triton, Triton W8A8 and the torch grouped GEMM
        c["n"], c["h"] = (torch.randn(1, T, H, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        got = stations._p_moe(c)["h"].float()
        ref = stations._r_p_moe({**c, **{k: v.float() for k, v in c.items()
                                         if isinstance(v, torch.Tensor) and k in ("n", "h")}})["h"]
        err = (got - ref).abs().max().item()
        assert err < 0.25 * ref.abs().max().item(), (T, err)
