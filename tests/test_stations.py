"""The decode stations (models/stations.py) on CPU, the Triton kernels replaced by the package's torch versions of the
same math: the station list's reads and writes chain, the stations compute the layer's decode forward, override
resolution (a no-op override changes nothing; bad ones are refused), and `reference()` (pure, the real routing layout,
close to the stations in fp32). Bit-identity with the real kernels is a GPU check."""

import sys
import types

import pytest
import torch

from inference_server.models import fused, stations
from inference_server.models.paged_kv_cache import step_inputs
from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM

BS = 4


@pytest.fixture
def torch_kernels(monkeypatch):
    """The fused and Triton kernels the stations call, as the package's torch versions (stations._ref_*)."""
    for name in ("rmsnorm", "add_rmsnorm", "rmsnorm_rope", "silu_mul", "add_cast"):
        monkeypatch.setattr(fused, name, getattr(stations, f"_ref_{name}"))
    mk = types.SimpleNamespace(route_fused=stations._ref_route, route_multi=stations._ref_route,
                               moe_gemm=stations._ref_moe_gemm, moe_gemm_scatter=stations._ref_moe_gemm_scatter)
    monkeypatch.setitem(sys.modules, "inference_server.models.moe_kernel", mk)
    monkeypatch.setattr(sys.modules["inference_server.models"], "moe_kernel", mk, raising=False)
    pak = types.SimpleNamespace(paged_decode_attention=stations._ref_paged_decode_attention)
    monkeypatch.setitem(sys.modules, "inference_server.models.paged_attention_kernel", pak)


@pytest.fixture
def model(tiny_qwen3_moe_dir):
    m = Qwen3MoeForCausalLM.from_hf(tiny_qwen3_moe_dir).eval()
    for layer in m.model.layers:
        layer.self_attn.fuse_qkv()
    return m


def _decode_case(model, seq_lens=(5, 9)):
    """A decode step over rows with K/V already in paged pools: (h, cos, sin, ctx, per-layer past K/V per row)."""
    torch.manual_seed(0)
    inner = model.model
    attn = inner.layers[0].self_attn
    B, Hkv, D = len(seq_lens), attn.num_kv_heads, attn.head_dim
    cols = max((L + 1 + BS - 1) // BS for L in seq_lens)
    bt = torch.arange(B * cols).view(B, cols).flip(1)                   # any block ids: not in order
    sl = torch.tensor(seq_lens)
    pools, past = [], []
    for _ in inner.layers:
        pool = types.SimpleNamespace(k=torch.zeros(B * cols, Hkv, BS, D, dtype=torch.bfloat16))
        pool.v = torch.zeros_like(pool.k)
        rows = []
        for r, L in enumerate(seq_lens):
            k, v = (torch.randn(Hkv, L, D, dtype=torch.bfloat16) for _ in range(2))
            for t in range(L):
                pool.k[bt[r, t // BS], :, t % BS], pool.v[bt[r, t // BS], :, t % BS] = k[:, t], v[:, t]
            rows.append((k, v))
        pools.append(pool)
        past.append(rows)
    ctx = types.SimpleNamespace(pools=pools, step_inputs=lambda: step_inputs(bt, sl, BS))
    h = torch.randn(B, 1, inner.hidden_size, dtype=torch.bfloat16)
    cos, sin = (t.to(h.dtype) for t in inner.rope(sl[:, None]))
    return h, cos, sin, ctx, past, bt


def test_reads_and_writes_chain(model):
    """Every read is a step input, the layer's weights or config, the layer's input, or written by an earlier station;
    every write is read later or outlives the layer."""
    w = set(stations.layer_weights(model, 0))
    assert w == {r for s in stations.DECODE for r in s.reads if r.startswith("w.")}
    have = set(stations.STEP) | w | {"cfg", "h", "k_cache", "v_cache"}
    for i, s in enumerate(stations.DECODE):
        assert set(s.reads) <= have, (s.name, set(s.reads) - have)
        later = {r for t in stations.DECODE[i + 1:] for r in t.reads} | set(stations.LIVE)
        assert set(s.writes) <= later, (s.name, set(s.writes) - later)
        have |= set(s.writes)


def test_stations_compute_the_decode_layer(model, torch_kernels):
    """The stations' residual stream and K/V writes match the layer's own (CPU) forward over the same past."""
    h, cos, sin, ctx, past, bt = _decode_case(model)
    with torch.no_grad():
        out = stations.forward(model.model, h, cos, sin, ctx)
        for r, L in enumerate((5, 9)):
            hr = h[r:r + 1]
            for i, layer in enumerate(model.model.layers):
                k, v = past[i][r]
                hr, (k_new, v_new) = layer(hr, cos[r:r + 1], sin[r:r + 1], past_kv=(k[None], v[None]))
                blk = bt[r, L // BS]
                torch.testing.assert_close(ctx.pools[i].k[blk, :, L % BS], k_new[0, :, 0], atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(ctx.pools[i].v[blk, :, L % BS], v_new[0, :, 0], atol=0, rtol=0)
            torch.testing.assert_close(out[r:r + 1], hr, atol=3e-2, rtol=3e-2)


def _compose(first, last):
    """An override that runs its stretch's own stations in order."""
    names = [s.name for s in stations.DECODE]
    stretch = stations.DECODE[names.index(first):names.index(last) + 1]

    def fn(c):
        c, out = dict(c), {}
        for s in stretch:
            w = s.fn(c)
            c.update(w)
            out.update(w)
        return out
    return fn


@pytest.mark.parametrize("stretch", [("router", "down"), ("input_norm", "residual"), ("kv_append", "attention"),
                                     ("silu_mul", "silu_mul")])
def test_noop_override_changes_nothing(model, torch_kernels, monkeypatch, stretch):
    with torch.no_grad():
        h, cos, sin, ctx, _, _ = _decode_case(model)
        want = stations.forward(model.model, h, cos, sin, ctx)
        want_k = [p.k.clone() for p in ctx.pools]
        h, cos, sin, ctx, _, _ = _decode_case(model)
        monkeypatch.setattr(stations, "plan", lambda: stations.resolve(stations.DECODE, {stretch: _compose(*stretch)}))
        got = stations.forward(model.model, h, cos, sin, ctx)
    assert torch.equal(got, want)
    assert all(torch.equal(p.k, k) for p, k in zip(ctx.pools, want_k))


def test_resolve_replaces_the_stretch():
    fn = _compose("router", "down")
    steps = stations.resolve(stations.DECODE, {("router", "down"): fn})
    assert len(steps) == len(stations.DECODE) - 4
    assert steps[:7] == tuple(s.fn for s in stations.DECODE[:7]) and steps[8:] == (stations.DECODE[-1].fn,)


@pytest.mark.parametrize("overrides", [
    {("router", "nope"): lambda c: {}},
    {("down", "router"): lambda c: {}},
    {("router", "down"): lambda c: {}, ("down", "residual"): lambda c: {}},
])
def test_bad_overrides_are_refused(overrides):
    with pytest.raises(ValueError):
        stations.resolve(stations.DECODE, overrides)


def test_override_contract_is_checked():
    """An override sees only its stretch's inputs, must write what later stations read, may write only its stretch's
    writes, and writes the K/V pool in place."""
    def run(key, fn, c):
        i = [s.name for s in stations.DECODE].index(key[0])
        return stations.resolve(stations.DECODE, {key: fn})[i](c)
    c = {"n": torch.ones(2, 1, 4), "w.router": torch.ones(3, 4), "h": torch.ones(1), "k_cache": torch.ones(1)}
    assert set(run(("router", "router"), lambda c: {"logits": c["n"]}, c)) == {"logits"}
    with pytest.raises(KeyError):
        run(("router", "router"), lambda c: {"logits": c["h"]}, c)
    with pytest.raises(ValueError):
        run(("router", "router"), lambda c: {}, c)
    with pytest.raises(ValueError):
        run(("router", "router"), lambda c: {"logits": c["n"], "h": c["n"]}, c)
    kv = {k: torch.ones(1) for k in ("k", "v", "k_cache", "v_cache", "block_tables", "kv_rows", "kv_block", "kv_slot")}
    with pytest.raises(ValueError):
        run(("kv_append", "kv_append"), lambda c: {"k_cache": c["k_cache"].clone(), "v_cache": c["v_cache"]},
            {**kv, "cfg": None})


def test_no_overrides_shipped():
    assert stations.overrides() == {} and stations.plan() == tuple(s.fn for s in stations.DECODE)


def test_functional_stations_do_not_mutate(model, torch_kernels):
    h, cos, sin, ctx, _, bt = _decode_case(model)
    s = {st.name: st for st in stations.functional()}["kv_append"]
    k_cache = ctx.pools[0].k
    before = k_cache.clone()
    c = {**ctx.step_inputs(), "cfg": stations.layer_config(model), "k_cache": k_cache, "v_cache": ctx.pools[0].v,
         "k": torch.ones(2, 2, 1, 16, dtype=torch.bfloat16), "v": torch.ones(2, 1, 32, dtype=torch.bfloat16)}
    out = s.fn(c)
    assert torch.equal(k_cache, before) and not torch.equal(out["k_cache"], before)


def _layer_ctx(model, h, cos, sin, ctx, i=0):
    return {**ctx.step_inputs(), "cos": cos, "sin": sin, "cfg": stations.layer_config(model),
            **stations.layer_weights(model, i), "h": h, "k_cache": ctx.pools[i].k, "v_cache": ctx.pools[i].v}


def _run(stretch, c):
    c = dict(c)
    for s in stretch:
        c.update(s.fn(c))
    return c


def test_reference_has_the_stations_names_reads_and_writes():
    assert [(s.name, s.reads, s.writes) for s in stations.reference()] == \
        [(s.name, s.reads, s.writes) for s in stations.DECODE]


def test_reference_in_fp32_is_close_and_pure(model, torch_kernels):
    """A layer's reference on fp32 copies of its inputs is close to the stations in bf16, and leaves its inputs as
    they were (the K/V pool included)."""
    h, cos, sin, ctx, _, _ = _decode_case(model)
    c = _layer_ctx(model, h, cos, sin, ctx)
    with torch.no_grad():
        want = _run(stations.functional(), c)
        up = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in c.items()}
        before = {k: v.clone() for k, v in up.items() if isinstance(v, torch.Tensor)}
        got = _run(stations.reference(), up)
    assert all(torch.equal(up[k], v) for k, v in before.items())
    for n in stations.LIVE:
        assert got[n].dtype == torch.float32
        torch.testing.assert_close(got[n], want[n].float(), atol=3e-2, rtol=3e-2)


def test_reference_route_layout():
    """route_fused's layout: each block holds pairs of its expert or the padding M; every pair once; blocks past the
    end carry expert E."""
    torch.manual_seed(0)
    T, E, k, bm = 5, 8, 2, 4
    sorted_ids, expert_ids, tok, pair_w = stations._ref_route(torch.randn(T, E), E, k, True, bm)
    M = T * k
    assert sorted_ids.numel() == expert_ids.numel() * bm == (M + E * (bm - 1)) // bm * bm
    real = sorted_ids[sorted_ids < M].long()
    assert sorted(real.tolist()) == list(range(M))
    for b, e in enumerate(expert_ids.tolist()):
        rows = sorted_ids[b * bm:(b + 1) * bm]
        assert e == E and (rows == M).all() or e < E
    assert torch.equal(tok, (torch.arange(M) // k).int()) and torch.allclose(pair_w.view(T, k).sum(1), torch.ones(T))


def test_reference_gemm_takes_fp8_weights():
    """fp8 weights with their per-channel scales give the dequantized product."""
    torch.manual_seed(0)
    E, N, K, T, k = 4, 6, 8, 3, 2
    w = torch.randn(E, N, K)
    scale = w.abs().amax(-1) / 448
    w8 = (w / scale[..., None]).to(torch.float8_e4m3fn)
    a = torch.randn(T, K)
    sorted_ids, expert_ids, tok, _ = stations._ref_route(torch.randn(T, E), E, k, True, 4)
    got = stations._ref_moe_gemm(a, tok, w8, sorted_ids, expert_ids, T * k, w_scale=scale)
    deq = w8.float() * scale[..., None]
    plain = stations._ref_moe_gemm(a, tok, deq, sorted_ids, expert_ids, T * k)
    assert torch.equal(got, plain)
