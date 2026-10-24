"""The decode stations (models/stations.py) on CPU, the Triton kernels replaced by torch versions of the same math:
the station list's reads and writes chain, the stations compute the layer's decode forward, and override resolution
(a no-op override changes nothing; bad ones are refused). Bit-identity with the real kernels is a GPU check."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from inference_server.models import fused, stations
from inference_server.models.common import apply_rotary
from inference_server.models.paged_kv_cache import step_inputs
from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM

BS = 4


def _rmsnorm(x, w, eps):
    x32 = x.float()
    return w * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _route(logits, E, k, renorm, block_m):
    w, e = torch.topk(F.softmax(logits, dim=-1, dtype=torch.float), k, dim=-1)
    if renorm:
        w = w / w.sum(dim=-1, keepdim=True)
    sorted_e, order = torch.sort(e.view(-1), stable=True)
    M = e.numel()
    return order.int(), sorted_e.int(), (torch.arange(M) // k).int(), w.view(-1)


def _moe_gemm(a, tok, w, sorted_ids, expert_ids, M, block_m, w_scale=None, a_scale=None):
    rows = a[tok[sorted_ids].long()].float()
    c = torch.empty(M, w.shape[1], dtype=torch.bfloat16)
    c[sorted_ids.long()] = torch.einsum("mk,mnk->mn", rows, w[expert_ids.long()].float()).to(torch.bfloat16)
    return c


def _moe_gemm_scatter(a, w, sorted_ids, expert_ids, M, tok, pair_w, out, block_m, w_scale=None, a_scale=None):
    s = sorted_ids.long()
    y = torch.einsum("mk,mnk->mn", a[s].float(), w[expert_ids.long()].float())
    out.index_add_(0, tok[s].long(), y * pair_w[s, None])


def _paged_decode_attention(q, k_pool, v_pool, bt, sl, scale):
    out = []
    for r in range(q.shape[0]):
        L = int(sl[r])
        blocks = bt[r, :(L + BS - 1) // BS].long()
        k = k_pool[blocks].permute(1, 0, 2, 3).flatten(1, 2)[:, :L]          # [Hkv, L, D]
        v = v_pool[blocks].permute(1, 0, 2, 3).flatten(1, 2)[:, :L]
        rep = q.shape[1] // k.shape[0]
        out.append(F.scaled_dot_product_attention(q[r][:, None], k.repeat_interleave(rep, 0),
                                                  v.repeat_interleave(rep, 0), scale=scale)[:, 0])
    return torch.stack(out)


@pytest.fixture
def torch_kernels(monkeypatch):
    """The fused and Triton kernels the stations call, as torch."""
    monkeypatch.setattr(fused, "rmsnorm", _rmsnorm)
    monkeypatch.setattr(fused, "add_rmsnorm", lambda h, a, w, eps: (h + a, _rmsnorm(h + a, w, eps)))
    monkeypatch.setattr(fused, "rmsnorm_rope",
                        lambda x, w, cos, sin, eps: apply_rotary(_rmsnorm(x, w, eps), cos, sin, unsqueeze_dim=2))
    monkeypatch.setattr(fused, "silu_mul", lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
    monkeypatch.setattr(fused, "add_cast", lambda h, y: (h.float() + y.view(h.shape)).to(h.dtype))
    mk = types.SimpleNamespace(route_fused=_route, route_multi=_route, moe_gemm=_moe_gemm,
                               moe_gemm_scatter=_moe_gemm_scatter)
    monkeypatch.setitem(sys.modules, "inference_server.models.moe_kernel", mk)
    monkeypatch.setattr(sys.modules["inference_server.models"], "moe_kernel", mk, raising=False)
    pak = types.SimpleNamespace(paged_decode_attention=_paged_decode_attention)
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
