"""The decode step through the instruction interpreter (megakernel/framework.py, the default program) against today's
megakernel on the real model (CUDA, nvcc, the 60 GB weights: MEGAKERNEL_TEST=1), the framework's acceptance bars.
One layer: what the attention half writes (h, qkv, attn, o, h1, the new K/V) and the MoE half (n2, router logits,
experts and weights, act) bit for bit; out2 to fp32 rounding (both kernels add it up with atomics in any order). The
whole step: logits, tokens and every layer's new K/V bit for bit whenever today's kernel reproduces itself on that step
(those atomics may make it not). Greedy tokens over 300 steps of real prompts; a copy of the default program built and
run in its place changes nothing, and so does shipping it as the engine's STEP override; the step no slower at batch 1
and 2."""

import copy
import importlib.util
import os
import shutil
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("MEGAKERNEL_TEST") != "1",
                                reason="needs CUDA, nvcc and MEGAKERNEL_TEST=1 (loads the real model)")

BATCHES = [1, 2, 4, 8, 16]
CONTEXTS = {"short": 37, "long": 3000}         # one attention chunk, and many (COMBINE runs)
TEXTS = ["Explain how a paged KV cache lets an inference server batch requests of different lengths.",
         "Write a short story about a lighthouse keeper who finds a message in a bottle.",
         "What is the derivative of x^3 * sin(x)? Show each step.",
         "List five differences between TCP and UDP and when you would use each."]


@pytest.fixture(scope="module")
def backend():
    os.environ.update(CUSTOM_BACKEND_CUDA_GRAPH="0", CUSTOM_BACKEND_QUANT="bf16", MEGAKERNEL_FP8="0", CUSTOM_BACKEND_BLOCKS="8192")
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = CustomTorchBackend("cuda", name)
    b.load_model(name)
    return b


@pytest.fixture(scope="module")
def models(backend):
    """Today's megakernel models (bf16 and fp8 experts), whole and cut to layer 0, each with its framework."""
    from inference_server.models import megakernel as mk
    from inference_server.models.megakernel.framework import Framework
    fp8 = mk.Model(backend.model, backend.device, fp8=True)
    fp8.bind_pools(backend.pools)
    out = {}
    for which, m in (("bf16", backend._mega), ("fp8", fp8)):
        m1 = copy.copy(m)
        m1.table, m1.n_layers = m.table[:1].contiguous(), 1
        m1.weights, m1.k_pools, m1.v_pools = {**m.weights, "n_layers": 1}, m.k_pools[:1], m.v_pools[:1]
        out[which] = (m, Framework(m), m1, Framework(m1))
    return out


def _state(b, prompts):
    from inference_server.models.paged_kv_cache import BatchedDecodeState
    from inference_server.sampling import GREEDY
    rows = [b.prefill_chunk_batch([(None, p, True, GREEDY)])[0] for p in prompts]
    st = BatchedDecodeState(b.pools, b.device)
    for kv, _, _ in rows:
        st.add_row(kv)
    st.prepare_step()
    return st, [int(t) for _, t, _ in rows]


def _prompts(B, ctx, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1000, 30000, (ctx - 7 * i,), generator=g).tolist() for i in range(B)]


def _new_kv(backend, st, n_layers):
    """Every layer's K and V at each row's new position: [layers, 2, B, NKV, D]."""
    p = st.seq_lens
    blk = st.bt.gather(1, (p // 16).view(-1, 1))[:, 0]
    return torch.stack([torch.stack([pool.k[blk, :, p % 16], pool.v[blk, :, p % 16]])
                        for pool in backend.pools[:n_layers]])


def _step(backend, fn, mdl, st, tokens):
    logits, nxt = fn(mdl, tokens, st.seq_lens, st.bt)
    torch.cuda.synchronize()
    return logits, nxt, _new_kv(backend, st, mdl.n_layers)


def _scratch(obj, B):
    """The first B rows of each scratch buffer (common.cuh's carve_scratch), chunk partials left out."""
    from inference_server.models import megakernel as mk
    S, QKV = 32768 // mk.CHUNK, (mk.NQ + 2 * mk.NKV) * mk.D
    out = {}
    for t, layout in ((obj.f32, [("part_o", mk.BMAX * mk.NQ * S, mk.D), ("part_m", mk.BMAX * mk.NQ, S), ("part_l", mk.BMAX * mk.NQ, S),
                                 ("w", mk.BMAX, mk.K), ("out2", mk.BMAX, mk.H), ("rlogits", mk.BMAX, mk.E)]),
                      (obj.bf16, [("h", mk.BMAX, mk.H), ("qkv", mk.BMAX, QKV), ("attn", mk.BMAX, mk.NQ * mk.D), ("o", mk.BMAX, mk.H),
                                  ("h1", mk.BMAX, mk.H), ("n2", mk.BMAX, mk.H), ("act", mk.BMAX * mk.K, mk.I)]),
                      (obj.i32, [("ids", mk.BMAX, mk.K)])):
        n = 0
        for name, rows, width in layout:
            out[name] = t[n:n + rows * width].view(rows, width)
            n += rows * width
    return {k: (v[:B * 8] if k == "act" else v[:B]).clone() for k, v in out.items() if not k.startswith("part")}


def _free(st):
    st.remove_rows(list(range(st.n_rows)))


@pytest.mark.parametrize("ctx", list(CONTEXTS))
@pytest.mark.parametrize("B", BATCHES)
@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_one_layer_bit_identical(backend, models, which, B, ctx):
    """Stages 1 and 2, op by op: layer 0 alone through both kernels, every intermediate compared."""
    from inference_server.models import megakernel as mk
    from inference_server.models.megakernel import framework
    _, _, m1, fw1 = models[which]
    st, _ = _state(backend, _prompts(B, CONTEXTS[ctx], seed=B))
    tokens = torch.randint(1000, 30000, (B,), device="cuda")
    ref = _step(backend, mk.decode_step, m1, st, tokens)
    ref_s = _scratch(m1, B)
    got = _step(backend, framework.decode_step, fw1, st, tokens)
    got_s = _scratch(fw1, B)
    for k in ("h", "qkv", "attn", "o", "h1", "n2", "rlogits", "ids", "w", "act"):
        assert torch.equal(got_s[k], ref_s[k]), k
    torch.testing.assert_close(got_s["out2"], ref_s["out2"], rtol=1e-5, atol=1e-6)
    assert torch.equal(got[2], ref[2]), "layer 0's new K/V"
    _free(st)


@pytest.mark.parametrize("ctx", list(CONTEXTS))
@pytest.mark.parametrize("B", BATCHES)
@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_step_bit_identical(backend, models, which, B, ctx):
    """The whole step: bit for bit where today's kernel reproduces itself, else its own near-tie bounds."""
    from inference_server.models import megakernel as mk
    from inference_server.models.megakernel import framework
    from tests.test_megakernel import picks_agree
    m, fw, _, _ = models[which]
    st, _ = _state(backend, _prompts(B, CONTEXTS[ctx], seed=100 + B))
    tokens = torch.randint(1000, 30000, (B,), device="cuda")
    ref = _step(backend, mk.decode_step, m, st, tokens)
    ref2 = _step(backend, mk.decode_step, m, st, tokens)
    got = _step(backend, framework.decode_step, fw, st, tokens)
    assert torch.equal(got[2][0], ref[2][0]), "layer 0's new K/V"
    if all(torch.equal(x, y) for x, y in zip(ref, ref2)):
        for x, y, name in zip(got, ref, ("logits", "tokens", "K/V")):
            assert torch.equal(x, y), name
    else:                                         # today's kernel differs from itself here: the bound its own tests use
        ok, deficit = picks_agree(got[0], ref[0])
        assert ok, deficit
        assert (got[0].float() - ref[0].float()).abs().mean() < 0.1 * ref[0].float().abs().mean()
    _free(st)


@pytest.mark.parametrize("B", [1, 2, 4])
@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_greedy_tokens_identical_over_steps(backend, models, which, B):
    """Stage 2's bar: 300 greedy steps of real prompts, each kernel feeding back its own tokens."""
    from inference_server.models import megakernel as mk
    from inference_server.models.megakernel import framework
    m, fw, _, _ = models[which]
    prompts = [backend.tokenizer.encode(t) for t in TEXTS[:B]]
    runs = []
    for fn, mdl in ((mk.decode_step, m), (mk.decode_step, m), (framework.decode_step, fw)):
        st, first = _state(backend, prompts)
        cur, out = torch.tensor(first, device="cuda"), [first]
        for _ in range(300):
            _, cur = fn(mdl, cur, st.seq_lens, st.bt)
            st.advance()
            st.prepare_step()
            out.append(cur.tolist())
        runs.append(out)
        _free(st)
    first_diff = next((i for i, (a, b) in enumerate(zip(runs[2], runs[0])) if a != b), None)
    assert runs[2] == runs[0] or runs[1] != runs[0], f"first differing step {first_diff}; today's kernel reproduces itself"


def _load_program(path: Path):
    spec = importlib.util.spec_from_file_location(f"program_copy_{abs(hash(path))}", path / "__init__.py",
                                                  submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _time(fn, mdl, st, tokens, n=50):
    """Median ms of `n` steps (the same position written each time)."""
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(mdl, tokens, st.seq_lens, st.bt)
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return sorted(ts)[n // 2]


@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_a_copy_of_the_default_program_changes_nothing(backend, models, which, tmp_path):
    """Stage 3's bar: the default program copied elsewhere, built as its own extension and run in its place."""
    from inference_server.models.megakernel import framework
    from inference_server.models.megakernel import program
    from inference_server.models.megakernel.framework import Framework
    m, fw, m1, fw1 = models[which]
    shutil.copytree(Path(program.__file__).parent, tmp_path / "candidate", ignore=shutil.ignore_patterns("__pycache__"))
    cand = _load_program(tmp_path / "candidate")
    assert cand.OPS != program.OPS
    cfw, cfw1 = Framework(m, program=cand), Framework(m1, program=cand)
    for B in (1, 2, 8):
        st, _ = _state(backend, _prompts(B, 1000, seed=200 + B))
        tokens = torch.randint(1000, 30000, (B,), device="cuda")
        ref, got = _step(backend, framework.decode_step, fw1, st, tokens), _step(backend, framework.decode_step, cfw1, st, tokens)
        assert all(torch.equal(x, y) for x, y in zip(got[1:], ref[1:]))
        ref_s, got_s = _scratch(fw1, B), _scratch(cfw1, B)
        assert all(torch.equal(got_s[k], ref_s[k]) for k in ref_s if k != "out2")
        if B <= 2:
            t_def, t_copy = _time(framework.decode_step, fw, st, tokens), _time(framework.decode_step, cfw, st, tokens)
            print(f"{which} B={B}: default {t_def:.3f} ms, copy {t_copy:.3f} ms")
            assert t_copy < 1.03 * t_def
        _free(st)


@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_the_framework_shipped_as_a_step_override(backend, models, which, monkeypatch):
    """megakernel.decode_step with a STEP override built by framework.step(): the framework's own outputs."""
    from inference_server.models import megakernel as mk
    from inference_server.models.megakernel import framework
    _, _, m1, fw1 = models[which]
    monkeypatch.setattr(mk, "step_override", lambda s=framework.step(): s)
    for B in (1, 2, 8):
        st, _ = _state(backend, _prompts(B, 1000, seed=400 + B))
        tokens = torch.randint(1000, 30000, (B,), device="cuda")
        ref, got = _step(backend, framework.decode_step, fw1, st, tokens), _step(backend, mk.decode_step, m1, st, tokens)
        assert all(torch.equal(x, y) for x, y in zip(got[1:], ref[1:]))
        _free(st)


@pytest.mark.parametrize("ctx", [300, 1000])
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("which", ["bf16", "fp8"])
def test_step_no_slower_than_todays(backend, models, which, B, ctx):
    """Stages 1 and 2's time bar at batch 1 and 2."""
    from inference_server.models import megakernel as mk
    from inference_server.models.megakernel import framework
    m, fw, _, _ = models[which]
    st, _ = _state(backend, _prompts(B, ctx, seed=300 + B))
    tokens = torch.randint(1000, 30000, (B,), device="cuda")
    _time(mk.decode_step, m, st, tokens, 10), _time(framework.decode_step, fw, st, tokens, 10)     # warm
    t_mk, t_fw = _time(mk.decode_step, m, st, tokens), _time(framework.decode_step, fw, st, tokens)
    print(f"{which} B={B} ctx={ctx}: megakernel {t_mk:.3f} ms, framework {t_fw:.3f} ms")
    _free(st)
    assert t_fw <= 1.01 * t_mk
