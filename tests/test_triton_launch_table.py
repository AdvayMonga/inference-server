"""The Triton launch table: load/lookup/fallback, the kernel wiring with triton stubbed out (no
table == today's launches, exactly), and the tuning instrument's pure half."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference_server.models import launch_table

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "bench"))
sys.path.insert(0, str(REPO / "scripts" / "tools"))
import tune_triton_launch as tune  # noqa: E402

GPU, MODEL = "NVIDIA A100 80GB PCIe", "google/gemma-4-E4B-it"


@pytest.fixture(autouse=True)
def _empty_table():
    launch_table.clear()
    yield
    launch_table.clear()


def _doc(entries, model=MODEL):
    return {"kind": launch_table.KIND, "model": model, "entries": entries}


def _entry(kernel="paged_decode", shape=8, head_dim=256, block_size=16, gpu=GPU, **config):
    return {"kernel": kernel, "shape": shape, "head_dim": head_dim, "block_size": block_size,
            "gpu": gpu, "config": config or {"splits": 4, "num_warps": 2, "num_stages": 3}}


# ------------------------------------------------------------------ table semantics

def test_load_keeps_only_this_gpus_entries_and_lookup_is_exact(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_doc([_entry(), _entry(gpu="NVIDIA GeForce RTX 4090", shape=16)])))
    assert launch_table.load(p, GPU, MODEL) == 1
    assert launch_table.lookup("paged_decode", 8, 256, 16) == {"splits": 4, "num_warps": 2,
                                                               "num_stages": 3}
    assert launch_table.lookup("paged_decode", 16, 256, 16) is None      # other GPU's entry
    assert launch_table.lookup("paged_decode", 4, 256, 16) is None       # no nearest-match
    assert launch_table.lookup("paged_decode", 8, 512, 16) is None
    assert launch_table.lookup("paged_decode", 8, 256, 32) is None
    assert launch_table.lookup("paged_prefill", 8, 256, 16) is None


def test_another_models_table_loads_nothing_and_clears_the_old_one(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_doc([_entry()])))
    launch_table.load(p, GPU, MODEL)
    p.write_text(json.dumps(_doc([_entry()], model="google/gemma-4-E2B-it")))
    assert launch_table.load(p, GPU, MODEL) == 0
    assert launch_table.lookup("paged_decode", 8, 256, 16) is None


def test_a_file_that_is_not_a_launch_table_is_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"decode": [0.1, 0.0, 0.0]}))           # a TimingModel, say
    with pytest.raises(ValueError):
        launch_table.load(p, GPU, MODEL)


def test_compile_kwargs():
    assert launch_table.compile_kwargs(None) == {}
    assert launch_table.compile_kwargs({"splits": 2}) == {}
    assert launch_table.compile_kwargs({"splits": 2, "num_warps": 8, "num_stages": 1}) == {
        "num_warps": 8, "num_stages": 1}


# ------------------------------------------------------------------ the kernel wiring

@pytest.fixture
def kernels(monkeypatch):
    """paged_attention_kernel loaded against a stub triton whose kernels record their launches."""
    calls: list[tuple[str, tuple, dict]] = []

    class Kernel:
        def __init__(self, fn):
            self.name = fn.__name__

        def __getitem__(self, grid):
            return lambda *args, **kw: calls.append((self.name, tuple(grid), kw))

    triton = types.ModuleType("triton")
    triton.jit = Kernel
    triton.cdiv = lambda a, b: (a + b - 1) // b
    tl = types.ModuleType("triton.language")
    triton.language = tl
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    path = REPO / "src/inference_server/models/paged_attention_kernel.py"
    spec = importlib.util.spec_from_file_location("_stubbed_paged_attention_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.calls = calls
    return mod


def _decode(K, n, hq=8, hkv=1, d=256, bs=16):
    q = torch.zeros(n, hq, d, dtype=torch.bfloat16)
    pool = torch.zeros(4, hkv, bs, d, dtype=torch.bfloat16)
    K.paged_decode_attention(q, pool, pool, torch.zeros(n, 4, dtype=torch.int32),
                             torch.ones(n, dtype=torch.int32))


def _prefill(K, s, hq=8, hkv=1, d=256, bs=16):
    q = torch.zeros(1, s, hq, d, dtype=torch.bfloat16)
    pool = torch.zeros(4, hkv, bs, d, dtype=torch.bfloat16)
    one = torch.ones(1, dtype=torch.int32)
    K.paged_prefill_attention(q, pool, pool, torch.zeros(1, 4, dtype=torch.int32), one, one)


def test_no_table_decode_launches_exactly_as_before(kernels):
    """The launches origin/main makes: grid from decode_splits, and no num_warps/num_stages."""
    for n in (1, 2, 4, 8, 16, 32, 64, 256):
        kernels.calls.clear()
        _decode(kernels, n)
        s = kernels.decode_splits(n, 8)
        if s > 1:
            assert kernels.calls == [
                ("_paged_decode_splitk_kernel", (n, 8, s),
                 {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256, "SPLITS": s}),
                ("_splitk_combine_kernel", (n, 8), {"D": 256, "SPLITS": s})]
        else:
            assert kernels.calls == [
                ("_paged_decode_kernel", (n, 8), {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256})]


def test_no_table_prefill_launches_exactly_as_before(kernels, monkeypatch):
    for s in (64, 512, 1024):
        kernels.calls.clear()
        _prefill(kernels, s)
        assert kernels.calls == [
            ("_paged_prefill_kernel", (1, 8, s), {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256})]
    monkeypatch.setattr(kernels, "_TILED_PREFILL", True)
    kernels.calls.clear()
    _prefill(kernels, 512)
    assert kernels.calls == [("_paged_prefill_tiled_kernel", (1, 8, 32),
                              {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256, "BLOCK_M": 16,
                               "num_warps": 4})]


def test_a_table_entry_changes_only_its_own_key(kernels, monkeypatch):
    key = launch_table.entry_key
    launch_table._TABLE[key("paged_decode", 16, 256, 16)] = {"splits": 4, "num_warps": 2,
                                                             "num_stages": 1}
    launch_table._TABLE[key("paged_prefill", 64, 256, 16)] = {"num_warps": 8, "num_stages": 2}
    launch_table._TABLE[key("paged_prefill_tiled", 512, 256, 16)] = {
        "block_m": 32, "num_warps": 8, "num_stages": 2}

    _decode(kernels, 16)
    _decode(kernels, 32)                                      # untabled bucket: default launch
    _decode(kernels, 16, d=512)                               # untabled head_dim: default launch
    _prefill(kernels, 64)
    _prefill(kernels, 128)
    _prefill(kernels, 512)                                    # tiling off: table cannot enable it
    monkeypatch.setattr(kernels, "_TILED_PREFILL", True)
    _prefill(kernels, 512)
    assert kernels.calls == [
        ("_paged_decode_splitk_kernel", (16, 8, 4),
         {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256, "SPLITS": 4, "num_warps": 2, "num_stages": 1}),
        ("_splitk_combine_kernel", (16, 8), {"D": 256, "SPLITS": 4}),
        ("_paged_decode_kernel", (32, 8), {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256}),
        ("_paged_decode_kernel", (16, 8), {"GROUP": 8, "BLOCK_SIZE": 16, "D": 512}),
        ("_paged_prefill_kernel", (1, 8, 64),
         {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256, "num_warps": 8, "num_stages": 2}),
        ("_paged_prefill_kernel", (1, 8, 128), {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256}),
        ("_paged_prefill_kernel", (1, 8, 512), {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256}),
        ("_paged_prefill_tiled_kernel", (1, 8, 16),
         {"GROUP": 8, "BLOCK_SIZE": 16, "D": 256, "BLOCK_M": 32, "num_warps": 8,
          "num_stages": 2}),
    ]


def test_backend_loads_the_table_before_any_graph_is_captured():
    """Read at load, not first call: a captured graph freezes whatever launch was active."""
    src = (REPO / "src/inference_server/backends/custom_torch_backend.py").read_text()
    body = src[src.index("    def load_model("):src.index("    def set_cache_adapter(")]
    assert "launch_table.load(" in body and "CUSTOM_BACKEND_LAUNCH_TABLE" in body


# ------------------------------------------------------------------ the instrument's pure half

def test_config_space_is_the_full_cross_product():
    dec = tune.decode_space()
    assert len(dec) == 4 * 4 * 4 and len({tuple(sorted(c.items())) for c in dec}) == 64
    assert {c["splits"] for c in dec} == {1, 2, 4, 8}
    assert {c["num_warps"] for c in dec} == {1, 2, 4, 8}
    assert {c["num_stages"] for c in dec} == {1, 2, 3, 4}
    assert len(tune.prefill_space()) == 16
    assert all("splits" not in c for c in tune.prefill_space())
    assert len(tune.tiled_space()) == 3 * 16
    assert {c["block_m"] for c in tune.tiled_space()} == {16, 32, 64}


def test_attention_shapes_from_a_gemma_config():
    cfg = SimpleNamespace(layer_types=["sliding_attention"] * 4 + ["full_attention"],
                          head_dim=256, global_head_dim=512, num_attention_heads=8,
                          num_key_value_heads=2, sliding_window=512)
    assert tune.attention_shapes(cfg) == [tune.Shape(256, 8, 2, 512),
                                          tune.Shape(512, 8, 2, tune.FULL)]


def test_plan_sweeps_every_bucket_and_tiles_only_where_the_engine_could():
    shapes = [tune.Shape(256, 8, 2, 512), tune.Shape(512, 8, 2, tune.FULL)]
    cases = tune.plan(shapes, (2, 4, 8), (64, 512, 1024), 16, 512, {256})
    by = {}
    for c in cases:
        by.setdefault(c.kernel, []).append((c.shape, c.attn.head_dim))
    assert sorted(by["paged_decode"]) == [(n, d) for n in (2, 4, 8) for d in (256, 512)]
    assert sorted(by["paged_prefill"]) == [(s, d) for s in (64, 512, 1024) for d in (256, 512)]
    assert sorted(by["paged_prefill_tiled"]) == [(512, 256), (1024, 256)]


def _row(cfg, ms, diff=0.0, error=None):
    return {"config": cfg, "ms": ms, "max_abs_diff": diff, "exact": diff == 0.0, "error": error}


def test_pick_takes_the_fastest_valid_config_and_only_past_the_gain_bar():
    case = tune.Case("paged_decode", 8, tune.Shape(256, 8, 2, 512), 16, [])
    rows = [_row({"splits": 8}, 0.5, error="OutOfResources"),
            _row({"splits": 4}, 0.6, diff=1.0),                    # fast but wrong
            _row({"splits": 2}, 0.8, diff=1e-3),
            _row({"splits": 1}, 0.9)]
    e = tune.pick(case, 1.0, rows)
    assert e["config"] == {"splits": 2} and e["ms"] == 0.8 and e["default_ms"] == 1.0
    assert e["exact"] is False
    assert tune.pick(case, 0.82, rows) is None                    # 2.5% is noise, not a win
    assert tune.pick(case, 1.0, [_row({"splits": 4}, 0.1, diff=1.0)]) is None


def test_prefill_uses_the_tighter_parity_tolerance():
    case = tune.Case("paged_prefill", 64, tune.Shape(256, 8, 2, 512), 16, [])
    assert tune.pick(case, 1.0, [_row({"num_warps": 8}, 0.5, diff=2e-2)]) is None
    assert tune.Case("paged_decode", 8, case.attn, 16, []).tol == 5e-2


def test_a_written_table_is_what_the_engine_loads(tmp_path):
    case = tune.Case("paged_decode", 8, tune.Shape(256, 8, 2, 512), 16, [])
    e = tune.pick(case, 1.0, [_row({"splits": 4, "num_warps": 2, "num_stages": 3}, 0.5)])
    doc = tune.build_table([e], model=MODEL, gpu=GPU, sha="abc1234", run_group="grp-x")
    path = tune.write_table(doc, tmp_path / tune.table_filename(GPU, "abc1234"))
    assert path.name == "triton-launch-nvidia-a100-80gb-pcie-abc1234.json"
    assert launch_table.load(path, GPU, MODEL) == 1
    assert launch_table.lookup("paged_decode", 8, 256, 16) == {"splits": 4, "num_warps": 2,
                                                               "num_stages": 3}


def test_the_launcher_writes_the_pods_table_home(tmp_path):
    import run_on_runpod as rr

    doc = tune.build_table([], model=MODEL, gpu=GPU, sha="abc1234", run_group="grp-x")
    written = rr.write_launch_table({"launch_table": doc, "sweep": [{"ms": 1.0}]}, "grp-x",
                                    tmp_path / "runs", timing_dir=tmp_path / "timing")
    assert written == [tmp_path / "timing" / "triton-launch-nvidia-a100-80gb-pcie-abc1234.json",
                       tmp_path / "runs" / "grp-x" / "triton_launch_sweep.json"]
    assert json.loads(written[0].read_text()) == doc
    assert json.loads(written[1].read_text()) == [{"ms": 1.0}]
