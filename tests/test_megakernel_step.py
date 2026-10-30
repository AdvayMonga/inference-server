"""The whole-step override seam of the megakernel (megakernel.step_override, Model.weights / from_weights,
framework.step) and the self-contained kernel source (megakernel.source). CPU only: the extension is faked."""

import sys
import types

import pytest
import torch

from inference_server.models import megakernel as mk


def _package(tmp_path, name, modules):
    d = tmp_path / name
    d.mkdir()
    (d / "__init__.py").write_text("")
    for mod, text in modules.items():
        (d / f"{mod}.py").write_text(text)
    sys.path.insert(0, str(tmp_path))
    return name


def test_step_override_found_once(tmp_path):
    assert mk.step_override() is None                          # the engine ships none
    pkg = _package(tmp_path, "regions_one", {"a": "OVERRIDES = {}\n", "b": "def STEP(*a):\n    return 'b'\n"})
    assert mk.step_override(pkg)() == "b"
    pkg = _package(tmp_path, "regions_two", {"a": "STEP = 1\n", "b": "STEP = 2\n"})
    with pytest.raises(ValueError):
        mk.step_override(pkg)


class _Ext:
    def scratch_f32_elems(self):
        return 4

    scratch_bf16_elems = scratch_i32_elems = scratch_f32_elems

    def decode_step(self, *args):
        return "extension"


def _weights(n=2, fp8=False):
    from inference_server.models.stations import LayerConfig
    w = {f"{i}.{k}": torch.zeros(2) for i in range(n) for k in mk.ROW}
    if not fp8:
        for i in range(n):
            w[f"{i}.w.gate_up_scale"] = w[f"{i}.w.down_scale"] = None
    w.update(embed=torch.zeros(4, 2), final_norm=torch.zeros(2), lm_head=torch.zeros(4, 2), inv_freq=torch.zeros(2),
             cfg=LayerConfig(32, 4, 128, 0.5, 1e-6, 128, 8, True), n_layers=n)
    return w


def _pools(n=2):
    z = [torch.zeros(3, mk.NKV, 16, mk.D, dtype=torch.bfloat16) for _ in range(2 * n)]
    return z[:n], z[n:]


def test_from_weights_builds_the_kernels_table(monkeypatch):
    monkeypatch.setattr(mk, "_ext", lambda *a: _Ext())
    w, (k, v) = _weights(fp8=True), _pools()
    m = mk.Model.from_weights(w, k, v)
    assert m.fp8 and m.n_layers == 2 and m.scale == 0.5 and m.weights is w and m.k_pools == k
    assert m.table[1].tolist() == [w[f"1.{n}"].data_ptr() for n in mk.ROW] + [k[1].data_ptr(), v[1].data_ptr()]
    assert not mk.Model.from_weights(_weights(), k, v).fp8


def test_decode_step_runs_the_override_for_a_model(monkeypatch):
    monkeypatch.setattr(mk, "_ext", lambda *a: _Ext())
    seen = []
    monkeypatch.setattr(mk, "step_override", lambda: lambda *a: seen.append(a) or "override")
    w, (k, v) = _weights(), _pools()
    m = mk.Model.from_weights(w, k, v)
    t = torch.zeros(1, dtype=torch.long)
    assert mk.decode_step(m, t, t, t.view(1, 1)) == "override"
    assert seen[0][3] is m.k_pools and seen[0][4] is m.v_pools and seen[0][5] is w
    assert mk.decode_step(m, t, t, t.view(1, 1), times=torch.zeros(16, dtype=torch.long)) == "extension"
    other = types.SimpleNamespace(**{a: getattr(m, a) for a in vars(m)})      # e.g. the kernel layer's baseline
    assert mk.decode_step(other, t, t, t.view(1, 1)) == "extension"


def test_model_weights_use_the_stations_names():
    from inference_server.models import stations
    from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM
    from tests.conftest import tiny_qwen3_moe_cfg
    model = Qwen3MoeForCausalLM.from_config(tiny_qwen3_moe_cfg())
    for layer in model.model.layers:
        layer.self_attn.fuse_qkv()
    assert tuple(stations.layer_weights(model, 0)) == mk.ROW


def test_source_is_self_contained():
    src = mk.source()
    assert '#include "common.cuh"' not in src and src.count("inline Scratch carve_scratch") == 1
    assert src.count("unsigned long long gtimer()") == 1 and "#pragma once" not in src
