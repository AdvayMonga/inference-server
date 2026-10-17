"""FlashInfer prefill: the wave's row structure (CPU) and parity with the paged kernel (CUDA + flashinfer)."""

import importlib.util

import pytest
import torch

from inference_server.models import prefill_flashinfer as fi


def test_plan_inputs_and_kv_indices():
    prefix, suffix, bs = [0, 40, 33], [300, 7, 1], 16
    qo, kv, last, nb = fi.plan_inputs(prefix, suffix, bs)
    assert qo.tolist() == [0, 300, 307, 308] and nb == [19, 3, 3]
    assert kv.tolist() == [0, 19, 22, 25] and last.tolist() == [12, 15, 2]
    assert qo.dtype == kv.dtype == last.dtype == torch.int32 and qo.device.type == "cpu"
    bt = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0], [1, 2, 3, 4]])
    assert fi.kv_indices(bt, [3, 2, 4]).tolist() == [5, 6, 7, 8, 9, 1, 2, 3, 4]
    assert fi.kv_indices(bt, [3, 2, 4]).dtype == torch.int32


def test_unavailable_package_is_reported_not_raised(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "flashinfer":
            raise ImportError("no flashinfer")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert fi.available() is False


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("flashinfer") is None,
                    reason="needs CUDA and flashinfer")
@pytest.mark.parametrize("rows", [[(0, 300)], [(40, 7)], [(33, 1)], [(0, 300), (0, 120), (0, 7)],
                                  [(48, 200), (0, 333), (17, 1), (160, 64)]])
def test_flashinfer_prefill_matches_the_paged_kernel(rows):
    from tests.test_prefill_attention import D, _wave
    q, k, v, ctx, ref = _wave(rows)
    got = fi.flashinfer_prefill(q, ctx, 0, D ** -0.5)
    for r, (_, s) in enumerate(rows):
        torch.testing.assert_close(got[r, :s].float(), ref[r, :s].float(), atol=2e-2, rtol=2e-2)
