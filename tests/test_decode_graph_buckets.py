"""Decode CUDA-graph row buckets: ladder shape + bucket selection invariants."""

import pytest

from inference_server.backends.custom_torch_backend import (
    CustomTorchBackend,
    _decode_buckets,
)


def _pick(buckets, n):
    """Call the real selection logic against a stub holding just the ladder."""
    stub = object.__new__(CustomTorchBackend)
    stub._graph_buckets = buckets
    return CustomTorchBackend._decode_bucket(stub, n)


def test_ladder_is_powers_of_two_up_to_max():
    assert _decode_buckets(256) == (2, 4, 8, 16, 32, 64, 128, 256)
    assert _decode_buckets(32) == (2, 4, 8, 16, 32)


def test_ladder_skips_bucket_1():
    """A batch-1 shape forces its own Inductor specialization (~126s startup) to buy 4% at
    n=1, so the ladder starts at 2 — except when max_batch_size makes 2 impossible."""
    assert 1 not in _decode_buckets(256)
    assert _decode_buckets(2) == (2,)
    assert _decode_buckets(1) == (1,)          # degenerate max_batch_size=1


def test_ladder_appends_non_power_of_two_max():
    assert _decode_buckets(24) == (2, 4, 8, 16, 24)


def test_env_override(monkeypatch):
    monkeypatch.setenv("CUSTOM_BACKEND_GRAPH_BUCKETS", "1,4,16,64")
    assert _decode_buckets(256) == (1, 4, 16, 64)
    # values above max_rows clamp to max_rows (never capture a graph we can't fill)
    monkeypatch.setenv("CUSTOM_BACKEND_GRAPH_BUCKETS", "1,8,512")
    assert _decode_buckets(32) == (1, 8, 32)   # explicit override may still ask for 1


@pytest.mark.parametrize("max_rows", [1, 8, 32, 256])
def test_selected_bucket_never_drops_rows(max_rows):
    """Correctness: the replayed graph must have >= n rows, or rows would be silently cut."""
    buckets = _decode_buckets(max_rows)
    for n in range(1, max_rows + 1):
        assert _pick(buckets, n) >= n


@pytest.mark.parametrize("max_rows", [8, 32, 256])
def test_padding_waste_bounded_at_2x(max_rows):
    """Perf: the whole point of the ladder — never pad more than 2x (was up to 64x).
    n=1 is the equality case now that the ladder starts at 2."""
    buckets = _decode_buckets(max_rows)
    for n in range(1, max_rows + 1):
        assert _pick(buckets, n) <= 2 * n
