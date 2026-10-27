"""The decode path routing table (models/megakernel/routing.py): lookup by row count and longest context, checks at
load, and a table equal to MEGAKERNEL_MAX_ROWS's cutoff routes every step as the rule does."""

import json
import types

import pytest

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.models.megakernel import routing


def _write(tmp_path, table):
    p = tmp_path / "routing.json"
    p.write_text(json.dumps(table))
    return p


def _route(table, cutoff, n, context):
    backend = types.SimpleNamespace(_mega_routing=table, _mega_max_rows=cutoff)
    state = types.SimpleNamespace(n_rows=n, max_len=lambda: context)
    return CustomTorchBackend._use_megakernel(backend, state)


def test_lookup(tmp_path):
    t = routing.load(_write(tmp_path, {"context_buckets": [1024, 8192],
                                       "rows": {"1": ["megakernel", "kernels"], "3": ["kernels", "megakernel"]}}), 16)
    assert t.path(1, 1) == "megakernel" and t.path(1, 1024) == "megakernel" and t.path(1, 1025) == "kernels"
    assert t.path(3, 8192) == "megakernel" and t.path(1, 8193) is None and t.path(2, 10) is None


def test_no_file_is_the_rule(tmp_path):
    assert routing.load(tmp_path / "missing.json", 16) is None
    assert [_route(None, 2, n, 100) for n in (1, 2, 3, 16)] == [True, True, False, False]


def test_table_equal_to_the_cutoff_changes_nothing(tmp_path):
    buckets = [512, 4096, 40960]
    t = routing.load(_write(tmp_path, {"context_buckets": buckets, "rows": {
        str(n): ["megakernel" if n <= 2 else "kernels"] * len(buckets) for n in range(1, 9)}}), 16)
    for n in range(1, 1025, 3):
        for context in (0, 1, 511, 512, 513, 4096, 40960, 40961, 100_000):
            assert _route(t, 2, n, context) == _route(None, 2, n, context), (n, context)


def test_outside_the_table_takes_the_rule(tmp_path):
    t = routing.load(_write(tmp_path, {"context_buckets": [1024], "rows": {"4": ["megakernel"]}}), 16)
    assert _route(t, 2, 4, 100) is True and _route(t, 2, 4, 5000) is False and _route(t, 2, 2, 100) is True


@pytest.mark.parametrize("table", [
    {"context_buckets": [4096, 1024], "rows": {"1": ["kernels", "kernels"]}},
    {"context_buckets": [1024, 1024], "rows": {"1": ["kernels", "kernels"]}},
    {"context_buckets": [1024], "rows": {"1": ["kernels", "kernels"]}},
    {"context_buckets": [1024], "rows": {"1": ["graphs"]}},
    {"context_buckets": [1024], "rows": {"17": ["megakernel"]}},
])
def test_bad_tables_are_refused(tmp_path, table):
    with pytest.raises(ValueError):
        routing.load(_write(tmp_path, table), 16)
