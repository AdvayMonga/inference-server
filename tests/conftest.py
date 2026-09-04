"""Shared test config — and the guard rails that keep the suite from eating the machine.

The full suite loads a 9.6 GB model repeatedly: test_gemma4_parity does 5 separate from_hf()
calls, test_sliding_window 2 more, and test_custom_backend_scheduler / test_prefill_batch each
hold their own module-scoped backend. On a 24 GB laptop that only survives because CPython frees
each one before the next allocates — there is no headroom, and two suite runs at once will take
the machine down.

So: model-heavy tests are marked `heavy` and DESELECTED by default. Run them deliberately:

    pytest                  # fast tests only, safe, seconds
    pytest -m heavy         # the model-loading ones, one process at a time
    pytest -m ''            # everything (what used to be the default)
"""

import pytest

# Any test module that constructs a real Gemma model or backend.
_HEAVY_MODULES = {
    "test_gemma4_parity",
    "test_sliding_window",
    "test_custom_backend_scheduler",
    "test_prefill_batch",
    "test_chunked_prefill",
    "test_scheduler_splice",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: loads a real model (GBs of RAM); run alone")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.module.__name__.rsplit(".", 1)[-1] in _HEAVY_MODULES:
            item.add_marker(pytest.mark.heavy)

    if config.getoption("-m"):
        return  # caller chose explicitly; respect it
    skip = pytest.mark.skip(reason="model-heavy; run with -m heavy (needs several GB, alone)")
    for item in items:
        if item.get_closest_marker("heavy"):
            item.add_marker(skip)
