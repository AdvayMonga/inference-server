"""The load generator must not silently measure only the PrefixCache hit path.

LOOP.md records a cache-hit benchmark that hid every miss-path bug as one of this
project's two most expensive mistakes. These tests pin the fix: prompts are unique by
default, sharing is opt-in, and the choice is recorded in the panel.
"""

import importlib.util
import random
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "load_test.py"


@pytest.fixture(scope="module")
def lt():
    spec = importlib.util.spec_from_file_location("load_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _heads(lt, workload: str, prefix_share: str, n: int = 12) -> set[str]:
    """First 32 chars of n drawn prompts — the PrefixCache matches from the front."""
    cfg = lt.WORKLOADS[workload]
    draws = [lt.make_draw(cfg, random.Random(1000 + i), prefix_share) for i in range(n)]
    return {d()[0][:32] for d in draws}


@pytest.mark.parametrize("workload", ["short", "long", "mixed"])
def test_tiled_workloads_are_unique_by_default(lt, workload):
    assert len(_heads(lt, workload, "none")) == 12


@pytest.mark.parametrize("workload", ["short", "long", "mixed"])
def test_prefix_share_full_pins_one_prompt(lt, workload):
    assert len(_heads(lt, workload, "full")) == 1


def test_build_prompt_holds_its_length_target(lt):
    """The unique preamble is charged against the target, not added on top."""
    plain = lt.build_prompt(200)
    unique = lt.build_prompt(200, random.Random(0))
    assert len(plain) == len(unique) == 800


def test_regime_is_recorded_in_the_panel(lt):
    """A latency number whose cache regime is unknown is not evidence."""
    row = lt.LevelResult(workload="short", concurrency=1, duration_s=1.0,
                         prefix_share="full").summary()
    assert row["prefix_share"] == "full"
