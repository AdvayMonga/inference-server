"""replay_local.py without a model: the plan, the stats shapes, and what the panel records.

The runs themselves are the instrument's job and need real hardware. What is tested here is
everything that decides whether those runs are *interpretable* afterwards — the ABBA order the
null experiment needs, the two shapes `/cache/stats` comes in, and the harness_config keys that
make two different engine configs refuse to compare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from inference_server.research.compare import comparable
from inference_server.research.corpus import Manifest, WorkloadClass
from inference_server.research.session import _alternating, arm_label

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "bench"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "tools"))
import fit_timing_from_runs as ft  # noqa: E402
import replay_local as rl  # noqa: E402
import replay_trace as rt  # noqa: E402

CLS = WorkloadClass("cold_start", "", 2000.0, None, 0.5, "cold_start/seen.jsonl",
                    "cold_start/heldout.jsonl")
MANIFEST = Manifest(1, "corpus-v1", {"cold_start": CLS}, {}, "")
HARDWARE = "Apple M4 Pro (MPS)"


def _result() -> rt.ReplayResult:
    res = rt.ReplayResult(wall_s=30.0)
    res.rows = [rt.Row(i, f"s-{i}", 0, f"p-{i}", float(i), float(i), 100.0 + i, 20.0, 8, None)
                for i in range(4)]
    return res


def _telemetry(prefix: str) -> list[dict]:
    rows = [{"trace_id": f"{prefix}-{i}", "terminal_state": "ok",
             "queue_wait_s": 0.001 * (i + 1), "prefill_s": 0.100 * (i + 1)} for i in range(4)]
    # A warm-up row and a shed row: neither belongs to this replay's TTFT split.
    rows.append({"trace_id": "warmup-0", "terminal_state": "ok",
                 "queue_wait_s": 9.0, "prefill_s": 9.0})
    rows.append({"trace_id": f"{prefix}-9", "terminal_state": "expired",
                 "queue_wait_s": 30.0, "prefill_s": None})
    return rows


# --------------------------------------------------------------------------- the plan

def test_null_plan_is_balanced_and_alternates():
    """`session.arms_for` refuses block-ordered arms, because with a monotonic warm-up drift the
    second block wins whatever the change was. The null plan has to satisfy the same rule it is
    calibrating."""
    plan = rl.null_plan("cold_start", "seen", 3, 1.0)
    assert len(plan) == 6
    order = [s.arm for s in plan]
    assert order == ["baseline", "treatment", "treatment", "baseline", "baseline", "treatment"]
    assert _alternating(order)
    assert order.count("baseline") == order.count("treatment") == 3


def test_null_plan_runs_one_config_in_both_arms():
    """A null is only a null if the arms are byte-identical; any difference makes it an A/B."""
    envs = {json.dumps(s.engine_env(), sort_keys=True) for s in rl.null_plan("cold_start", "seen",
                                                                            3, 1.0)}
    assert len(envs) == 1


def test_configs_plan_reads_labels_classes_and_env(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([
        {"label": "serial", "class": "cold_start", "rate_scale": 4.0,
         "env": {"MAX_BATCH_SIZE": 1}},
        {"label": "fair", "env": {"SCHEDULING_POLICY": "fair"}},
    ]))
    plan = rl.configs_plan(p)
    assert [s.label for s in plan] == ["serial", "fair"]
    assert plan[0].rate_scale == 4.0 and plan[0].engine_env()["MAX_BATCH_SIZE"] == "1"
    assert plan[1].engine_env()["SCHEDULING_POLICY"] == "fair"
    assert plan[1].engine_env()["MAX_BATCH_SIZE"] == rl.ENGINE_BASE["MAX_BATCH_SIZE"]


def test_the_base_config_pins_every_knob_the_simulator_needs():
    """sim_config refuses a knob the run did not record; the base config is where that is met."""
    assert ft.sim_config(rl.ENGINE_BASE, 1.0).max_batch_size == 8


def test_the_simulated_pool_is_the_pool_the_custom_backend_serves_from():
    """The custom backend owns CUSTOM_BACKEND_BLOCKS and ignores the CacheManager, but SIM_KNOBS
    reads KV_CACHE_NUM_BLOCKS. If they disagree the simulator models a different pool."""
    assert rl.ENGINE_BASE["KV_CACHE_NUM_BLOCKS"] == rl.ENGINE_BASE["CUSTOM_BACKEND_BLOCKS"]
    assert rl.ENGINE_BASE["KV_CACHE_BLOCK_SIZE"] == rl.ENGINE_BASE["CUSTOM_BACKEND_BLOCK_SIZE"]


def test_the_warmup_prompt_is_not_a_corpus_prompt():
    """Warming with a trace prompt would leave the replay hitting its own prefix cache, turning
    a cache_miss_heavy measurement into a cache_hit_heavy one without saying so."""
    from inference_server.research.corpus import load_trace
    _, trace = load_trace("cold_start", "seen")
    assert all(rl.WARMUP_PROMPT not in r.prompt for r in trace)


# --------------------------------------------------------------------------- stats shapes

def test_cache_stats_are_flattened_for_either_backend():
    flat = {"hit_rate": 0.25, "lookups": 4}
    assert rl.flatten_cache_stats(flat) == flat
    nested = {"backend": "CustomTorchBackend", "prefix_cache": {"hit_rate": 0.5, "lookups": 8}}
    assert rl.flatten_cache_stats(nested) == {"hit_rate": 0.5, "lookups": 8}


def test_the_ttft_split_comes_from_this_replays_ok_rows_only():
    s = rl.split_from_telemetry(_telemetry("cold_start-seen-abc"), "cold_start-seen-abc")
    # seconds -> ms, over the four ok rows only; H.pct is the same nearest-rank the rest of
    # the panel uses, so p95 of four samples is the third.
    assert s["ttft_queue_p50"] == 2.0 and s["ttft_queue_p95"] == 3.0
    assert s["ttft_prefill_p50"] == 200.0 and s["ttft_prefill_p95"] == 300.0
    assert rl.split_from_telemetry([], "x")["ttft_queue_p50"] is None


# --------------------------------------------------------------------------- the panel

def _panel(label: str, **env):
    spec = rl.RunSpec(label=label, arm=label, cls="cold_start", env=env)
    prefix = "cold_start-seen-abc"
    return rl.build_panel(spec, spec.engine_env(), MANIFEST, _result(),
                          {"active_high_water": 3, "wave_sizes": {1: 4}},
                          {"hit_rate": 0.0, "lookups": 4}, prefix, None,
                          _telemetry(prefix), HARDWARE)


def test_panel_carries_the_arm_label_and_the_ttft_split():
    p = _panel("null-baseline-0")
    assert arm_label(p) == "null-baseline-0"
    assert p.ttft_prefill_p50 == 200.0 and p.ttft_queue_p50 == 2.0
    assert p.validity.harness == "replay_trace" and p.validity.workload_class == "cold_start"
    p.validate()


def test_two_different_engine_configs_are_not_comparable():
    """The MAX_BATCH_SIZE 256-vs-32 error, in the shape this instrument could reproduce: one
    local run group deliberately holds several configs, so the panels must refuse each other."""
    assert comparable(_panel("a"), _panel("b"))
    for knob in ("MAX_BATCH_SIZE", "SCHEDULING_POLICY", "MAX_QUEUE_WAIT_S",
                 "CUSTOM_BACKEND_BLOCKS", "MAX_QUEUE_SIZE"):
        c = comparable(_panel("a"), _panel("b", **{knob: "99"}))
        assert not c.ok, f"{knob} changed the scheduler but the panels compared anyway"


def test_the_panel_carries_the_env_its_own_server_ran_under():
    """fit_timing_from_runs simulates each panel under its OWN config; a group-level env would
    rank a sweep as if every config had been the base one."""
    p = _panel("serial", MAX_BATCH_SIZE="1")
    env = ft.panel_env(p, rl.ENGINE_BASE)
    assert ft.sim_config(env, 1.0).max_batch_size == 1
    assert ft.sim_config(rl.ENGINE_BASE, 1.0).max_batch_size == 8
