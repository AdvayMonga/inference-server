"""The simulator contract: deterministic, open-loop, driven by the SHARED policy code.

Loop-lane compatible (pytest + ruff only): the simulator imports `scheduling_policy`, which is
stdlib-only pure code — the one sanctioned engine import in research/.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import pytest

from inference_server.research import loop
from inference_server.research.compare import comparable
from inference_server.research import harness as H
from inference_server.research.corpus import (
    CORPUS_DIR,
    TraceRequest,
    WorkloadClass,
    load_trace,
)
from inference_server.research.schemas import Vitals
from inference_server.research.simulator import (
    PLACEHOLDER_A100_E4B,
    SimConfig,
    TimingModel,
    decode_limit,
    fit_timing_model,
    rank_correlation,
    simulate,
)

CLS = WorkloadClass(name="t", description="", slo_ttft_ms=200.0, slo_tpot_ms=50.0,
                    arrival_rate_rps=1.0, seen="x", heldout="y")
FAST = TimingModel(prefill=(0.01, 0.0, 0.0), decode=(0.01, 0.0, 0.0), fitted_from="test")
SLOW = TimingModel(prefill=(1.0, 0.0, 0.0), decode=(1.0, 0.0, 0.0), fitted_from="test")


def req(t: float, sid: str = "s", prompt: str = "x" * 64, max_tokens: int = 10,
        turn: int = 0) -> TraceRequest:
    return TraceRequest(arrival_s=t, session_id=sid, turn_index=turn, prompt=prompt,
                       max_tokens=max_tokens)


def errors(res) -> list[str | None]:
    return [r.error for r in res.rows]


# ---------------------------------------------------------------- determinism and the clock

def test_same_inputs_give_identical_rows():
    trace = [req(i * 0.1, sid=f"s{i % 3}", max_tokens=5 + i) for i in range(20)]
    a = simulate(trace, SimConfig(max_batch_size=4), PLACEHOLDER_A100_E4B)
    b = simulate(trace, SimConfig(max_batch_size=4), PLACEHOLDER_A100_E4B)
    assert a.rows == b.rows and a.summary(CLS) == b.summary(CLS)


def test_arrivals_follow_the_trace_clock_not_the_service_time():
    """The open-loop property: a slow engine gets the same arrivals as a fast one."""
    trace = [req(i * 0.05, max_tokens=20) for i in range(10)]
    fast = simulate(trace, SimConfig(max_batch_size=2), FAST)
    slow = simulate(trace, SimConfig(max_batch_size=2), SLOW)
    assert [r.fired_s for r in fast.rows] == [r.fired_s for r in slow.rows]
    assert [r.fired_s for r in fast.rows] == [round(i * 0.05, 4) for i in range(10)]
    assert slow.summary(CLS)["ttft_p95"] > fast.summary(CLS)["ttft_p95"]


def test_rate_scale_compresses_the_arrival_clock():
    trace = [req(1.0), req(2.0)]
    res = simulate(trace, SimConfig(rate_scale=2.0), FAST)
    assert [r.fired_s for r in res.rows] == [0.5, 1.0]
    assert [r.arrival_s for r in res.rows] == [1.0, 2.0]        # the trace value is kept


# ---------------------------------------------------------------- admission rules

def test_queue_cap_rejects_with_429():
    trace = [req(0.0) for _ in range(5)]
    res = simulate(trace, SimConfig(max_queue_size=2, max_batch_size=1), FAST)
    assert errors(res).count("429") == 3
    assert res.summary(CLS)["total_rejected"] == 3
    assert res.summary(CLS)["n_ok"] == 2


def test_admission_deadline_expires_stale_requests():
    trace = [req(0.0, max_tokens=100) for _ in range(4)]     # each takes ~1s at batch 1
    res = simulate(trace, SimConfig(max_batch_size=1, max_queue_wait_s=0.5), FAST)
    s = res.summary(CLS)
    assert errors(res)[0] is None
    assert errors(res).count("expired") == 3 and s["total_expired"] == 3
    assert s["total_rejected"] == 3, "expired requests count as rejected, as in the engine"

    res = simulate(trace, SimConfig(max_batch_size=1, max_queue_wait_s=0), FAST)
    assert res.summary(CLS)["n_ok"] == 4, "<= 0 disables the deadline"


def test_kv_exhaustion_blocks_admission_until_blocks_free():
    # 16 prompt tokens + 48 output = 4 blocks each; a 10-block pool fits two at a time.
    trace = [req(0.0, prompt="x" * 64, max_tokens=48) for _ in range(3)]
    res = simulate(trace, SimConfig(kv_blocks=10, block_size=16, max_batch_size=8), FAST)
    s = res.summary(CLS)
    assert s["kv_admit_blocked"] > 0
    assert s["n_ok"] == 3, "the blocked request is admitted once a row finishes"
    assert s["active_high_water"] == 2
    third = res.rows[2]
    assert third.queue_ms > res.rows[0].queue_ms
    assert s["pool_free_min"] == 2 and s["pool_free_end"] == 10
    panel = res.to_panel(CLS)
    assert panel.pool_free_blocks == 10 and panel.pool_utilization == 0.8


def test_request_larger_than_the_pool_is_rejected_not_stuck():
    trace = [req(0.0, prompt="x" * 64, max_tokens=48)]
    res = simulate(trace, SimConfig(kv_blocks=2, block_size=16), FAST)
    assert errors(res) == ["kv_too_large"]


def test_batch_slots_cap_the_active_set():
    trace = [req(0.0, max_tokens=10) for _ in range(6)]
    res = simulate(trace, SimConfig(max_batch_size=4), FAST)
    assert res.summary(CLS)["active_high_water"] == 4


# ---------------------------------------------------------------- prefill modes

def test_monolithic_prefill_is_serial_with_no_wave_term_and_empty_wave_sizes():
    """The engine default: serial prefill, wave_sizes never populated."""
    timing = TimingModel(prefill=(0.0, 0.001, 0.010), decode=(0.001, 0.0, 0.0))
    trace = [req(0.0, prompt="x" * 400, max_tokens=2) for _ in range(3)]  # 100 tokens each
    res = simulate(trace, SimConfig(prefill_mode="monolithic"), timing)
    assert res.summary(CLS)["wave_sizes"] == {}
    assert [r.prefill_ms for r in res.rows] == [100.0, 200.0, 300.0]


def test_batched_prefill_applies_the_wave_term_and_records_wave_sizes():
    timing = TimingModel(prefill=(0.0, 0.001, 0.010), decode=(0.001, 0.0, 0.0))
    trace = [req(0.0, prompt="x" * 400, max_tokens=2) for _ in range(3)]
    res = simulate(trace, SimConfig(prefill_mode="batched"), timing)
    assert res.summary(CLS)["wave_sizes"] == {3: 1}
    # each prefill costs 0.1s own + 0.01 * 300 wave tokens = 3.1s, cumulative
    assert [r.prefill_ms for r in res.rows] == [3100.0, 6200.0, 9300.0]


def test_unknown_prefill_mode_is_rejected():
    with pytest.raises(ValueError, match="prefill_mode"):
        SimConfig(prefill_mode="chunked")


# ---------------------------------------------------------------- prefix cache

def test_prefix_hit_shortens_prefill_for_a_repeated_prompt():
    prompt = "shared system prompt " * 20             # 420 chars -> 105 tokens
    timing = TimingModel(prefill=(0.0, 0.001, 0.0), decode=(0.001, 0.0, 0.0), fitted_from="t")
    trace = [req(0.0, prompt=prompt), req(1.0, prompt=prompt), req(2.0, prompt="y" * 420)]
    res = simulate(trace, SimConfig(block_size=4), timing)
    first, repeat, other = res.rows
    assert first.matched_tokens == 0 and other.matched_tokens == 0
    assert repeat.matched_tokens == 104, "longest block-aligned prefix, in tokens"
    assert repeat.prefill_ms < first.prefill_ms
    assert res.summary(CLS)["cache_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_prefix_hit_does_not_shrink_the_kv_reservation():
    """The engine reserves prompt + max_tokens before any lookup (scheduler._admit_pending).
    A sim that reserved less would admit more than the engine and could bless a policy
    the engine cannot run — wrong in the one direction a tier-1 tool must never be."""
    prompt = "shared system prompt " * 20             # 105 tokens + 10 out = 29 blocks of 4
    timing = TimingModel(prefill=(0.0, 0.001, 0.0), decode=(0.001, 0.0, 0.0), fitted_from="t")
    trace = [req(0.0, prompt=prompt), req(0.05, prompt=prompt)]   # 2nd arrives mid-decode
    # 50 blocks: two fit only if the 104 matched tokens were subtracted (29 + 3 blocks).
    res = simulate(trace, SimConfig(block_size=4, kv_blocks=50), timing)
    first, repeat = res.rows
    assert repeat.matched_tokens == 104 and repeat.prefill_ms < first.prefill_ms
    assert res.summary(CLS)["kv_admit_blocked"] >= 1 and res.summary(CLS)["active_high_water"] == 1
    assert res.summary(CLS)["pool_free_min"] == 50 - 29


# ---------------------------------------------------------------- the shared policy code

def flood_and_trickle() -> list[TraceRequest]:
    flood = [req(0.0, sid="flood", max_tokens=50) for _ in range(20)]
    return flood + [req(0.1, sid="trickle", max_tokens=5)]


def trickle_wait_ms(policy: str) -> float:
    res = simulate(flood_and_trickle(), SimConfig(max_batch_size=4, policy=policy), FAST)
    row = next(r for r in res.rows if r.session_id == "trickle")
    assert row.error is None
    return row.queue_ms


def test_fair_serves_the_trickle_session_before_fcfs_does():
    """One session floods, one trickles: VTC must let the trickle in as soon as a slot frees,
    FCFS makes it wait behind the whole flood. Proves scheduling_policy drives the decisions."""
    fcfs, fair = trickle_wait_ms("fcfs"), trickle_wait_ms("fair")
    assert fair < fcfs / 3, f"fair={fair}ms fcfs={fcfs}ms"


def test_fair_counters_come_from_the_shared_functions(monkeypatch):
    import inference_server.research.simulator as sim

    calls = {"order": 0, "charge": 0, "initial": 0}

    def spy(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(sim, "fair_order", spy("order", sim.fair_order))
    monkeypatch.setattr(sim, "fair_charge", spy("charge", sim.fair_charge))
    monkeypatch.setattr(sim, "fair_initial_counter", spy("initial", sim.fair_initial_counter))
    simulate(flood_and_trickle(), SimConfig(max_batch_size=4, policy="fair"), FAST)
    assert calls["initial"] == 21 and calls["order"] > 0 and calls["charge"] > 0


# ---------------------------------------------------------------- timing model

def test_fit_timing_model_recovers_known_coefficients():
    truth = TimingModel(prefill=(0.01, 0.001, 0.0005), decode=(0.005, 0.0002, 1e-6))
    rows = []
    for i in range(1, 30):
        p, bp, b, kv = 10 * i, 25 * i % 400, i % 8 + 1, 300 * i
        rows.append({"prompt_tokens": p, "batch_prompt_tokens": bp,
                     "prefill_s": truth.prefill_s(p, bp),
                     "batch_size": b, "total_kv_tokens": kv,
                     "decode_step_s": truth.decode_step_s(b, kv)})
    fit = fit_timing_model(rows, fitted_from="run-x", model="m", hardware="h")
    assert fit.prefill == pytest.approx(truth.prefill, abs=1e-9)
    assert fit.decode == pytest.approx(truth.decode, abs=1e-9)
    assert fit.fitted_from == "run-x"


def test_fit_without_the_optional_predictor_sets_its_coefficient_to_zero():
    rows = [{"prompt_tokens": p, "prefill_s": 0.02 + 0.003 * p,
             "batch_size": b, "decode_step_s": 0.01 + 0.0004 * b}
            for p, b in zip(range(10, 100, 10), range(1, 10))]
    fit = fit_timing_model(rows)
    assert fit.prefill == pytest.approx((0.02, 0.003, 0.0), abs=1e-9)
    assert fit.decode == pytest.approx((0.01, 0.0004, 0.0), abs=1e-9)


def test_timing_model_json_roundtrip(tmp_path):
    PLACEHOLDER_A100_E4B.to_json(tmp_path / "t.json")
    back = TimingModel.from_json(tmp_path / "t.json")
    assert back == PLACEHOLDER_A100_E4B and back.fitted_from == "placeholder"


# ---------------------------------------------------------------- validation primitive

def test_rank_correlation_on_known_permutations():
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert rank_correlation([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)
    assert rank_correlation([1, 1, 2], [1, 2, 3]) == pytest.approx(0.866, abs=1e-3)
    with pytest.raises(ValueError):
        rank_correlation([1, 2], [1])


# ---------------------------------------------------------------- the panel

def test_to_panel_validates_and_is_unmistakably_simulated():
    res = simulate([req(0.0), req(0.5, prompt="y" * 64)], SimConfig(policy="fair"),
                   PLACEHOLDER_A100_E4B)
    panel = res.to_panel(CLS, corpus_version="abc", split="seen")
    panel.validate()
    v = panel.validity
    assert v.harness == "simulator"
    assert v.harness_config["policy"] == "fair"
    assert v.harness_config["timing"]["fitted_from"] == "placeholder"
    assert v.workload_regime == "cache_miss_heavy" and v.workload_class == "t"
    assert panel.ttft_p95 == res.summary(CLS)["ttft_p95"]
    assert panel.ttft_queue_p95 is not None and panel.ttft_prefill_p95 is not None
    hardware = H.panel_from_stats(H.build_validity("replay_trace", {"x": 1}, n_samples=2,
                                                   workload_regime="mixed"))
    assert not comparable(panel, hardware), "a simulated panel never compares to hardware"


# ---------------------------------------------------------------- real corpus + CLI

def test_steady_interactive_runs_end_to_end_in_under_a_second():
    m, trace = load_trace("steady_interactive", "seen")
    t0 = time.perf_counter()
    res = simulate(trace, SimConfig(), PLACEHOLDER_A100_E4B)
    assert time.perf_counter() - t0 < 1.0
    s = res.summary(m.classes["steady_interactive"])
    assert s["n_ok"] + s["n_err"] == len(trace) and s["n_ok"] > 0
    assert s["decode_steps"] > 0 and s["cache_hit_rate"] is not None


def test_loop_simulate_prints_one_row_per_config_and_emits_panels(tmp_path, capsys):
    rc = loop.main(["simulate", "--class", "steady_interactive", "--split", "seen",
                    "--config", '{"policy": "fcfs"}', "--config", '{"policy": "fair"}',
                    "--emit", "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count('"policy": "fcfs"') == 1 and out.count('"policy": "fair"') == 1
    panels = [Vitals.load(p) for p in tmp_path.glob("*.json")]
    assert len(panels) == 2 and {p.validity.harness for p in panels} == {"simulator"}


def test_loop_simulate_rejects_an_unknown_knob(tmp_path):
    with pytest.raises(TypeError):
        loop.main(["simulate", "--class", "steady_interactive", "--config", '{"pool": 1}'])


def test_screen_points_cheap_tiers_at_the_simulator(tmp_path, capsys):
    hyp = {"statement": "fair share cuts p95 queue wait", "gap_id": "ttft-queue",
           "predicted_metric": "ttft_queue_p95", "predicted_direction": "decrease",
           "predicted_magnitude": ">30%", "falsification_tier": 1,
           "falsification_test": "replay steady_interactive under fcfs vs fair",
           "tags": ["scheduler"]}
    p = tmp_path / "h.json"
    p.write_text(json.dumps([hyp]))
    assert loop.main(["screen", str(p)]) == 0
    assert "loop simulate" in capsys.readouterr().out


# ---------------------------------------------------------------- termination

# sha256 over {rows, summary} for every committed (class, split) at two batch sizes, measured at
# 3aed53b, BEFORE TraceRequest gained `expected_output_tokens`. No committed trace populates the
# field, so `decode_limit` returns `max_tokens` and every one of these must still match: this is
# the guarantee that the seam moved no stored panel. It is a golden, not a property — if a corpus
# ever ships the field, the affected entries change and must be re-measured deliberately.
PRE_TERMINATION_GOLDEN = {
    "cold_start/heldout/mbs2": "cae8df6b839ed332bcfef522f7601b69b36cf7958611a770ae34a7b22b730db1",
    "cold_start/heldout/mbs8": "cae8df6b839ed332bcfef522f7601b69b36cf7958611a770ae34a7b22b730db1",
    "cold_start/seen/mbs2": "4b4804e32b702707745dddd4faf95db4d78f2c4f87a81a95deb03b0cc3e3e584",
    "cold_start/seen/mbs8": "4b4804e32b702707745dddd4faf95db4d78f2c4f87a81a95deb03b0cc3e3e584",
    "long_context/heldout/mbs2": "0914ca37dbfa53f9d087d902fd95affe7ac0137431e4872a0cdfcd4e94ac4044",
    "long_context/heldout/mbs8": "9de2ae019f2800a7585ef0a63197141cbfb5072bfde56bb30f20a5b686ae15d9",
    "long_context/seen/mbs2": "69e09473b67c352a6254bde157aab395d3eb5b0603b9c69ae0edad869201a76b",
    "long_context/seen/mbs8": "550948de4f6f027069da6b1d65b1c8d2ea3060d10c0ca253be1ab8b9efbeb049",
    "steady_interactive/heldout/mbs2": "917299b66a0d117e81a118adc5075fee6e20b3550f340a8c592e3a93de5c6518",
    "steady_interactive/heldout/mbs8": "690773e629be2b9c001ee1da72cbd77c02545aa9ab8839852b03dd3c52a6d117",
    "steady_interactive/seen/mbs2": "0966b315fd1f0f11746d0960353701ae50dbc812e0c3a60ac9483c16424357dd",
    "steady_interactive/seen/mbs8": "ae75a01fcbe9ffe7130f55d446628f9bf29bf0c0064eeb6173681005e85d16e3",
}


@pytest.mark.parametrize("key", sorted(PRE_TERMINATION_GOLDEN))
def test_committed_corpus_simulates_byte_identically_to_before_the_field(key):
    name, split, mbs = key.rsplit("/", 2)
    m, trace = load_trace(name, split, CORPUS_DIR)
    assert all(r.expected_output_tokens is None for r in trace), "no corpus populates it yet"
    res = simulate(trace, SimConfig(max_batch_size=int(mbs[3:])), PLACEHOLDER_A100_E4B)
    blob = json.dumps({"rows": [asdict(r) for r in res.rows],
                       "summary": res.summary(m.classes[name])}, sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == PRE_TERMINATION_GOLDEN[key]


def test_absent_expected_output_tokens_means_run_to_budget():
    assert decode_limit(req(0.0, max_tokens=7)) == 7
    res = simulate([req(0.0, max_tokens=7)], SimConfig(), FAST)
    assert res.rows[0].out_tokens == 7


def test_expected_output_tokens_stops_the_request_early():
    """The engine's stop token, modelled: decode ends at the shorter of budget and observed."""
    short = req(0.0, max_tokens=50)
    short.expected_output_tokens = 6
    res = simulate([short], SimConfig(), FAST)
    assert res.rows[0].out_tokens == 6 and res.rows[0].error is None
    assert res.decode_steps == 5, "1 token from prefill, 5 decode steps"


def test_the_budget_still_caps_an_over_long_expectation():
    over = req(0.0, max_tokens=4)
    over.expected_output_tokens = 999
    assert decode_limit(over) == 4
    assert simulate([over], SimConfig(), FAST).rows[0].out_tokens == 4
    zero = req(0.0, max_tokens=4)
    zero.expected_output_tokens = 0            # a request that emitted only its stop token
    assert decode_limit(zero) == 1
    assert simulate([zero], SimConfig(), FAST).rows[0].out_tokens == 1


def test_early_termination_frees_the_batch_slot_and_its_full_reservation():
    """The mechanism kb-20260919-94acfdb8 names: phantom rows holding a narrow batch. The
    reservation released is the budget, not the length emitted — as scheduler._evict_row does."""
    trace = [req(0.0, prompt="x" * 64, max_tokens=40), req(0.0, prompt="y" * 64, max_tokens=40)]
    trace[0].expected_output_tokens = 2
    res = simulate(trace, SimConfig(max_batch_size=1, block_size=16), FAST)
    first, second = res.rows
    assert first.out_tokens == 2 and second.out_tokens == 40
    # 16 prompt + 40 budget = 56 tokens = 4 blocks, for both rows, whichever way they ended
    assert res.pool_free_end == 4096 and res.pool_free_min == 4096 - 4
    slow = simulate([req(0.0, prompt="x" * 64, max_tokens=40), trace[1]],
                    SimConfig(max_batch_size=1, block_size=16), FAST)
    assert second.queue_ms < slow.rows[1].queue_ms, "the phantom row is what invented the wait"
