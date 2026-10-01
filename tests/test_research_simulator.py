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

@pytest.mark.needs_host
def test_steady_interactive_runs_end_to_end_in_under_a_second():
    m, trace = load_trace("steady_interactive", "seen")
    t0 = time.perf_counter()
    res = simulate(trace, SimConfig(), PLACEHOLDER_A100_E4B)
    assert time.perf_counter() - t0 < 1.0
    s = res.summary(m.classes["steady_interactive"])
    assert s["n_ok"] + s["n_err"] == len(trace) and s["n_ok"] > 0
    assert s["decode_steps"] > 0 and s["cache_hit_rate"] is not None


@pytest.mark.needs_host
def test_loop_simulate_prints_one_row_per_config_and_emits_panels(tmp_path, capsys):
    rc = loop.main(["simulate", "--class", "steady_interactive", "--split", "seen",
                    "--config", '{"policy": "fcfs"}', "--config", '{"policy": "fair"}',
                    "--emit", "--runs-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count('"policy": "fcfs"') == 1 and out.count('"policy": "fair"') == 1
    panels = [Vitals.load(p) for p in tmp_path.glob("*.json")]
    assert len(panels) == 2 and {p.validity.harness for p in panels} == {"simulator"}


@pytest.mark.needs_host
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

# sha256 over the simulated ROWS for every committed (class, split) at two batch sizes, measured
# at 3aed53b, BEFORE TraceRequest gained `expected_output_tokens`. No committed trace populates
# the field, so `decode_limit` returns `max_tokens` and every one of these must still match: this
# is the guarantee that the seam moved no stored panel. It is a golden, not a property — if a
# corpus ever ships the field, the affected entries change and must be re-measured deliberately.
#
# The hash covered {rows, summary} until PANEL_VERSION 2, which moved `ttft_p50` / `ttft_p95` onto
# a per-attempt ceiling-based percentile and added `n_failed` — a change to what the PANEL means,
# not to what the simulator does. Hashing the rows alone keeps this golden pointed at the seam it
# was written to guard and makes it immune to the next panel-definition change; the rows half was
# verified byte-identical across that bump for all twelve configs, which is what let it be split
# rather than simply re-pinned. `SUMMARY_GOLDEN` below carries the panel half at version 2.
# Both re-pinned 2026-10-01 on the real-trace corpus (a new corpus_version; simulator
# unchanged), and `spike` added.
PRE_TERMINATION_GOLDEN = {
    "cold_start/heldout/mbs2": "78f11328fbbf3ae8f4bf83cd1b0f2b4f16e74513c42c2eb95c1f46790089b4f3",
    "cold_start/heldout/mbs8": "37c98f74cb73062e9a95ca85572ccc1158f52510e37a59f7cef3b6375c7c8bb5",
    "cold_start/seen/mbs2": "b0188b82abec34ded05fd9be857c2b030cb83d41877b3c83b2a2e119add330f0",
    "cold_start/seen/mbs8": "0f229ea4ad472ce045b7393f50f63dcc8c170370610302a16558468b0291782d",
    "long_context/heldout/mbs2": "987f7620775c9d249f6363f93630402d2d0dc09d97d617ab01289b65b70aab9b",
    "long_context/heldout/mbs8": "bf41c91bc616d769d16b9940cb4c9087ae0e1ac64865f6d0f05b27f8ddaf6f08",
    "long_context/seen/mbs2": "b3f067da63155f542164c2bc55ad8df3a14baf1aff503529bdb7c6e825b8b39a",
    "long_context/seen/mbs8": "8a6c3d56949746149f28e4995d3b5eabd8168ba85f49325f774fc1165d029468",
    "steady_interactive/heldout/mbs2": "61084a8e6e28db5884c53756b0da6cb14f2e6f44404cf6483f2576a6c828679e",
    "steady_interactive/heldout/mbs8": "a1cc3ba782608196bb87fa7f798070b7fb3a079457331a1caf838c5bc20b060b",
    "steady_interactive/seen/mbs2": "898809590501b6cd4082bf4d0b9f20f85ee5c2e82836b871968e3a60d8990b41",
    "steady_interactive/seen/mbs8": "9a6a0a7c4d4c63dd4d24d7a5dddc236e80cd2ce597e679431bfa4cf1f8a1ab18",
    "spike/heldout/mbs2": "a06698b0dfd0365348f7aa2d1b29a6ea4fff871865aef1003f8d3e9bc52cbd8f",
    "spike/heldout/mbs8": "316bb2150f71bc36b36795be3e7813f6c6b8bf69ca89851da3c5f3104d7741b1",
    "spike/seen/mbs2": "2c7287cbb8062c734dd8e20b1aecc853889f34abdd903141310a63aa5ec04228",
    "spike/seen/mbs8": "b2c0c6ad5a67aa48641eb35c2362d9c98dd2296557e15107e7bdc38a81ed4f01",
}


# The panel half, re-pinned at PANEL_VERSION 2. Separate constant because the two guard different
# things: the rows above must never move until a corpus populates the field, while these move
# whenever the panel's definition does — deliberately, and only in a PR that bumps the version.
SUMMARY_GOLDEN = {
    "cold_start/heldout/mbs2": "e69461ddd9e9edc9e31b458ad79ecfbd477f24e6b4419827a4c624cc98d4abbe",
    "cold_start/heldout/mbs8": "174c6125e9ecfc7c75886fa4614bf1221d1268492dd5908b84872553e11fb8bb",
    "cold_start/seen/mbs2": "0a9ebb6f710109ed3a7545e32b742530b6e1f662d04835ec5695bc652577095d",
    "cold_start/seen/mbs8": "a33553653e2451d2307e0f83d8c0c43edd28405b924ec7da0588d7e894ac2f50",
    "long_context/heldout/mbs2": "773d10cd6faa906f45c96dd12f71b8e58b1d0ca1edd383f81698b3a8c251b82a",
    "long_context/heldout/mbs8": "18e1a41111b11ddd48ec15470d20f5d75c21411a8a54d98ae02aec23599868e3",
    "long_context/seen/mbs2": "b8d11af17e7952a46b83ee7c1f77b1bef79ff5226e9112cfa25701055347daff",
    "long_context/seen/mbs8": "21c1d731206bc37283781d7b1038466b890d8f2f6b7b3797db2e04763d254ec0",
    "steady_interactive/heldout/mbs2": "f553f26b376d5e227f94039da0a6018fece86b3a9c410f2817c1edd3ea40fe1c",
    "steady_interactive/heldout/mbs8": "566329215f8f7978ae9420e8fa62ea44a272d20538feacb1855ccd3a96cee42c",
    "steady_interactive/seen/mbs2": "9db5e5fd024c29388d8f2262df1aa9b1edd250a6d229d9076c6778031bce5058",
    "steady_interactive/seen/mbs8": "8536eedaf88dc83bf74546f312e99f5ce46dc558cf87ad37456b957d01a9c7b7",
    "spike/heldout/mbs2": "9801564527a8fac3c3701a8e83c968824fc043afdc6c1e7f4f2829c1cb0dbee0",
    "spike/heldout/mbs8": "3af4633f8b36149fa185a439f99def8d540009823dffb92eb118150e2cab9c1c",
    "spike/seen/mbs2": "0979143edafdbea9b172cfa529697a43bea67db3b9bdecb872a100b81c231d2e",
    "spike/seen/mbs8": "b93882769f63ada3839713aa30ff72533c577ccffb918643888ee049abf72ec6",
}


@pytest.mark.needs_host
@pytest.mark.parametrize("key", sorted(PRE_TERMINATION_GOLDEN))
def test_committed_corpus_simulates_byte_identically_to_before_the_field(key):
    name, split, mbs = key.rsplit("/", 2)
    m, trace = load_trace(name, split, CORPUS_DIR)
    assert all(r.expected_output_tokens is None for r in trace), "no corpus populates it yet"
    res = simulate(trace, SimConfig(max_batch_size=int(mbs[3:])), PLACEHOLDER_A100_E4B)
    rows = json.dumps([asdict(r) for r in res.rows], sort_keys=True)
    assert hashlib.sha256(rows.encode()).hexdigest() == PRE_TERMINATION_GOLDEN[key]
    blob = json.dumps({"rows": rows, "summary": res.summary(m.classes[name])}, sort_keys=True)
    assert hashlib.sha256(blob.encode()).hexdigest() == SUMMARY_GOLDEN[key]


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
