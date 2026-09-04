"""Acceptance: the loop must independently reach conclusions we already know are right.

These use REAL numbers measured on A100/E4B during the sessions that produced them, replayed
through the loop's own stages. This tests the loop's reasoning, not the engine.

Two cases, deliberately one of each kind:
  * a genuine win  — prefill CUDA graph on prefix hits halved TTFT p50
  * a genuine dud  — length-grouped prefill waves were inert, because 83-91% of waves are K=1

A loop that only recognises wins is a rubber stamp. Recognising the dud is the harder half and
is what stopped two planned projects.
"""

from __future__ import annotations

from inference_server.research.attribute import attribute
from inference_server.research.gates import judge
from inference_server.research.schemas import Hypothesis, Validity, Vitals

SLO_TTFT, SLO_TPOT = 200.0, 50.0

CFG = {"model": "google/gemma-4-E4B-it", "gpu": "A100-80GB", "max_batch_size": 256,
       "prefill_mode": "batched", "compile": "0", "prefill_graph": "1", "blocks": "8192",
       "sliding_blocks": "4096", "context_window": "8192", "rates": "1", "duration": 30,
       "pool_size": 4000, "max_queue_wait_s": 30.0, "prefix_cache_impl": "radix",
       "wave_window_mult": 0}


def _panel(*, run_group="acceptance", regime="cache_miss_heavy", n=24, stderr=6.0,
           prefill_graph="1", wave_window_mult=0, **fields):
    cfg = {**CFG, "prefill_graph": prefill_graph, "wave_window_mult": wave_window_mult}
    return Vitals(
        validity=Validity(engine_sha="acc0000", dirty=False, harness="bench_serving",
                          harness_config=cfg, workload_regime=regime, n_samples=n,
                          run_group=run_group, stderr=stderr),
        slo_ttft_ms=SLO_TTFT, slo_tpot_ms=SLO_TPOT, **fields)


# --------------------------------------------------------------- case 1: the real win

def test_loop_confirms_the_prefill_graph_win():
    """Measured: TTFT p50 98 -> 46ms at rate 1 when the prefill graph started firing on
    prefix-cache hits, with throughput up and no new errors."""
    before = _panel(prefill_graph="0", ttft_p50=98.0, ttft_p95=158.0,
                    ttft_queue_p50=10.0, ttft_queue_p95=25.0,
                    ttft_prefill_p50=88.0, ttft_prefill_p95=133.0,
                    tpot_p50=23.1, tpot_p95=27.3, tok_s_within_slo=117.0,
                    total_iteration_errors=0, cache_lookups=200, cache_hit_rate=0.03,
                    wave_sizes={"1": 20, "2": 3})
    after = _panel(prefill_graph="0", ttft_p50=46.0, ttft_p95=109.0,
                   ttft_queue_p50=10.0, ttft_queue_p95=25.0,
                   ttft_prefill_p50=36.0, ttft_prefill_p95=84.0,
                   tpot_p50=22.0, tpot_p95=25.0, tok_s_within_slo=121.1,
                   total_iteration_errors=0, cache_lookups=200, cache_hit_rate=0.03,
                   wave_sizes={"1": 20, "2": 3})

    hyp = Hypothesis(
        statement="Let the K=1 prefill CUDA graph fire on prefix-cache hits, not just misses",
        gap_id="ttft-prefill", predicted_metric="ttft_p50", predicted_direction="decrease",
        predicted_magnitude=">30%", falsification_tier=3,
        falsification_test="single-GPU prefill A/B on a warm cache")

    j = judge(hyp, before, after, run_tests=False)
    assert j.passed, {k: g.reason for k, g in j.gates.items()}
    assert j.verdict == "confirmed"
    assert j.gates["significance"].evidence["pct"] < -50   # halved


# --------------------------------------------------------------- case 2: the real dud

def test_loop_rejects_length_grouped_waves_as_untestable_on_this_workload():
    """Measured: 83-91% of prefill waves are K=1, so there is no padding to save and the A/B
    was noise. The loop must reach 'invalid' — the harness cannot test this — rather than
    reading the small delta as a win."""
    waves = {"1": 264, "2": 41, "3": 7, "4": 2, "5": 2}   # measured at rate 6
    before = _panel(wave_window_mult=0, ttft_p95=405.0, tok_s_within_slo=529.1,
                    ttft_queue_p95=60.0, ttft_prefill_p95=345.0,
                    wave_sizes=waves, total_iteration_errors=0)
    after = _panel(wave_window_mult=0, ttft_p95=352.0, tok_s_within_slo=565.0,
                   ttft_queue_p95=58.0, ttft_prefill_p95=294.0,
                   wave_sizes=waves, total_iteration_errors=0)

    hyp = Hypothesis(
        statement="Group similar prompt lengths into a prefill wave to cut padding waste",
        gap_id="prefill-padding", predicted_metric="wave_sizes",
        predicted_direction="increase", predicted_magnitude="wider waves",
        falsification_tier=1,
        falsification_test="wave-size histogram: are waves ever wider than K=1?")

    j = judge(hyp, before, after, run_tests=False)
    assert not j.passed
    assert j.verdict == "invalid", j.gates["validity"].reason
    assert "K=1" in j.gates["validity"].reason


def test_attribution_finds_the_k1_harness_limit_on_its_own():
    """Step 1 must surface 'this run cannot test wave composition' before anyone proposes it."""
    panel = _panel(ttft_p95=405.0, ttft_queue_p95=60.0, ttft_prefill_p95=345.0,
                   wave_sizes={"1": 264, "2": 41, "3": 7})
    ids = [g.id for g in attribute(panel)]
    assert "harness-k1" in ids


def test_attribution_blames_prefill_not_queueing():
    """Measured at rate 1: queue p95 21ms vs prefill p95 203ms. The loop must say prefill, which
    is what ruled out every scheduling lever at once."""
    panel = _panel(ttft_p95=213.0, ttft_queue_p95=21.0, ttft_prefill_p95=203.0)
    gaps = attribute(panel)
    assert gaps and gaps[0].id == "ttft-prefill"
    assert "prefill" in gaps[0].title


def test_attribution_flags_a_thrashing_cache():
    """Measured under load: hit_rate 0.0015 with 22589 evictions against 1024 blocks held."""
    panel = _panel(cache_hit_rate=0.0015, cache_lookups=3000, cache_evictions=22589,
                   cache_blocks_held=1024, cache_max_blocks=1024)
    gap = next(g for g in attribute(panel) if g.id == "cache-hit-rate")
    assert "thrashing" in gap.title


def test_attribution_puts_stability_above_everything():
    """Correctness before performance: iteration errors must outrank any latency gap."""
    panel = _panel(ttft_p95=900.0, ttft_queue_p95=100.0, ttft_prefill_p95=800.0,
                   total_iteration_errors=3)
    assert attribute(panel)[0].id == "stability"
