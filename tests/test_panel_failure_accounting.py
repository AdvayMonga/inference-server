"""Panel version 2: a failure to serve is a measurement, not an absence of one.

Under version 1 `ttft_p95` was taken over the requests that answered, so an arm could improve it
by shedding the requests it would have been slowest on — the OPEN GAP `kb-20260918-4f4c85b7` left
for a `PANEL_VERSION` bump. These tests pin the two halves of the closure: the percentile counts
failures, and the client's blindness to empty-decoding tokens is not mistaken for one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from inference_server.research import harness as H
from inference_server.research.accounting import Accounting
from inference_server.research.compare import significance_replicated
from inference_server.research.corpus import WorkloadClass
from inference_server.research.schemas import PANEL_VERSION, Validity, Vitals
from inference_server.research.session import primary_metric

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "bench"))
import replay_trace as rt  # noqa: E402

CLS = WorkloadClass("steady_interactive", "", 200.0, 50.0, 4.0, "a", "b")


# --------------------------------------------------------------------------- the percentile

def test_with_nothing_failed_it_is_the_percentile_we_always_had():
    xs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    for q in (0.5, 0.95):
        assert H.pct_over_attempts(xs, q, len(xs)) == H.pct(xs, q)


def test_shedding_the_slowest_requests_cannot_lower_the_percentile():
    """The reward hack, in one assertion. Serving everything must not read worse than serving
    the fast half — under version 1 the second line was 40ms against the first line's 80ms."""
    served_all = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    shed_the_slow_half = [10.0, 20.0, 30.0, 40.0]
    honest = H.pct_over_attempts(shed_the_slow_half, 0.95, 8)
    assert honest > H.pct_over_attempts(served_all, 0.95, 8)
    assert H.pct(shed_the_slow_half, 0.95) < H.pct(served_all, 0.95)   # the old, backwards answer


def test_the_percentile_is_infinite_when_the_tail_falls_among_the_failures():
    """Not a large number and not None: nothing this run measured is as bad as never answering,
    and the p50 of the same run stays finite, so the panel still says something."""
    assert H.pct_over_attempts([10.0, 20.0, 30.0, 40.0], 0.95, 8) == math.inf
    assert H.pct_over_attempts([10.0, 20.0, 30.0, 40.0], 0.50, 8) == 40.0
    # ONE failure of eight is enough: ceil(0.95 * 8) - 1 = 7 is the eighth attempt, which never
    # answered. Under the floor convention this read 70.0 and said nothing had gone wrong.
    assert H.pct_over_attempts([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], 0.95, 8) == math.inf
    assert H.pct_over_attempts([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], 0.95, 8) == 80.0


@pytest.mark.parametrize("n, failed", [(2, 1), (8, 0), (8, 1), (8, 2), (9, 1), (20, 1),
                                       (20, 2), (100, 5), (100, 6)])
def test_the_percentile_and_primary_metrics_failure_rule_are_the_same_rule(n, failed):
    """They were written independently and are now provably identical: the percentile is
    infinite exactly when `primary_metric` says the ceiling is broken by failures alone.
    Before this they disagreed at n=8, where primary_metric broke and the percentile did not."""
    served = [float(i + 1) for i in range(n - failed)]
    infinite = H.pct_over_attempts(served, 0.95, n) == math.inf
    v = Validity(engine_sha="a", dirty=False, harness="replay_trace",
                 harness_config={"model": "E2B"}, workload_regime="cache_miss_heavy",
                 n_samples=n - failed, run_group="g")
    panel = Vitals(validity=v, ttft_p95=1.0, slo_ttft_ms=1e9, n_failed=failed)
    Accounting(wall_s_from_process_start=90.0, serving_wall_s=12.0, sessions_served=n,
               peak_host_rss_gb=1.0, peak_device_mem_gb=1.0, device_mem_source="x",
               storage_read_bytes=None, unmeasured={}).apply(panel)
    assert infinite is (primary_metric(panel).ceiling_met is False)


# --------------------------------------------------------------------------- the replay summary

def _row(i: int, *, ttft: float | None = None, error: str | None = None,
         out: int = 5, server_out: int | None = None) -> rt.Row:
    return rt.Row(i, f"s-{i}", 0, f"t-{i}", 0.0, 0.0, ttft, 1.0 if ttft else None,
                  out, error, 12, server_out)


def _result(rows: list[rt.Row]) -> rt.ReplayResult:
    res = rt.ReplayResult(rows=rows)
    res.wall_s = 10.0
    return res


def test_a_shed_request_counts_in_the_denominator_and_breaks_the_slo():
    rows = [_row(i, ttft=float(10 * (i + 1)), server_out=5) for i in range(6)]
    rows += [_row(6, error="http 429", out=0), _row(7, error="drain_timeout", out=0)]
    s = _result(rows).summary(CLS)
    assert s["n_ok"] == 6 and s["n_failed"] == 2 and s["n_blind"] == 0
    assert s["ttft_p95"] == math.inf and s["within_slo"] is False


def test_a_request_the_client_could_not_see_is_not_a_request_the_server_shed():
    """The server answered; the shim emitted no chunk because every token decoded to "".
    Charging it as a shed would manufacture an SLO breach out of an instrument defect."""
    rows = [_row(i, ttft=float(10 * (i + 1)), server_out=5) for i in range(7)]
    rows += [_row(7, error="no_visible_tokens", out=0, server_out=3)]
    s = _result(rows).summary(CLS)
    assert s["n_failed"] == 0 and s["n_blind"] == 1
    assert s["ttft_p95"] == 70.0 and s["within_slo"] is True   # p95 of 7, not of 8
    # But a server that genuinely produced nothing IS a failure.
    rows[-1] = _row(7, error="no_tokens", out=0, server_out=0)
    assert _result(rows).summary(CLS)["n_failed"] == 1


def test_throughput_counts_the_tokens_the_engine_generated_not_the_ones_the_client_saw():
    rows = [_row(i, ttft=10.0, out=4, server_out=5) for i in range(4)]
    s = _result(rows).summary(CLS)
    assert s["tok_per_s"] == 2.0          # 4 x 5 server tokens / 10s, not 4 x 4
    assert s["invisible_tokens"] == 4     # one per request, and the panel says so


def test_the_panel_carries_both_counts():
    rows = [_row(i, ttft=10.0, out=4, server_out=5) for i in range(7)]
    rows += [_row(7, error="http 429", out=0)]
    panel = rt.build_panel(_result(rows), CLS, corpus_version="c" * 64, split="seen",
                           rate_scale=1.0, scheduler_stats={}, cache_stats={})
    assert panel.panel_version == PANEL_VERSION
    assert panel.n_failed == 1 and panel.invisible_tokens == 7


# --------------------------------------------------------------------------- the gate


def _panel(ttft_p95: float, arm: str) -> Vitals:
    v = Validity(engine_sha="abc", dirty=False, harness="replay_trace",
                 harness_config={"model": "E2B", "rates": 1.0}, workload_regime="cache_miss_heavy",
                 n_samples=8, run_group="g1", notes=f"arm={arm}")
    return Vitals(validity=v, ttft_p95=ttft_p95)


def test_an_arm_that_did_not_serve_the_workload_cannot_claim_a_latency_win():
    """inf through the t-test gives nan, which reads as `noise` — the right refusal for the
    wrong reason. Say what actually happened instead."""
    base = [_panel(x, "baseline") for x in (900.0, 800.0, 810.0, 820.0)]
    treat = [_panel(x, "treatment") for x in (900.0, math.inf, math.inf, math.inf)]
    s = significance_replicated(base, treat, "ttft_p95", direction="decrease")
    assert s.verdict == "unmeasurable" and "treatment" in s.detail
    assert "failure to serve" in s.detail
