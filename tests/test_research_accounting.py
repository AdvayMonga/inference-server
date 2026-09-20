"""Total accounting and the primary metric it makes computable.

Every test here is one of the two failure modes notes/07 names: cost moved off the measured
window, or a resource burned without anyone counting it. The metric's refusals matter more than
its arithmetic — a number derived from an incomplete accounting is a confident wrong answer,
and this project has shipped one of those before.
"""

from __future__ import annotations

import pytest

from inference_server.research.accounting import (
    Accounting,
    peak_rss_gb,
    process_uptime_s,
    storage_read_bytes,
)
from inference_server.research.gates import cost_gate
from inference_server.research.schemas import PANEL_VERSION, Validity, Vitals
from inference_server.research.session import primary_metric


def _v(**over):
    base = dict(engine_sha="abc", dirty=False, harness="replay_trace",
                harness_config={"model": "E2B", "split": "seen", "n_requests": 8},
                workload_regime="cache_miss_heavy", n_samples=8, run_group="g1")
    base.update(over)
    return Validity(**base)


def _acc(**over) -> Accounting:
    base = dict(wall_s_from_process_start=90.0, serving_wall_s=12.0, sessions_served=8,
                peak_host_rss_gb=11.2, peak_device_mem_gb=6.4,
                device_mem_source="sampled", storage_read_bytes=None,
                unmeasured={"storage_read_bytes": "no /proc on darwin"})
    base.update(over)
    return Accounting(**base)


_DEFAULT = object()      # "the usual accounting"; None means a panel that carries none


def _panel(acc: Accounting | None = _DEFAULT, **over) -> Vitals:
    validity = over.pop("validity", None) or _v()
    d = dict(ttft_p95=1400.0, slo_ttft_ms=2000.0, tpot_p95=40.0, wall_s=12.0, n_failed=0)
    d.update(over)
    p = Vitals(validity=validity, **d)
    acc = _acc() if acc is _DEFAULT else acc
    if acc is not None:
        acc.apply(p)
    return p


# ---------------------------------------------------------------- measuring the terms

def test_rss_unit_differs_by_platform(monkeypatch):
    """ru_maxrss is bytes on Darwin and kibibytes on Linux; confusing them is a 1024x error."""
    import inference_server.research.accounting as A

    class _R:
        ru_maxrss = 12_000_000_000
    monkeypatch.setattr(A.resource, "getrusage", lambda _who: _R())
    monkeypatch.setattr(A.sys, "platform", "darwin")
    assert peak_rss_gb() == 12.0
    monkeypatch.setattr(A.sys, "platform", "linux")
    assert peak_rss_gb() == 12000.0


def test_storage_read_bytes_names_its_gap_rather_than_guessing():
    value, why = storage_read_bytes()
    assert (value is None) == bool(why), "a None must arrive with a reason, a number without one"
    if value is None:
        assert "stdlib" in why


def test_process_uptime_is_measured_from_import_not_from_now():
    assert process_uptime_s() >= 0.0


def test_idle_time_is_derivable_not_baked_in():
    """'GPU-seconds allocated, including idle time in warm pools' is what makes pooling cost."""
    a = _acc(wall_s_from_process_start=90.0, serving_wall_s=12.0)
    assert a.idle_s == 78.0
    assert a.serving_fraction == 0.1333
    assert Accounting(wall_s_from_process_start=90.0).idle_s is None


def test_accounting_mirrors_into_the_panel_fields_the_cost_gate_already_reads():
    p = _panel()
    assert p.peak_host_rss_gb == 11.2 and p.peak_gpu_mem_gb == 6.4
    assert p.accounting["idle_s"] == 78.0


def test_accounting_does_not_clobber_a_value_the_instrument_already_measured():
    p = Vitals(validity=_v(), peak_gpu_mem_gb=1.0)
    _acc().apply(p)
    assert p.peak_gpu_mem_gb == 1.0


# ---------------------------------------------------------------- the panel contract

def test_the_accounting_block_round_trips_without_moving_panel_version(tmp_path):
    p = _panel()
    p.to_json(tmp_path / "p.json")
    back = Vitals.load(tmp_path / "p.json")
    assert back.panel_version == PANEL_VERSION
    assert Accounting.of(back).sessions_served == 8


def test_a_panel_written_before_accounting_existed_still_loads(tmp_path):
    """Optional None-default field, same precedent as the batch-occupancy fields: an added
    field cannot change a measurement already taken."""
    p = Vitals(validity=_v(), ttft_p95=100.0)
    p.to_json(tmp_path / "old.json")
    blob = (tmp_path / "old.json").read_text().replace('"accounting": null,\n  ', "")
    (tmp_path / "old.json").write_text(blob)
    assert Accounting.of(Vitals.load(tmp_path / "old.json")) is None


# ---------------------------------------------------------------- the primary metric

def test_primary_metric_is_gpu_seconds_per_session_at_a_stated_ceiling():
    m = primary_metric(_panel())
    assert m and m.gpu_s_per_session == 11.25      # 90 GPU-s / 8 sessions
    assert m.ceiling_ms == 2000.0 and m.ceiling_met is True
    assert "11.25 GPU-seconds per session" in m.format()


def test_a_broken_ceiling_still_produces_a_number_and_says_it_was_broken():
    m = primary_metric(_panel(ttft_p95=9000.0))
    assert m and m.ceiling_met is False and "BROKEN" in m.format()


def test_the_ceiling_can_be_stated_by_the_caller():
    assert primary_metric(_panel(), ceiling_ms=500.0).ceiling_met is False


def test_it_refuses_rather_than_falling_back_to_the_serving_window():
    """wall_s starts at the first arrival. Substituting it would drop the model load — which is
    exactly the 'move cost off the measured window' hack this accounting exists to block."""
    p = _panel(_acc(wall_s_from_process_start=None))
    assert p.wall_s == 12.0, "the serving window is present and tempting"
    m = primary_metric(p)
    assert not m and any("not a substitute" in r for r in m.refusals)


def test_it_refuses_a_panel_with_no_accounting_at_all():
    m = primary_metric(_panel(None))
    assert not m and any("no accounting block" in r for r in m.refusals)


def test_it_refuses_when_nothing_says_how_many_sessions_were_served():
    m = primary_metric(_panel(_acc(sessions_served=None)))
    assert not m and any("sessions_served" in r for r in m.refusals)


def test_it_refuses_without_a_ceiling_because_cost_alone_degenerates():
    m = primary_metric(_panel(slo_ttft_ms=None))
    assert not m and any("ceiling" in r for r in m.refusals)


def test_it_refuses_when_the_ceiling_cannot_be_checked():
    m = primary_metric(_panel(ttft_p95=None))
    assert not m and any("ttft_p95" in r for r in m.refusals)


def test_unmeasured_terms_travel_with_the_number_as_caveats():
    m = primary_metric(_panel())
    assert m and any("storage_read_bytes" in c for c in m.caveats)
    assert "CAVEAT" in m.format()


def _with_failures(n_requests: int, failed: int) -> Vitals:
    cfg = {"model": "E2B", "split": "seen", "n_requests": n_requests}
    return _panel(n_failed=failed,
                  validity=_v(n_samples=n_requests - failed, harness_config=cfg))


def test_failed_requests_count_against_the_ceiling_not_for_it():
    """ttft_p95 is over successful requests only, so dropping requests would flatter it. Measured
    on MPS: 2 of 8 cold_start requests returned no tokens while the survivors' p95 read 464ms."""
    m = primary_metric(_with_failures(8, 2))
    assert m and m.ceiling_met is False
    assert any("2 of 8 requests produced no first token" in c for c in m.caveats)


@pytest.mark.parametrize("n_requests, failed, met", [
    (8, 1, False),     # the floor-interpolation index let this through: int(0.95 * 7) = 6 < 7
    (9, 1, False),     # likewise: int(0.95 * 8) = 7 < 8
    (20, 1, True),     # exactly 5% failed: 19 of 20 answered, so nearest-rank p95 is finite
    (20, 2, False),
    (100, 5, True),
    (100, 6, False),
])
def test_one_failure_breaks_the_ceiling_exactly_when_it_reaches_the_p95_rank(
        n_requests, failed, met):
    """The adversarial case is ONE dropped request at small n. Pinned from both sides."""
    m = primary_metric(_with_failures(n_requests, failed))
    assert m and m.ceiling_met is met
    assert any(f"{failed} of {n_requests}" in c for c in m.caveats)


@pytest.mark.parametrize("n_failed", [None, -1])
def test_it_refuses_when_the_failure_count_is_unknown_or_impossible(n_failed):
    """A panel that cannot say how many requests failed cannot be held to a ceiling: an
    uncounted failure is the hack this check exists to stop."""
    m = primary_metric(_panel(n_failed=n_failed))
    assert not m and any("n_failed" in r for r in m.refusals)


def test_several_runs_aggregate_and_every_one_must_meet_the_ceiling():
    a, b = _panel(), _panel(ttft_p95=9000.0)
    m = primary_metric([a, b])
    assert m.gpu_s == 180.0 and m.sessions == 16 and m.gpu_s_per_session == 11.25
    assert m.ceiling_met is False and m.runs == 2


def test_panels_stating_different_ceilings_are_not_one_metric():
    m = primary_metric([_panel(), _panel(slo_ttft_ms=200.0)])
    assert not m and any("one SLO" in r for r in m.refusals)


# ---------------------------------------------------------------- the cost gate

def test_cost_gate_is_unchanged_when_neither_arm_carries_an_accounting_block():
    """Every existing experiment must judge exactly as it did before."""
    a = Vitals(validity=_v(), graph_capture_s=1.0, peak_gpu_mem_gb=10.0)
    b = Vitals(validity=_v(), graph_capture_s=1.0, peak_gpu_mem_gb=10.0)
    g = cost_gate(a, b)
    assert g.passed and "cannot see cost moved outside" in g.reason


def test_cost_gate_catches_host_ram_burned_for_the_win():
    """The resource the gate populated but never read until now."""
    a = _panel(_acc(peak_host_rss_gb=11.0))
    b = _panel(_acc(peak_host_rss_gb=19.0))
    g = cost_gate(a, b)
    assert not g.passed and "peak host RSS +8.0GB" in g.reason


def test_cost_gate_catches_a_win_paid_for_before_the_measured_window():
    a = _panel(_acc(wall_s_from_process_start=90.0))
    b = _panel(_acc(wall_s_from_process_start=400.0))
    g = cost_gate(a, b)
    assert not g.passed and "total wall from process start" in g.reason


def test_cost_gate_is_conservative_about_wall_clock():
    """A gate that fires spuriously is worse than one that fires late: a short run that grows by
    a large FRACTION but a small number of seconds is not evidence of anything."""
    a = _panel(_acc(wall_s_from_process_start=8.0))
    b = _panel(_acc(wall_s_from_process_start=16.0))       # +100%, but only +8s
    assert cost_gate(a, b).passed


def test_cost_gate_reports_what_nobody_measured():
    g = cost_gate(_panel(), _panel())
    assert g.passed and "storage_read_bytes" in g.reason
    assert g.evidence["unmeasured"] == ["storage_read_bytes"]
