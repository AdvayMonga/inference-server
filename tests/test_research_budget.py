"""Cost control: the cap must be enforceable, not advisory."""

from __future__ import annotations

import pytest

from inference_server.research.budget import (
    VENUES,
    Budget,
    Unroutable,
    estimate_usd,
    guard,
    parse_venues,
    route,
)
from inference_server.research.schemas import Hypothesis, SchemaError


def _hyp(tier: int, requires: list[str] | None = None) -> Hypothesis:
    return Hypothesis(statement="s", gap_id="g", predicted_metric="ttft_p95",
                      predicted_direction="decrease", predicted_magnitude=">10%",
                      falsification_tier=tier, falsification_test="t", requires=requires or [])


LOCAL = [VENUES["local-cpu"], VENUES["local-mps"]]


def test_cpu_tiers_are_free_so_the_ladder_is_always_affordable():
    """The screening ladder is a cost rule too: the KV leak was proven on CPU in ~2s."""
    assert estimate_usd(1) == 0.0
    assert estimate_usd(2) == 0.0
    assert estimate_usd(3) > 0.0
    assert estimate_usd(4) > estimate_usd(3)


def test_budget_refuses_a_sweep_it_cannot_afford():
    b = Budget(limit_usd=0.50)
    ok, why = b.can_afford(4)
    assert not ok and "cheaper tier first" in why


def test_cheap_tiers_still_run_when_broke():
    b = Budget(limit_usd=0.0)
    assert b.can_afford(2)[0]
    assert not b.can_afford(3)[0]


def test_spending_accumulates_and_then_blocks():
    b = Budget(limit_usd=2.0)
    b.record(3, "probe a")
    b.record(3, "probe b")
    assert b.spent_usd == pytest.approx(2 * estimate_usd(3))
    assert not b.can_afford(4)[0]


def test_guard_raises_rather_than_warning(tmp_path):
    b = Budget(limit_usd=0.10)
    with pytest.raises(RuntimeError, match="budget refused"):
        guard(4, "full sweep", budget=b)


def test_ledger_survives_across_processes(tmp_path):
    p = tmp_path / "budget.json"
    b = Budget(limit_usd=5.0)
    b.record(3, "probe")
    b.save(p)
    again = Budget.load(p)
    assert again.spent_usd == pytest.approx(b.spent_usd)
    assert again.entries[0]["label"] == "probe"


def test_reset_starts_a_fresh_iteration():
    b = Budget(limit_usd=5.0)
    b.record(4, "sweep")
    assert b.spent_usd > 0
    b.reset()
    assert b.spent_usd == 0.0 and b.entries == []


# -- placement: tier says how much, `requires` says where -----------------------------

def test_cheap_tiers_route_to_cpu_even_when_a_gpu_is_present():
    assert route(_hyp(2), LOCAL).venue.name == "local-cpu"


def test_gpu_probe_without_cuda_needs_stays_local_and_free():
    """A scheduler-logic probe is tier 3 but needs no CUDA; MPS falsifies it for $0."""
    pl = route(_hyp(3), LOCAL + [VENUES["vast-4090"]])
    assert pl.venue.name == "local-mps" and pl.est_usd == 0.0


def test_kernel_work_routes_to_the_cheapest_cuda_venue():
    pl = route(_hyp(3, ["triton"]), LOCAL + [VENUES["modal-a100"], VENUES["vast-4090"]])
    assert pl.venue.name == "vast-4090"
    assert pl.est_usd == pytest.approx(estimate_usd(3, usd_per_hour=0.35))


def test_large_vram_skips_the_consumer_card():
    pl = route(_hyp(4, ["large_vram"]), LOCAL + [VENUES["vast-4090"], VENUES["runpod-a100"]])
    assert pl.venue.name == "runpod-a100"


def test_unroutable_says_what_to_enable():
    with pytest.raises(Unroutable, match="RESEARCH_VENUES=vast-4090"):
        route(_hyp(3, ["cuda"]), LOCAL)


def test_unknown_requirement_is_a_schema_error():
    with pytest.raises(SchemaError, match="requires"):
        _hyp(2, ["gpu_go_brr"]).validate()


def test_unknown_venue_fails_loudly():
    with pytest.raises(ValueError, match="catalogue"):
        parse_venues("modal-h100")


def test_guard_prices_by_venue_not_by_modal(tmp_path, monkeypatch):
    monkeypatch.setattr("inference_server.research.budget.LEDGER", tmp_path / "budget.json")
    b = Budget(limit_usd=0.10)
    with pytest.raises(RuntimeError):
        guard(3, "probe", budget=b, venue=VENUES["modal-a100"])
    guard(3, "probe", budget=b, venue=VENUES["vast-4090"])
    assert b.entries[-1]["venue"] == "vast-4090"
