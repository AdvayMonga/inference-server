"""Cost control: the cap must be enforceable, not advisory."""

from __future__ import annotations

import pytest

from inference_server.research.budget import Budget, estimate_usd, guard


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
