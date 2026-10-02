"""A dollar budget the run cannot talk itself past: model tokens plus GPU time at the provider's rate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lab import ledger


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    cap_usd: float
    spent_usd: float = 0.0

    @classmethod
    def resume(cls, cap_usd: float, run: str, root: Path = ledger.ROOT) -> "Budget":
        """What this run has already spent, summed from its ledger records."""
        spent = sum(float((r.get("cost") or {}).get("usd", 0.0)) for r in ledger.records(root, run=run))
        return cls(cap_usd, spent)

    @property
    def remaining_usd(self) -> float:
        return max(self.cap_usd - self.spent_usd, 0.0)

    def charge(self, usd: float, what: str) -> None:
        """Record `usd` against the cap; refuse if it would cross."""
        if usd < 0:
            raise ValueError("a charge is never negative")
        if self.spent_usd + usd > self.cap_usd + 1e-9:
            raise BudgetExceeded(f"{what}: ${usd:.2f} would exceed the remaining ${self.remaining_usd:.2f}")
        self.spent_usd += usd

    @staticmethod
    def gpu_usd(seconds: float, price_per_hour: float) -> float:
        return seconds / 3600.0 * price_per_hour
