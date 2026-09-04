"""Cost control. An unattended loop that can spend money needs a cap it cannot talk itself past.

Numbers are measured from this project's own Modal usage, not guessed: an A100-80GB sweep runs
3-8 minutes at roughly $0.3-0.5, and a single compile-on startup burned ~21 minutes before it
measured anything.

The rule this enforces is the screening ladder from LOOP.md: never enter tier N+1 while a tier-N
test could still falsify the hypothesis. That is a cost rule as much as a rigour one — the KV
block leak was proven on CPU in about two seconds, where the equivalent GPU sweep would have
cost 20 minutes and real money.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inference_server.research.schemas import REPO_ROOT

LEDGER = REPO_ROOT / "runs" / "budget.json"

# Modal A100-80GB, observed. Update if the instance class or pricing changes.
GPU_USD_PER_HOUR = 3.80

# Typical wall time per tier, from this project's runs. Used to estimate BEFORE spending.
TIER_MINUTES = {1: 0.05, 2: 0.1, 3: 8.0, 4: 25.0}
TIER_NAMES = {1: "static/arithmetic", 2: "CPU repro", 3: "single-GPU probe", 4: "full sweep"}


def estimate_usd(tier: int, minutes: float | None = None) -> float:
    """What a tier-N experiment is expected to cost. Tiers 1-2 are free (no GPU)."""
    if tier <= 2:
        return 0.0
    mins = minutes if minutes is not None else TIER_MINUTES.get(tier, TIER_MINUTES[4])
    return mins / 60.0 * GPU_USD_PER_HOUR


@dataclass
class Budget:
    """A spend cap for one loop iteration, with a ledger so the cap is real across processes."""

    limit_usd: float = 5.0
    spent_usd: float = 0.0
    entries: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def can_afford(self, tier: int, minutes: float | None = None) -> tuple[bool, str]:
        cost = estimate_usd(tier, minutes)
        if cost == 0.0:
            return True, f"tier {tier} ({TIER_NAMES[tier]}) is free"
        if cost > self.remaining_usd:
            return False, (
                f"tier {tier} ({TIER_NAMES[tier]}) needs ~${cost:.2f}, only "
                f"${self.remaining_usd:.2f} left of ${self.limit_usd:.2f}. Falsify at a cheaper "
                f"tier first, or raise the cap deliberately.")
        return True, f"~${cost:.2f} of ${self.remaining_usd:.2f} remaining"

    def record(self, tier: int, label: str, minutes: float | None = None,
               actual_usd: float | None = None) -> float:
        cost = actual_usd if actual_usd is not None else estimate_usd(tier, minutes)
        self.spent_usd += cost
        self.entries.append({"tier": tier, "label": label, "usd": round(cost, 4),
                             "at": time.time()})
        return cost

    # -- persistence ----------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        target = path or LEDGER
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2))
        return target

    @classmethod
    def load(cls, path: Path | None = None, limit_usd: float = 5.0) -> "Budget":
        target = path or LEDGER
        if not target.exists():
            return cls(limit_usd=limit_usd)
        d = json.loads(target.read_text())
        d.setdefault("limit_usd", limit_usd)
        return cls(**d)

    def reset(self, limit_usd: float | None = None) -> "Budget":
        """Start a fresh iteration. Spend does not carry over; the cap is per-iteration."""
        self.limit_usd = limit_usd if limit_usd is not None else self.limit_usd
        self.spent_usd = 0.0
        self.entries = []
        self.started_at = time.time()
        return self


def guard(tier: int, label: str, *, budget: Budget | None = None,
          minutes: float | None = None) -> Budget:
    """Raise unless this experiment is affordable. Call BEFORE launching anything on a GPU."""
    b = budget or Budget.load()
    ok, why = b.can_afford(tier, minutes)
    if not ok:
        raise RuntimeError(f"budget refused '{label}': {why}")
    b.record(tier, label, minutes)
    b.save()
    return b
