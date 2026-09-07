"""Cost control and placement. An unattended loop that can spend money needs a cap it cannot
talk itself past, and a rule for WHERE each test runs so the cap is rarely even approached.

Numbers are measured from this project's own Modal usage, not guessed: an A100-80GB sweep runs
3-8 minutes at roughly $0.3-0.5, and a single compile-on startup burned ~21 minutes before it
measured anything. Cloud prices are on-demand list prices at the time of writing; edit VENUES.

Two rules, enforced not remembered:
- the screening ladder from LOOP.md: never enter tier N+1 while a tier-N test could still
  falsify the hypothesis. The KV block leak was proven on CPU in about two seconds, where the
  equivalent GPU sweep would have cost 20 minutes and real money.
- the least capable machine that can falsify a hypothesis is the right one. A scheduler change
  needs no CUDA; a Triton kernel needs nothing else. `route()` picks by requirements, not habit.

Estimates assume tier wall time is venue-independent. It is not (a 4090 is slower than an A100
at batch 256), but the error is a factor, and the cap exists to stop an order of magnitude.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inference_server.research.schemas import REPO_ROOT, Hypothesis

LEDGER = REPO_ROOT / "runs" / "budget.json"

# Modal A100-80GB, observed. Kept as the default so old call sites still price a GPU minute.
GPU_USD_PER_HOUR = 3.80

# Typical wall time per tier, from this project's runs. Used to estimate BEFORE spending.
TIER_MINUTES = {1: 0.05, 2: 0.1, 3: 8.0, 4: 25.0}
TIER_NAMES = {1: "static/arithmetic", 2: "CPU repro", 3: "single-GPU probe", 4: "full sweep"}


# -- venues ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Venue:
    """Somewhere a test can run: what it costs and what it can do."""

    name: str
    usd_per_hour: float
    capabilities: frozenset[str]
    local: bool = False


_CUDA = frozenset({"gpu", "cuda", "triton", "cuda_graphs", "compile", "vllm", "linux", "bf16"})

VENUES: dict[str, Venue] = {v.name: v for v in (
    Venue("local-cpu", 0.0, frozenset(), local=True),
    Venue("local-mps", 0.0, frozenset({"gpu", "bf16"}), local=True),
    Venue("local-cuda", 0.0, _CUDA, local=True),
    Venue("vast-4090", 0.35, _CUDA),                       # 24 GB; E4B bf16 + KV pool fits
    Venue("runpod-a100", 1.30, _CUDA | {"large_vram"}),    # 80 GB; comparable to the old CSVs
    Venue("modal-a100", GPU_USD_PER_HOUR, _CUDA | {"large_vram"}),
)}

VENUES_ENV = "RESEARCH_VENUES"


def local_venues() -> list[Venue]:
    """What this machine can run, by asking torch rather than assuming."""
    import torch  # lazy: keeps `loop screen` fast on the no-GPU path
    out = [VENUES["local-cpu"]]
    if torch.backends.mps.is_available():
        out.append(VENUES["local-mps"])
    if torch.cuda.is_available():
        out.append(VENUES["local-cuda"])
    return out


def parse_venues(spec: str) -> list[Venue]:
    """'vast-4090,runpod-a100' -> venues. Unknown names fail loudly with the catalogue."""
    out = []
    for name in (n.strip() for n in spec.split(",") if n.strip()):
        if name not in VENUES:
            raise ValueError(f"unknown venue {name!r}; catalogue: {', '.join(VENUES)}")
        out.append(VENUES[name])
    return out


def available_venues(cloud: str | None = None) -> list[Venue]:
    """Local venues torch confirms, plus cloud venues named explicitly (env or argument).

    Cloud is opt-in because whether you can pay for it is not something code can detect —
    this project ran out of Modal credits with modal-a100 still hardcoded as the only price.
    """
    spec = cloud if cloud is not None else os.environ.get(VENUES_ENV, "")
    return local_venues() + parse_venues(spec)


# -- estimates -------------------------------------------------------------------------

def estimate_usd(tier: int, minutes: float | None = None,
                 usd_per_hour: float = GPU_USD_PER_HOUR) -> float:
    """What a tier-N experiment is expected to cost. Tiers 1-2 are free (no GPU)."""
    if tier <= 2:
        return 0.0
    mins = minutes if minutes is not None else TIER_MINUTES.get(tier, TIER_MINUTES[4])
    return mins / 60.0 * usd_per_hour


# -- placement -------------------------------------------------------------------------

class Unroutable(RuntimeError):
    """No available venue can run this test. The message says what to enable."""


@dataclass
class Placement:
    hypothesis_id: str
    venue: Venue
    tier: int
    est_usd: float
    reason: str


def requirements(h: Hypothesis) -> frozenset[str]:
    """Declared capabilities, plus the implicit one: tier 3 and 4 need a GPU of some kind."""
    need = set(h.requires)
    if h.falsification_tier >= 3:
        need.add("gpu")
    return frozenset(need)


def route(h: Hypothesis, venues: list[Venue] | None = None) -> Placement:
    """Cheapest venue that satisfies the hypothesis; ties go to the least capable machine."""
    venues = venues if venues is not None else available_venues()
    need = requirements(h)
    fits = [v for v in venues if need <= v.capabilities]
    if not fits:
        gaps = {v.name: sorted(need - v.capabilities) for v in venues}
        nearest = min(gaps, key=lambda n: len(gaps[n])) if gaps else None
        could = [n for n, v in VENUES.items() if need <= v.capabilities and not v.local]
        raise Unroutable(
            f"{h.id} needs {sorted(need)}; no available venue offers "
            f"{gaps[nearest] if nearest else sorted(need)}. Cloud venues are opt-in: "
            f"{VENUES_ENV}={','.join(could) or '<none in catalogue>'}")
    v = min(fits, key=lambda v: (v.usd_per_hour, len(v.capabilities)))
    cost = estimate_usd(h.falsification_tier, usd_per_hour=v.usd_per_hour)
    why = ("free" if cost == 0.0 else f"~${cost:.2f}") + f", needs {sorted(need) or 'nothing'}"
    return Placement(h.id, v, h.falsification_tier, cost, why)


# -- the cap ---------------------------------------------------------------------------

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

    def can_afford(self, tier: int, minutes: float | None = None,
                   usd_per_hour: float = GPU_USD_PER_HOUR) -> tuple[bool, str]:
        cost = estimate_usd(tier, minutes, usd_per_hour)
        if cost == 0.0:
            return True, f"tier {tier} ({TIER_NAMES[tier]}) is free"
        if cost > self.remaining_usd:
            return False, (
                f"tier {tier} ({TIER_NAMES[tier]}) needs ~${cost:.2f}, only "
                f"${self.remaining_usd:.2f} left of ${self.limit_usd:.2f}. Falsify at a cheaper "
                f"tier first, or raise the cap deliberately.")
        return True, f"~${cost:.2f} of ${self.remaining_usd:.2f} remaining"

    def record(self, tier: int, label: str, minutes: float | None = None,
               actual_usd: float | None = None, venue: str | None = None,
               usd_per_hour: float = GPU_USD_PER_HOUR) -> float:
        cost = actual_usd if actual_usd is not None else estimate_usd(tier, minutes, usd_per_hour)
        self.spent_usd += cost
        self.entries.append({"tier": tier, "label": label, "usd": round(cost, 4),
                             "venue": venue, "at": time.time()})
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
          minutes: float | None = None, venue: Venue | None = None) -> Budget:
    """Raise unless this experiment is affordable. Call BEFORE launching anything on a GPU."""
    b = budget or Budget.load()
    rate = venue.usd_per_hour if venue else GPU_USD_PER_HOUR
    ok, why = b.can_afford(tier, minutes, rate)
    if not ok:
        raise RuntimeError(f"budget refused '{label}': {why}")
    b.record(tier, label, minutes, venue=venue.name if venue else None, usd_per_hour=rate)
    b.save()
    return b
