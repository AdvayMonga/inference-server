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
from dataclasses import asdict, dataclass, field, replace
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
    mem_gb: float | None = None   # device memory; None = unknown, never filtered on
    unified: bool = False         # Apple silicon: host and device share one pool


_CUDA = frozenset({"gpu", "cuda", "triton", "cuda_graphs", "compile", "vllm", "linux", "bf16"})

VENUES: dict[str, Venue] = {v.name: v for v in (
    Venue("local-cpu", 0.0, frozenset(), local=True, unified=True),
    Venue("local-mps", 0.0, frozenset({"gpu", "bf16"}), local=True, unified=True),
    Venue("local-cuda", 0.0, _CUDA, local=True),
    Venue("vast-4090", 0.35, _CUDA, mem_gb=24.0),
    Venue("runpod-a100", 1.30, _CUDA | {"large_vram"}, mem_gb=80.0),
    Venue("modal-a100", GPU_USD_PER_HOUR, _CUDA | {"large_vram"}, mem_gb=80.0),
)}

VENUES_ENV = "RESEARCH_VENUES"


def host_total_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        return None


def host_available_gb() -> float | None:
    """Memory free RIGHT NOW, other apps included. On unified memory this is what MPS gets."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return None


def local_venues() -> list[Venue]:
    """What this machine can run, by asking torch rather than assuming. Sizes are measured."""
    import torch  # lazy: keeps `loop screen` fast on the no-GPU path
    total = host_total_gb()
    out = [replace(VENUES["local-cpu"], mem_gb=total and round(total * 0.8, 1))]
    if torch.backends.mps.is_available():
        out.append(replace(VENUES["local-mps"],
                           mem_gb=round(torch.mps.recommended_max_memory() / 1e9, 1)))
    if torch.cuda.is_available():
        out.append(replace(VENUES["local-cuda"],
                           mem_gb=round(torch.cuda.mem_get_info()[1] / 1e9, 1)))
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


# -- memory footprint ------------------------------------------------------------------

# Dense + per-layer-embedding params in billions and bf16 KV bytes per token, from
# scripts/roofline.py (which reads them off the HF configs). Copied, not imported: the loop
# must not depend on an instrument.
MODEL_PARAMS_B = {"E2B": 3.67, "E4B": 7.41}
KV_BYTES_PER_TOKEN = {"E2B": 18432, "E4B": 57344}
HEADROOM_GB = 1.5   # activations, 262k-vocab logits, allocator slack; a guess on the safe side


@dataclass(frozen=True)
class Footprint:
    """What a run of the real engine needs in memory, so a venue can refuse it before it OOMs."""

    model: str = "E2B"
    blocks: int = 512
    block_size: int = 16
    weight_bytes: int = 2

    @classmethod
    def from_env(cls) -> "Footprint":
        """Same knobs the engine reads, so the plan and the launch describe the same run."""
        name = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it").upper()
        model = "E4B" if "E4B" in name else "E2B"
        return cls(model=model,
                   blocks=int(os.environ.get("CUSTOM_BACKEND_BLOCKS", "512")),
                   block_size=int(os.environ.get("CUSTOM_BACKEND_BLOCK_SIZE", "16")))

    @property
    def weights_gb(self) -> float:
        return MODEL_PARAMS_B[self.model] * self.weight_bytes

    @property
    def pool_gb(self) -> float:
        return self.blocks * self.block_size * KV_BYTES_PER_TOKEN[self.model] / 1e9

    def needed_gb(self, venue: Venue) -> float:
        """Weights load on the host and are then moved; on unified memory both copies coexist."""
        load = 2.0 if venue.unified else 1.0
        return self.weights_gb * load + self.pool_gb + HEADROOM_GB

    def describe(self, venue: Venue) -> str:
        load = " x2 load" if venue.unified else ""
        return (f"{self.model} ~{self.needed_gb(venue):.1f} GB "
                f"(weights {self.weights_gb:.1f}{load} + pool {self.pool_gb:.2f} + "
                f"headroom {HEADROOM_GB})")


def fits(fp: Footprint, venue: Venue) -> bool:
    return venue.mem_gb is None or fp.needed_gb(venue) <= venue.mem_gb


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


def route(h: Hypothesis, venues: list[Venue] | None = None,
          footprint: Footprint | None = None) -> Placement:
    """Cheapest venue that satisfies the hypothesis; ties go to the least capable machine.

    Tiers 3-4 run the real engine, so with a footprint they also skip venues it cannot fit
    in. Tiers 1-2 are arithmetic and mocked repros and never load the model.
    """
    venues = venues if venues is not None else available_venues()
    need = requirements(h)
    capable = [v for v in venues if need <= v.capabilities]
    if capable and footprint is not None and h.falsification_tier >= 3:
        too_small = [v for v in capable if not fits(footprint, v)]
        capable = [v for v in capable if fits(footprint, v)]
        if not capable:
            raise Unroutable(
                f"{h.id}: {footprint.model} does not fit any capable venue: "
                + "; ".join(f"{v.name} has {v.mem_gb} GB, needs "
                            f"{footprint.needed_gb(v):.1f}" for v in too_small)
                + ". Use a smaller model/pool or enable a bigger venue.")
    if not capable:
        gaps = {v.name: sorted(need - v.capabilities) for v in venues}
        nearest = min(gaps, key=lambda n: len(gaps[n])) if gaps else None
        could = [n for n, v in VENUES.items() if need <= v.capabilities and not v.local]
        raise Unroutable(
            f"{h.id} needs {sorted(need)}; no available venue offers "
            f"{gaps[nearest] if nearest else sorted(need)}. Cloud venues are opt-in: "
            f"{VENUES_ENV}={','.join(could) or '<none in catalogue>'}")
    v = min(capable, key=lambda v: (v.usd_per_hour, len(v.capabilities)))
    cost = estimate_usd(h.falsification_tier, usd_per_hour=v.usd_per_hour)
    why = ("free" if cost == 0.0 else f"~${cost:.2f}") + f", needs {sorted(need) or 'nothing'}"
    return Placement(h.id, v, h.falsification_tier, cost, why)


def preflight(venue: Venue, fp: Footprint) -> tuple[bool, str]:
    """Will this run fit on the venue RIGHT NOW? Local venues share memory with whatever else
    is open; a plan that fit at screen time can still OOM at launch."""
    need = fp.needed_gb(venue)
    if venue.mem_gb is not None and need > venue.mem_gb:
        return False, f"{fp.describe(venue)} exceeds {venue.name}'s {venue.mem_gb} GB"
    if venue.local:
        free = host_available_gb()
        if free is not None and need > free:
            return False, (f"{fp.describe(venue)} but only {free:.1f} GB free on the host "
                           f"right now; close applications before launching")
    return True, f"{fp.describe(venue)} fits {venue.name}"


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
          minutes: float | None = None, venue: Venue | None = None,
          footprint: Footprint | None = None) -> Budget:
    """Raise unless this experiment is affordable AND fits. Call BEFORE launching anything."""
    if venue is not None and footprint is not None and tier >= 3:
        ok, why = preflight(venue, footprint)
        if not ok:
            raise RuntimeError(f"memory refused '{label}': {why}")
    b = budget or Budget.load()
    rate = venue.usd_per_hour if venue else GPU_USD_PER_HOUR
    ok, why = b.can_afford(tier, minutes, rate)
    if not ok:
        raise RuntimeError(f"budget refused '{label}': {why}")
    b.record(tier, label, minutes, venue=venue.name if venue else None, usd_per_hour=rate)
    b.save()
    return b
