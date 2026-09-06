"""Record types for the research loop. Single source of truth; versioned.

Every field here earns its place by catching a specific failure this project actually hit —
the docstrings say which. Records are plain JSON on disk so they diff, review and grep.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# Bump when the panel's meaning changes. Panels of different versions are NOT comparable —
# compare.py enforces that rather than trusting anyone to remember.
PANEL_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[3]


class SchemaError(ValueError):
    """A record is missing something that makes it uninterpretable. Always fatal."""


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"


def git_sha(short: bool = True) -> str:
    """Engine SHA for the validity block.

    Instruments run inside an ephemeral container with no git repo, so the launching side
    passes it through RESEARCH_ENGINE_SHA. Without that a panel measured on Modal would carry
    `unknown` and could not be attributed to a commit at all — which defeats the point of
    recording provenance.
    """
    from_env = os.environ.get("RESEARCH_ENGINE_SHA")
    if from_env:
        return from_env
    cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    env = os.environ.get("RESEARCH_ENGINE_DIRTY")
    if env is not None:
        return env not in ("0", "false", "False", "")
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return True     # unknown state is treated as dirty: never silently attribute a result


@dataclass
class Validity:
    """Without this block a panel is a number with no provenance — refuse to use it.

    `harness_config` is the field that would have caught POOL_SIZE=64 silently turning the SLO
    sweep into a cache-HIT benchmark, and MAX_BATCH_SIZE 256-vs-32 reading as a 2.4x regression.
    `run_group` is what makes cross-session comparison impossible rather than merely discouraged.
    """

    engine_sha: str
    dirty: bool
    harness: str                      # which instrument produced this
    harness_config: dict[str, Any]    # every knob that could change the answer
    workload_regime: str              # cache_hit_heavy | cache_miss_heavy | mixed | synthetic
    n_samples: int
    run_group: str                    # arms of one experiment share this; set once per session
    run_id: str = field(default_factory=lambda: _new_id("run"))
    started_at: float = field(default_factory=time.time)
    stderr: float | None = None       # of the panel's primary metric, when n_samples > 1
    concurrency_observed: int | None = None
    notes: str = ""

    REGIMES = ("cache_hit_heavy", "cache_miss_heavy", "mixed", "synthetic")

    def validate(self) -> None:
        if self.workload_regime not in self.REGIMES:
            raise SchemaError(f"workload_regime must be one of {self.REGIMES}, got "
                              f"{self.workload_regime!r}")
        if not self.harness:
            raise SchemaError("harness is required — a panel must say what produced it")
        if not self.harness_config:
            raise SchemaError(
                "harness_config is required and must be non-empty. A panel without the knobs "
                "that produced it cannot be compared to anything (this is the POOL_SIZE=64 and "
                "MAX_BATCH_SIZE classes of error).")
        if self.n_samples < 1:
            raise SchemaError("n_samples must be >= 1")
        if not self.run_group:
            raise SchemaError("run_group is required — it is what prevents cross-session compare")


@dataclass
class Vitals:
    """The fixed panel from LOOP.md step 0. Measured identically every iteration.

    Subsetting it is not allowed; extending it is a PANEL_VERSION bump, because a panel that
    measures different things than the last one cannot be compared to it.
    """

    validity: Validity

    # --- service level -------------------------------------------------------------
    tok_s_within_slo: float | None = None
    slo_ttft_ms: float | None = None
    slo_tpot_ms: float | None = None
    ttft_p50: float | None = None
    ttft_p95: float | None = None
    # The split that rules out whole classes of fix at once: measured 21ms queue vs 203ms
    # prefill, which eliminated every scheduling lever for TTFT p95 in one shot.
    ttft_queue_p50: float | None = None
    ttft_queue_p95: float | None = None
    ttft_prefill_p50: float | None = None
    ttft_prefill_p95: float | None = None
    tpot_p50: float | None = None
    tpot_p95: float | None = None
    saturation_knee_rate: float | None = None

    # --- pressure & scheduling -----------------------------------------------------
    # wave_sizes alone killed two planned projects by showing 83-91% of waves are K=1.
    wave_sizes: dict[str, int] = field(default_factory=dict)
    active_size: int | None = None
    pending_depth: int | None = None
    pending_high_water: int | None = None
    kv_admit_blocked: int | None = None
    total_rejected: int | None = None
    total_expired: int | None = None
    total_preempted: int | None = None
    total_iteration_errors: int | None = None

    # --- cache ---------------------------------------------------------------------
    cache_hit_rate: float | None = None
    cache_lookups: int | None = None
    cache_entries: int | None = None
    cache_evictions: int | None = None
    cache_blocks_held: int | None = None
    cache_max_blocks: int | None = None
    pool_free_blocks: int | None = None
    pool_total_blocks: int | None = None
    pool_utilization: float | None = None

    # --- cost attribution ----------------------------------------------------------
    prefill_ms_by_bucket: dict[str, float] = field(default_factory=dict)
    decode_ms_by_bucket: dict[str, float] = field(default_factory=dict)
    graph_capture_s: float | None = None
    pct_of_memory_roof: float | None = None
    ridge_batch: int | None = None

    # --- resources -----------------------------------------------------------------
    peak_gpu_mem_gb: float | None = None
    peak_host_rss_gb: float | None = None
    wall_s: float | None = None
    gpu_cost_usd: float | None = None

    panel_version: int = PANEL_VERSION

    def validate(self) -> None:
        self.validity.validate()
        if self.panel_version != PANEL_VERSION:
            raise SchemaError(
                f"panel_version {self.panel_version} != current {PANEL_VERSION}; this record was "
                f"produced by a different definition of the panel and is not comparable")

    # -- io -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None) -> str:
        self.validate()
        blob = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(blob)
        return blob

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Vitals":
        d = dict(d)
        v = d.pop("validity")
        known = {f.name for f in fields(cls)} - {"validity"}
        unknown = set(d) - known
        if unknown:
            raise SchemaError(f"unknown panel fields {sorted(unknown)} — likely a newer "
                              f"PANEL_VERSION written by different code")
        return cls(validity=Validity(**v), **d)

    @classmethod
    def load(cls, path: str | Path) -> "Vitals":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class Hypothesis:
    """A prediction made BEFORE measuring, so being wrong is visible rather than rationalised.

    `kind` lets the loop improve itself: 'harness_change' and 'instrument_change' are how a
    finding like "83-91% of waves are K=1, so this harness cannot test wave planning" enters
    the loop as a first-class result instead of an accident.
    """

    statement: str
    gap_id: str                      # which attributed gap this attacks
    predicted_metric: str            # a field name on Vitals
    predicted_direction: str         # "increase" | "decrease"
    predicted_magnitude: str         # written down before the run, e.g. ">30%", "2x"
    falsification_tier: int          # 1 static, 2 CPU, 3 single-GPU probe, 4 full sweep
    falsification_test: str          # the cheapest thing that could prove this wrong
    kind: str = "engine_change"      # engine_change | instrument_change | harness_change
    # Tags are the reliable matching key into the knowledge base. Free-text relevance proved
    # too weak on the first real iteration: it missed that "chunk long prefills across ticks" is
    # ruled out by an entry about TTFT being prefill-bound, because they share almost no words.
    tags: list[str] = field(default_factory=list)
    kb_check: list[str] = field(default_factory=list)   # KnowledgeEntry ids actually consulted
    # Panel fields this change CANNOT affect. If one of them moves, the arms are contaminated
    # regardless of how good the headline number looks. Iteration 3 produced a CONFIRMED
    # "-48.2%" from two different machines while TPOT p50 differed 95.3 vs 46.4ms between arms
    # that only differed in a PREFILL flag — a sanity metric would have caught that instantly.
    sanity_metrics: list[str] = field(default_factory=list)
    status: str = "proposed"         # proposed | screened_out | testing | confirmed | rejected
    id: str = field(default_factory=lambda: _new_id("hyp"))
    created_at: float = field(default_factory=time.time)

    KINDS = ("engine_change", "instrument_change", "harness_change")
    DIRECTIONS = ("increase", "decrease")

    def validate(self) -> None:
        if self.kind not in self.KINDS:
            raise SchemaError(f"kind must be one of {self.KINDS}")
        if self.predicted_direction not in self.DIRECTIONS:
            raise SchemaError(f"predicted_direction must be one of {self.DIRECTIONS}")
        if not 1 <= self.falsification_tier <= 4:
            raise SchemaError("falsification_tier must be 1..4 (cheapest first)")
        if not self.predicted_magnitude:
            raise SchemaError(
                "predicted_magnitude is required BEFORE measuring — roughly half the hypotheses "
                "in this project were wrong, and recording the prediction is what makes that "
                "visible instead of quietly rationalised afterwards")
        if self.predicted_metric not in {f.name for f in fields(Vitals)}:
            raise SchemaError(f"predicted_metric {self.predicted_metric!r} is not a panel field")


@dataclass
class Arm:
    name: str                        # "baseline" | "treatment"
    sha: str
    run_ids: list[str] = field(default_factory=list)


@dataclass
class Experiment:
    """One test of one hypothesis. The record a merge is checked against."""

    hypothesis_id: str
    engine_sha_base: str
    arms: list[Arm] = field(default_factory=list)
    branch: str = ""
    verdict: str = "pending"         # pending | confirmed | rejected | noise | invalid
    gates: dict[str, Any] = field(default_factory=dict)   # name -> GateResult.to_dict()
    delta: dict[str, Any] = field(default_factory=dict)   # metric -> {before, after, pct}
    cost_usd: float = 0.0
    # "loop" = produced by the loop's own gates. "reconstructed" = written after the fact from
    # notes. Reconstructed records are history, NOT authorisation: premerge refuses them, or the
    # gate could be satisfied by prose about a run nobody can reproduce.
    source: str = "loop"
    notes: str = ""                  # why a record was retracted, or anything the gates can't say
    id: str = field(default_factory=lambda: _new_id("exp"))
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    VERDICTS = ("pending", "confirmed", "rejected", "noise", "invalid")

    def validate(self) -> None:
        if self.verdict not in self.VERDICTS:
            raise SchemaError(f"verdict must be one of {self.VERDICTS}")

    SOURCES = ("loop", "reconstructed")

    def all_gates_green(self) -> bool:
        required = {"validity", "sanity", "significance", "correctness", "cost"}
        if not required.issubset(self.gates):
            return False
        return all(self.gates[g].get("passed") for g in required)

    def authorises_merge(self) -> tuple[bool, str]:
        """Only a record the loop actually produced may gate a merge."""
        if self.source != "loop":
            return False, (f"experiment {self.id} is {self.source}, not produced by the loop — "
                           f"history, not authorisation")
        if not self.all_gates_green():
            failed = [n for n, g in self.gates.items() if not g.get("passed")]
            return False, f"gate(s) not green: {', '.join(failed) or 'missing gates'}"
        # Structural check, not a verdict check: single-run arms measure within-run request
        # scatter, not the run-to-run spread that actually decides significance. Refuse them by
        # shape, so a record cannot authorise a merge on noise however green its gates look.
        thin = [a.name for a in self.arms if len(a.run_ids) < 3]
        if thin:
            return False, (f"{self.id} has single-run arm(s) ({', '.join(thin)}); "
                           f"significance needs >=3 runs per arm")
        return True, f"{self.id} verdict={self.verdict}, all gates green"


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeEntry:
    """A durable finding. Negative results carry the same weight as wins — they are what stop
    the loop re-treading dead ends (capture ordering, mixed-batch prefill, length-grouped waves
    were each rejected on measurement and must never be retried blind)."""

    title: str
    summary: str
    status: str = "open"             # open | deferred | resolved | rejected | obsolete
    tags: list[str] = field(default_factory=list)
    evidence: list[dict[str, str]] = field(default_factory=list)   # {experiment_id|run_id|url}
    triggers: list[str] = field(default_factory=list)              # what would reopen this
    superseded_by: str | None = None
    id: str = field(default_factory=lambda: _new_id("kb"))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    STATUSES = ("open", "deferred", "resolved", "rejected", "obsolete")

    def validate(self) -> None:
        if self.status not in self.STATUSES:
            raise SchemaError(f"status must be one of {self.STATUSES}")
        if not self.title or not self.summary:
            raise SchemaError("title and summary are required")
