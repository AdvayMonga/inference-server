"""The noise floor: what this harness measures when NOTHING changed.

LOOP.md's significance gate asks whether an effect "exceeds the variance budget". Until now
that budget was a sentence in a knowledge entry ("3.2x on ttft_prefill_p95, 6.5x on ttft_p95")
that a human had to remember to apply. The procedure in notes/03 is explicit about what it
should be instead:

    Run baseline against baseline, same config, same node, interleaved arms. Record the
    variance per metric per workload class. That variance is the band. Any delta inside it is
    filed INCONCLUSIVE, never a win. Store the band keyed on hardware and corpus version.

This module is the "store" and the "consult" halves of that. A band is measured by a null
experiment (`scripts/bench/replay_local.py --null`), reduced here, written to
`knowledge/noise/<key>.json` — the same shape `knowledge/timing/` already uses for the
simulator's fitted coefficients — and consulted by `compare.significance_replicated`.

The band is a PROPERTY OF A HARNESS ON ONE MACHINE FOR ONE WORKLOAD CLASS. A band measured on
cold_start says nothing about steady_interactive, and one measured on an M4 Pro says nothing
about an A100. `find_band` therefore matches on all of (harness, workload_class, model,
hardware) and refuses to fall back to a near-miss: an absent band means the gate behaves
exactly as it did before, which is the safe direction.

Stdlib only — the loop CI lane installs nothing else.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from inference_server.research.schemas import REPO_ROOT, Vitals

BAND_DIR = REPO_ROOT / "knowledge" / "noise"

# How many null standard deviations wide the band is. 2, not 1, and the first null experiment is
# why: with k=1 its own two arms produced a t-significant +2.7% on `tpot_p50`, whose null cv is
# 2.4% — the same config, on the same box, judged a win by every rule the loop had. A single run
# of either arm can land ~2 sd from the mean, so a difference that size is not distinguishable
# from having drawn two runs of ONE config; the measured min/max range backs that up (tpot_p50
# spans 1.08x, i.e. about +-4% around its mean, against a 2-sigma width of 4.8%).
#
# PROVISIONAL, and the weakest part of this module. The reasoning above is general, but the
# number is calibrated from n=1: one null experiment, one workload class (cold_start), one box
# (M4 Pro / E2B), one observed false positive on one metric. It is project-wide only because
# there is nothing else yet to key it on. Two ways it could be wrong that a second null would
# expose: a saturated regime may have a fatter-than-Gaussian tail, where 2 sd under-covers; and
# a metric whose null distribution is tight and well-behaved (`wall_s`, cv 1%) may deserve a
# smaller multiplier than one that is not (`ttft_queue_p50`, cv 45%), which a single scalar
# cannot express. Revisit when a second null lands on a different regime or hardware — and if
# they disagree, the fix is probably a per-metric multiplier stored in the band, not a different
# global constant. `inside(metric, pct, sigmas=...)` already takes an override, so a caller can
# disagree without editing this. See kb-20260917-aa6b0f4d.
BAND_SIGMAS = 2.0

# The panel fields a serving replay actually populates. Structural fields (wave_sizes,
# *_by_bucket) are not summarisable this way and counters that are 0 in every run carry no
# spread; `band_from_arms` drops whatever is absent or constant rather than inventing a band.
DEFAULT_METRICS = (
    "ttft_p50", "ttft_p95", "ttft_queue_p50", "ttft_queue_p95",
    "ttft_prefill_p50", "ttft_prefill_p95", "tpot_p50", "tpot_p95",
    "tok_s_within_slo", "wall_s", "active_mean", "active_high_water", "decode_steps",
    "cache_hit_rate", "pool_utilization",
)


@dataclass
class MetricBand:
    """Run-to-run spread of one panel metric with nothing changed."""

    n: int
    mean: float
    sd: float
    cv: float                # sd / |mean| — the relative width, the part that transfers
    min: float
    max: float
    min_max_ratio: float     # max / min, the number this repo has always quoted as "spread"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NoiseBand:
    """One null experiment, reduced. Keyed on what has to match for it to apply."""

    harness: str
    workload_class: str
    model: str
    hardware: str
    corpus_version: str | None = None
    engine_sha: str = ""
    run_group: str = ""
    run_ids: list[str] = field(default_factory=list)
    n_runs: int = 0                      # runs the band was computed over, warmup already dropped
    arms: list[str] = field(default_factory=list)
    metrics: dict[str, MetricBand] = field(default_factory=dict)
    entry_id: str = ""                   # the knowledge/ entry that tells the story
    notes: str = ""

    def key(self) -> str:
        return f"{self.harness}|{self.workload_class}|{self.model}|{self.hardware}"

    def identity(self) -> str:
        return (f"{self.harness}/{self.workload_class} on {self.hardware} with {self.model}"
                f"{f', corpus {self.corpus_version[:12]}' if self.corpus_version else ''}")

    def inside(self, metric: str, pct: float, sigmas: float = BAND_SIGMAS) -> bool | None:
        """Is a percentage change no bigger than what this metric does on its own?

        `None` means the band says nothing about `metric` — the caller must then behave exactly
        as it did before the band existed. The width is `sigmas` times the null coefficient of
        variation: relative, because a band measured at one absolute level (150 ms TTFT) is only
        usable at another if it is expressed as a fraction, and `sigmas`-wide because a single
        run of either arm can land that far from the mean (see BAND_SIGMAS).
        """
        w = self.width_pct(metric, sigmas)
        return None if w is None else abs(pct) <= w

    def width_pct(self, metric: str, sigmas: float = BAND_SIGMAS) -> float | None:
        """The band's half-width for `metric`, as a percentage of its null mean."""
        b = self.metrics.get(metric)
        return None if b is None else round(b.cv * 100.0 * sigmas, 2)

    # -- io ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = {k: v.to_dict() for k, v in self.metrics.items()}
        return d

    def to_json(self, path: str | Path | None = None) -> str:
        blob = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(blob)
        return blob

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NoiseBand":
        d = dict(d)
        d["metrics"] = {k: MetricBand(**v) for k, v in (d.get("metrics") or {}).items()}
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "NoiseBand":
        return cls.from_dict(json.loads(Path(path).read_text()))


def metric_band(values: Iterable[float]) -> MetricBand | None:
    """Reduce one metric's replicate values. `None` when there is nothing to say."""
    xs = [float(v) for v in values]
    if len(xs) < 2:
        return None
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs)
    lo, hi = min(xs), max(xs)
    return MetricBand(
        n=len(xs), mean=round(mean, 4), sd=round(sd, 4),
        cv=round(sd / abs(mean), 4) if mean else 0.0,
        min=round(lo, 4), max=round(hi, 4),
        min_max_ratio=round(hi / lo, 4) if lo else 0.0,
    )


def band_from_arms(
    arms: dict[str, list[Vitals]],
    *,
    drop_first_per_arm: bool = True,
    metrics: Iterable[str] = DEFAULT_METRICS,
    entry_id: str = "",
    notes: str = "",
) -> NoiseBand:
    """Reduce a NULL experiment — every arm the same config — to a band.

    `drop_first_per_arm` mirrors `significance_replicated`, which discards each arm's first run
    as warmup. The band has to describe the spread the gate actually faces, so it is measured
    over the same runs the gate would see; the raw with-warmup spread belongs in the write-up,
    not in the number that gates.
    """
    ordered = [p for arm in sorted(arms) for p in (arms[arm][1:] if drop_first_per_arm
                                                   else arms[arm])]
    if len(ordered) < 2:
        raise ValueError(f"a band needs >= 2 runs after warmup, got {len(ordered)}; each arm "
                         f"had {[len(v) for v in arms.values()]}")

    ref = ordered[0]
    hc = ref.validity.harness_config
    out: dict[str, MetricBand] = {}
    for m in metrics:
        vals = [getattr(p, m, None) for p in ordered]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in vals):
            continue                       # absent from some run: no band, rather than a wrong one
        b = metric_band(vals)
        if b is not None:
            out[m] = b
    return NoiseBand(
        harness=ref.validity.harness,
        workload_class=ref.validity.workload_class or str(hc.get("workload_class") or ""),
        model=str(hc.get("model") or ""),
        hardware=str((ref.validity.device_state or {}).get("gpu_name") or hc.get("hardware") or ""),
        corpus_version=ref.validity.corpus_version,
        engine_sha=ref.validity.engine_sha,
        run_group=ref.validity.run_group,
        run_ids=[p.validity.run_id for p in ordered],
        n_runs=len(ordered),
        arms=sorted(arms),
        metrics=out,
        entry_id=entry_id,
        notes=notes,
    )


# --------------------------------------------------------------------------- lookup

def load_bands(directory: Path = BAND_DIR) -> list[NoiseBand]:
    if not directory.exists():
        return []
    return [NoiseBand.load(p) for p in sorted(directory.glob("*.json"))]


def find_band(
    *,
    harness: str,
    workload_class: str,
    model: str,
    hardware: str,
    corpus_version: str | None = None,
    directory: Path = BAND_DIR,
) -> NoiseBand | None:
    """The band for exactly this situation, or None.

    No nearest-neighbour fallback, on purpose — same reasoning as `kb.covers`. A band from a
    different machine, model or workload class is not an approximation of this one; the
    cold_start band and the steady_interactive band on the same box differ by more than most
    effects anyone wants to claim.
    """
    for b in load_bands(directory):
        if (b.harness, b.workload_class, b.model, b.hardware) != \
                (harness, workload_class, model, hardware):
            continue
        if corpus_version and b.corpus_version and corpus_version != b.corpus_version:
            continue
        return b
    return None


def band_for(panel: Vitals, directory: Path = BAND_DIR) -> NoiseBand | None:
    """The band a panel falls under, read off its own validity block."""
    hc = panel.validity.harness_config
    return find_band(
        harness=panel.validity.harness,
        workload_class=panel.validity.workload_class or str(hc.get("workload_class") or ""),
        model=str(hc.get("model") or ""),
        hardware=str((panel.validity.device_state or {}).get("gpu_name") or hc.get("hardware") or ""),
        corpus_version=panel.validity.corpus_version,
        directory=directory,
    )


def format_band(b: NoiseBand) -> str:
    """The markdown table the knowledge entry carries, so the two never disagree."""
    lines = [f"| metric | mean | sd | cv | min | max | max/min | band (+-{BAND_SIGMAS:g}sd) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in b.metrics.items():
        lines.append(f"| {name} | {m.mean:g} | {m.sd:g} | {m.cv * 100:.0f}% | {m.min:g} | "
                     f"{m.max:g} | {m.min_max_ratio:.2f}x | {b.width_pct(name):g}% |")
    return "\n".join(lines)
