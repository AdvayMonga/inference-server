"""Step 1: turn a panel into a ranked list of gaps. Deterministic — no judgement, no ideas.

Attribution runs before hypotheses so the loop argues from where the time actually goes rather
than from what is interesting. Each gap carries the evidence that produced it, so a hypothesis
can cite it and the validity gate can check the harness reached it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from inference_server.research.schemas import Vitals


@dataclass
class Gap:
    id: str
    title: str
    magnitude: str          # human-readable size of the prize
    evidence: dict[str, Any]
    rank_key: float         # bigger = more important; ordering only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attribute(panel: Vitals) -> list[Gap]:
    """Rank the gaps this panel exposes. Returns [] when the panel cannot support a claim."""
    gaps: list[Gap] = []
    v = panel

    # --- is the SLO met at all, and which half of TTFT is to blame -----------------
    if v.ttft_p95 is not None and v.slo_ttft_ms:
        over = v.ttft_p95 - v.slo_ttft_ms
        if over > 0:
            q, p = v.ttft_queue_p95, v.ttft_prefill_p95
            if q is not None and p is not None:
                # The split that rules out whole classes of fix at once.
                dominant = "prefill" if p > q else "queue"
                share = (max(p, q) / (p + q) * 100) if (p + q) else 0
                gaps.append(Gap(
                    id=f"ttft-{dominant}",
                    title=f"TTFT p95 over SLO by {over:.0f}ms, dominated by {dominant} "
                          f"({share:.0f}% of the split)",
                    magnitude=f"{over:.0f}ms over a {v.slo_ttft_ms:.0f}ms budget",
                    evidence={"ttft_p95": v.ttft_p95, "queue_p95": q, "prefill_p95": p},
                    rank_key=over,
                ))
            else:
                gaps.append(Gap(
                    id="ttft-unattributed",
                    title=f"TTFT p95 over SLO by {over:.0f}ms, but the panel has no "
                          f"queue/prefill split — attribute before optimising",
                    magnitude=f"{over:.0f}ms",
                    evidence={"ttft_p95": v.ttft_p95},
                    rank_key=over,
                ))

    if v.tpot_p95 is not None and v.slo_tpot_ms and v.tpot_p95 > v.slo_tpot_ms:
        over = v.tpot_p95 - v.slo_tpot_ms
        gaps.append(Gap("tpot", f"TPOT p95 over SLO by {over:.1f}ms",
                        f"{over:.1f}ms over {v.slo_tpot_ms:.0f}ms",
                        {"tpot_p95": v.tpot_p95}, over))

    # --- how far from the analytical ceiling ---------------------------------------
    if v.pct_of_memory_roof is not None and v.pct_of_memory_roof < 0.6:
        head = (1 - v.pct_of_memory_roof) * 100
        gaps.append(Gap("roofline-headroom",
                        f"decode at {v.pct_of_memory_roof * 100:.0f}% of the memory roof",
                        f"{head:.0f}% headroom",
                        {"pct_of_memory_roof": v.pct_of_memory_roof,
                         "ridge_batch": v.ridge_batch}, head))

    # --- pressure signals ----------------------------------------------------------
    if v.total_iteration_errors:
        gaps.append(Gap("stability", f"{v.total_iteration_errors} scheduler iteration errors",
                        "correctness before performance",
                        {"total_iteration_errors": v.total_iteration_errors}, 1e6))
    if v.total_expired:
        gaps.append(Gap("shedding", f"{v.total_expired} requests shed at the admission deadline",
                        f"{v.total_expired} requests",
                        {"total_expired": v.total_expired,
                         "pending_high_water": v.pending_high_water}, float(v.total_expired)))
    if v.kv_admit_blocked:
        gaps.append(Gap("kv-capacity", f"admission blocked {v.kv_admit_blocked}x on KV",
                        f"{v.kv_admit_blocked} holds",
                        {"kv_admit_blocked": v.kv_admit_blocked,
                         "pool_utilization": v.pool_utilization}, float(v.kv_admit_blocked)))

    # --- cache effectiveness --------------------------------------------------------
    if v.cache_hit_rate is not None and v.cache_lookups and v.cache_hit_rate < 0.2:
        thrash = (v.cache_evictions or 0) > (v.cache_lookups or 0)
        gaps.append(Gap(
            "cache-hit-rate",
            f"prefix cache hit rate {v.cache_hit_rate:.1%}"
            + (" while evicting more often than it looks up (thrashing)" if thrash else ""),
            f"{v.cache_hit_rate:.1%} hit rate",
            {"hit_rate": v.cache_hit_rate, "evictions": v.cache_evictions,
             "lookups": v.cache_lookups, "blocks_held": v.cache_blocks_held,
             "max_blocks": v.cache_max_blocks}, 100 * (0.2 - v.cache_hit_rate)))

    # --- is the harness even capable of testing scheduling? -------------------------
    if v.wave_sizes:
        total = sum(v.wave_sizes.values())
        k1 = v.wave_sizes.get("1", 0)
        if total and k1 / total > 0.8:
            gaps.append(Gap(
                "harness-k1",
                f"{k1 / total:.0%} of prefill waves are K=1 — this run cannot test anything "
                f"about wave composition or prefill padding",
                "harness limitation, not an engine gap",
                {"wave_sizes": v.wave_sizes}, 0.5))

    return sorted(gaps, key=lambda g: -g.rank_key)
