"""One-shot: record the experiments already run, from the tuning log and HANDOFF.

These are marked `source="reconstructed"` and CANNOT authorise a merge — they are history, so
the loop can see what has already been tried and at what cost, without pretending prose is
evidence. Every future experiment goes through `loop judge` and is marked `source="loop"`.

Deltas are the measured numbers; gates are recorded as they were actually judged at the time,
including the ones that failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference_server.research.kb import save_experiment  # noqa: E402
from inference_server.research.schemas import Arm, Experiment  # noqa: E402


def _gates(validity=True, significance=True, correctness=True, cost=True, why=""):
    return {
        "validity": {"name": "validity", "passed": validity, "reason": why or "as judged", "evidence": {}},
        "significance": {"name": "significance", "passed": significance, "reason": why or "as measured", "evidence": {}},
        "correctness": {"name": "correctness", "passed": correctness, "reason": "suite green", "evidence": {}},
        "cost": {"name": "cost", "passed": cost, "reason": why or "declared", "evidence": {}},
    }


# (sha, branch, verdict, delta, gates, cost, note)
RECORD = [
    ("bfccfe3", "perf/bucketed-decode-graphs", "confirmed",
     {"decode_ms_by_bucket": {"before": 31.84, "after": 17.73, "pct": -44.3}},
     _gates(cost=False, why="1.80x ms/step at n=1; startup +~21min, declared not hidden"), 0.5),
    ("d875176", "perf/depython-decode-step", "confirmed",
     {"tok_s_within_slo": {"before": 598.2, "after": 732.3, "pct": 22.4}}, _gates(), 1.0),
    ("427b3d4", "fix/prefill-graph-ima", "confirmed",
     {"ttft_prefill_p50": {"before": 105.0, "after": 43.0, "pct": -59.0}}, _gates(), 0.4),
    ("990642c", "perf/split-k-decode-attention", "confirmed",
     {"decode_ms_by_bucket": {"before": 0.5635, "after": 0.0795, "pct": -85.9}},
     _gates(significance=True, why="7.09x at n=1 L=2048; end-to-end within noise on 60-tok prompts"), 0.4),
    ("67e5e54", "perf/lm-head-last-position", "confirmed",
     {"ttft_prefill_p50": {"before": 51.36, "after": 50.07, "pct": -2.5}},
     _gates(why="3-8%, smaller than predicted; kept for the 2GB logits saving"), 0.3),
    ("7171dfd", "perf/prefill-graph-general", "confirmed",
     {"ttft_prefill_p50": {"before": 282.3, "after": 228.0, "pct": -19.2}}, _gates(), 0.4),
    ("c30e22e", "perf/prefill-graph-prefix-hits", "confirmed",
     {"ttft_p50": {"before": 98.0, "after": 46.0, "pct": -53.1}}, _gates(), 0.6),
    ("b3a4201", "perf/prefill-graph-long-prompts", "confirmed",
     {"ttft_p95": {"before": 238.0, "after": 239.0, "pct": 0.4}},
     _gates(significance=False, why="neutral on this harness; correct and kept for cache-miss traffic"), 0.6),
    ("a525018", "perf/length-grouped-waves", "invalid",
     {"tok_s_within_slo": {"before": 529.1, "after": 565.0, "pct": 6.8}},
     _gates(validity=False, why="83-91% of prefill waves are K=1 — the harness cannot test wave composition"), 0.8),
    ("3945c4c", "perf/graph-compile-cost", "noise",
     {"graph_capture_s": {"before": 553.0, "after": 771.0, "pct": 39.4}},
     _gates(significance=False, why="high variance between runs; mechanism proven, total not"), 0.5),
    ("676222b", "perf/inductor-cache-volume", "confirmed",
     {"graph_capture_s": {"before": 705.5, "after": 574.5, "pct": -18.6}},
     _gates(why="-19%; the artifact cache is not the lever, bucket count is"), 0.6),
    ("bfec9a3", "feat/radix-prefix-cache", "confirmed",
     {"cache_hit_rate": {"before": 0.0, "after": 0.975, "pct": None}},
     _gates(why="shared system prompt: 0 tokens reused -> 4992"), 0.3),
    ("9f82dfe", "fix/kv-backpressure-and-cache-eviction", "confirmed",
     {"total_iteration_errors": {"before": 1, "after": 0, "pct": -100.0}},
     _gates(why="rate 48: 0 completions -> 454, pool_free 0 -> 637"), 0.9),
    ("d7843dd", "fix/radix-capacity-semantics", "confirmed",
     {"cache_max_blocks": {"before": 1024, "after": 4096, "pct": 300.0}}, _gates(), 0.0),
    ("89ada71", "obs/ttft-breakdown", "confirmed",
     {"ttft_prefill_p50": {"before": 44.0, "after": 35.0, "pct": -20.5}},
     _gates(why="instrument change: split TTFT into queue vs prefill; ruled out scheduling levers"), 0.5),
]

# Measured and rejected without a merge — the most valuable records in the ledger.
REJECTED = [
    ("feat/mixed-batch-prefill", "rejected",
     {"decode_ms_by_bucket": {"before": 20.6, "after": 75.6, "pct": 266.9}},
     _gates(significance=False, why="mixed step cannot replay the decode graph; 3.2-5.3x eager tax"), 0.4),
    ("perf/graph-capture-order", "rejected",
     {"graph_capture_s": {"before": 1288.0, "after": 1348.0, "pct": 4.7}},
     _gates(significance=False, why="ascending capture is no better; Dynamo re-specialises regardless"), 0.6),
]


def main() -> int:
    n = 0
    for sha, branch, verdict, delta, gates, cost in RECORD:
        e = Experiment(hypothesis_id=f"backfill:{branch}", engine_sha_base="dc7321e",
                       branch=branch, arms=[Arm("baseline", "dc7321e"), Arm("treatment", sha)],
                       verdict=verdict, gates=gates, delta=delta, cost_usd=cost,
                       source="reconstructed")
        e.id = f"exp-backfill-{branch.replace('/', '-')}"
        save_experiment(e)
        n += 1
    for branch, verdict, delta, gates, cost in REJECTED:
        e = Experiment(hypothesis_id=f"backfill:{branch}", engine_sha_base="dc7321e",
                       branch=branch, arms=[Arm("baseline", "dc7321e")],
                       verdict=verdict, gates=gates, delta=delta, cost_usd=cost,
                       source="reconstructed")
        e.id = f"exp-backfill-{branch.replace('/', '-')}"
        save_experiment(e)
        n += 1
    print(f"back-filled {n} experiments (source=reconstructed — history, not authorisation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
