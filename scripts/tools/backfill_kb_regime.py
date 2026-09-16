"""One-shot: backfill `regime`, `validity_range` and `mechanism` on knowledge/*.json.

The fields landed in PR #19 with every existing entry unassigned, so `loop kb --regime` and
`kb.covers()` had nothing to return. MAPPING below is the deliverable: one row per entry, with
the reason next to it, so a reviewer reads the table rather than the 69-file diff.

Rules the table follows (do not invent):
- `regime` is the workload class the finding was MEASURED under. Loop/harness/gate/process,
  instrumentation and pure code-structure entries have no workload regime and stay None.
- `validity_range` holds only what the entry or its cited harness states. A10G runs were all
  E2B and the September Modal harnesses default to A100-80GB/E4B (benchmarks/README.md,
  benchmarks/tuning_log.md, scripts/bench/bench_*_modal.py). Absent key = unconstrained.
- `mechanism` only where the entry already states the cause.

Idempotent: an entry already carrying its row is not rewritten, so a second run changes no
bytes. Writing goes through `kb.save_entry`, which bumps `updated_at` like every other write.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference_server.research.kb import KNOWLEDGE_DIR, load_entries, save_entry, write_index  # noqa: E402

SI, LC, CS = "steady_interactive", "long_context", "cold_start"
A100_E4B = {"model": "gemma-4-e4b", "hardware": "A100-80GB"}
A10G_E2B = {"model": "gemma-4-e2b", "hardware": "A10G"}


def _row(regime: str | None = None, validity: dict | None = None,
         mechanism: str | None = None) -> dict[str, Any]:
    return {"regime": regime, "validity_range": dict(validity or {}), "mechanism": mechanism}


MAPPING: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------ steady_interactive
    # why: A10G/E2B decode benchmarks under concurrent load (benchmarks/short_*.csv lineage).
    "kb-20260611-029": _row(SI, {**A10G_E2B, "concurrency": [1, 32]},
                            "decode_step_batched ran one forward per row, so B rows cost B "
                            "sequential forwards and nothing was GPU-batched."),
    "kb-20260611-030": _row(SI, {**A10G_E2B, "concurrency": [1, 32]},
                            "Reading K/V straight from paged blocks via the block table removes "
                            "the per-step dense gather/pad whose Python per-row loop caused the "
                            "N=32 knee."),
    "kb-20260612-031": _row(SI, {**A10G_E2B, "concurrency": [8, 32]},
                            "Decode was ~95% CPU dispatch-bound (~1000 launches/step, flat across "
                            "batch), so one captured graph removes the launch cost and padding to "
                            "max batch costs ~2 ms."),
    "kb-20260612-032": _row(SI, {**A10G_E2B, "concurrency": [1, 32]},
                            "Per-row Python tensor writes and block-table rebuilds cost ~N x layers "
                            "tiny launches per step; one vectorised indexed write removes them."),
    # why: A10G decode-step microbenchmark (eager vs graph vs compile), 4-32 real rows.
    "kb-20260530-014": _row(SI, A10G_E2B,
                            "Under the CUDA graph the decode floor is the matmuls reading all "
                            "weights from HBM each step (memory-bandwidth-bound), so op fusion has "
                            "nothing left to remove."),
    # why: full-batch decode sweep on A10G/E2B plus accuracy on the same model.
    "kb-20260613-015": _row(SI, A10G_E2B,
                            "Decode is weight-bandwidth-bound so int8 halves bytes per step, but "
                            "the CUDA win needs the dequant fused into the GEMV and A10G lacks the "
                            "headroom for Inductor to fuse it."),
    # why: dev-machine load test; the only number is MPS/E2B.
    "kb-20260514-025": _row(SI, {"model": "gemma-4-e2b", "hardware": "MPS"}),
    # why: A100/E4B mixed-step premise probe, rows 1/8/32.
    "kb-20260901-009": _row(SI, {**A100_E4B, "concurrency": [1, 32]},
                            "A mixed step has a variable (B, S) shape so it cannot replay the "
                            "decode CUDA graph, and the eager forward is dispatch-bound at 3-5x "
                            "the graphed step."),
    # why: bucketed decode graphs / pad-width numerics on A100/E4B (diag_pad_numerics_modal).
    "kb-20260901-011": _row(SI, A100_E4B,
                            "A [B,1,H]@[H,H] GEMM picks a different cuBLAS kernel per padded batch "
                            "width, so accumulation order and near-tie argmaxes change with the "
                            "number of rows in flight."),
    # why: batched-prefill numerics probes pin A100-80GB but default to E2B, so no model bound.
    "kb-20260902-007": _row(SI, {"hardware": "A100-80GB"},
                            "Batching changes GEMM shapes and rounding at layer 0; RMSNorm, GeGLU "
                            "and softmax amplify it over 15 layers into isolated positions, with "
                            "no cross-row dependence."),
    # why: open-loop serving harness on A100/E4B, rates as stated in each entry.
    "kb-20260902-006": _row(SI, {**A100_E4B, "arrival_rate_rps": [1, 6]},
                            "MAX_BATCH_SIZE=256 admits every arrival immediately, so 83-91% of "
                            "prefill waves are K=1 and there is no padding for grouping to remove."),
    "kb-20260903-001": _row(SI, {**A100_E4B, "arrival_rate_rps": [1, 3]},
                            "Queue wait is 21 ms of a 213 ms TTFT p95, so the tail is prefill and "
                            "scheduling cannot move it; the prefill itself is 4.4x off its "
                            "arithmetic floor, i.e. overhead, not irreducible attention."),
    "kb-20260906-6000f7f5": _row(SI, {**A100_E4B, "arrival_rate_rps": [8, 128]}),
    "kb-20260906-7efc7fcd": _row(SI, {**A100_E4B, "arrival_rate_rps": [16, 128]},
                                 "The 30 s admission deadline is 150x the TTFT budget, so under "
                                 "overload every request waits the full deadline before it is "
                                 "shed."),
    "kb-20260906-f13d8e3d": _row(SI, {**A100_E4B, "arrival_rate_rps": [1, 8]},
                                 "More concurrent rows amortise the same per-step decode cost, so "
                                 "TPOT is flat-to-improving with rate and the cost is per-step "
                                 "latency at low batch, not scheduling."),
    # why: the SLO under test is steady_interactive's; measured on the A100/E4B sweeps.
    "kb-20260906-9454d8a1": _row(SI, A100_E4B,
                                 "50 ms/token is roughly the best a fast A100 draw delivers for "
                                 "E4B at rate 1, so the budget has zero headroom and "
                                 "tok_s_within_slo is None on every run."),
    # why: prefill attribution on A100/E4B at the serving workload's p50-p95 prompt lengths.
    "kb-20260905-dcb78725": _row(SI, {**A100_E4B, "prompt_tokens": [240, 824]},
                                 "Attention is 1.5% of prefill FLOPs but 63.5% of prefill time, so "
                                 "the gap is an inefficient kernel (reclaimable), not irreducible "
                                 "O(S^2) work."),
    "kb-20260905-b9bc66c6": _row(SI, {**A100_E4B, "prompt_tokens": [240, 824]},
                                 "The prefill kernel launches one program per (sequence, head, "
                                 "query) and re-reads K/V blocks from HBM once per query: O(S^2) "
                                 "memory traffic at 0.34% of peak."),
    "kb-20260905-b170a1ac": _row(SI, A100_E4B,
                                 "Sliding layers cap attention at the 512-key window so their cost "
                                 "is bounded; full-attention layers (head_dim 512) attend the whole "
                                 "prompt and dominate, and tiling at D=512 spills registers."),
    "kb-20260905-481ec50a": _row(SI, {**A100_E4B, "prompt_tokens": [240, 824]},
                                 "Tiling fires only on sliding layers (D<=256, S>=512), which are "
                                 "window-bounded, while the D=512 full-attention layers where the "
                                 "work is lose 0.53x to register spill; the A/B also ran with the "
                                 "prefill graph off, where dispatch dominates."),
    # ------------------------------------------------------------ long_context
    # why: long (1000-token) workload to N=32 on Modal A10G/E2B; KV occupancy is the subject.
    "kb-20260612-035": _row(LC, {**A10G_E2B, "concurrency": [1, 32], "context_tokens": 1000},
                            "Full-attention pools bind (they grow with sequence, sliding pools cap "
                            "at the window), so right-sizing pools and gating per pool turns the "
                            "windowed memory saving into admission headroom."),
    # why: sliding-layer block release, validated on the long-workload Modal sweep.
    "kb-20260612-036": _row(LC, A10G_E2B,
                            "Sliding layers (28/35) only attend the last 512 tokens, so blocks "
                            "fully out of window can be released without touching the kernel."),
    # why: correctness of prompts past the 512 window; measured on E2B (CPU parity + Modal).
    "kb-20260612-037": _row(LC, {"model": "gemma-4-e2b"},
                            "The forward stored sliding_window but never applied it, and the "
                            "5-token parity fixture could not see a 512-token window."),
    # why: KV pool exhaustion under cache-missing overload (bench_stress_modal, A100/E4B).
    "kb-20260902-003": _row(LC, A100_E4B,
                            "Distinct-prompt traffic past the knee exhausts the pool, and every "
                            "path that allocated without releasing on failure leaked blocks "
                            "permanently."),
    # ------------------------------------------------------------ cold_start
    # why: startup cost only (graph capture + torch.compile time), A100/E4B bucketed graphs.
    "kb-20260901-010": _row(CS, A100_E4B,
                            "Every decode-graph bucket costs a fresh Inductor compile; without "
                            "compile the whole ladder captures in 5.6 s."),
    "kb-20260902-008": _row(CS, A100_E4B,
                            "Automatic-dynamic compile only generalises over large batch dims; "
                            "Dynamo re-specialises on each small bucket regardless of capture "
                            "order, so the number of small buckets sets the cost."),
    "kb-20260903-000": _row(CS, A100_E4B,
                            "Dynamo re-specialises on small batch dims whichever order they are "
                            "seen in; the levers are the number of distinct compiles and, weakly, "
                            "the persistent Inductor cache."),
    "kb-20260903-002": _row(CS, A100_E4B,
                            "Decode graphs go through torch.compile (~140 s per bucket, mostly "
                            "Dynamo tracing) while prefill graphs are plain captures, so the decode "
                            "ladder must be coarse and the prefill ladder can be fine."),
    # why: unmeasured precursor of the Inductor-cache work; the subject is fresh-container cost.
    "kb-20260530-013": _row(CS),
    # ------------------------------------------------------------ no workload regime
    # why: harness variance calibrations — no regime, but the hardware bound is explicit and is
    # exactly what the entry says to re-measure on change.
    "kb-20260905-1b1a2520": _row(None, A100_E4B),
    "kb-20260906-a81f6d18": _row(None, A100_E4B),
    # why: kernel/model constraints with a stated cause but no workload measurement.
    "kb-20260530-017": _row(None, None,
                            "FlashAttention-2's Ampere kernel caps head_dim at 256 and Gemma 4's "
                            "full-attention layers use 512."),
    "kb-20260612-033": _row(None, None,
                            "Triton recompiles per distinct constexpr value, so a length-derived "
                            "loop bound recompiled (~1 s, under the backend lock) for nearly every "
                            "block count."),
    "kb-20260905-33842618": _row(None, None,
                                 "The prefill kernel early-exits padded query positions, so bucket "
                                 "padding costs launch slots rather than compute."),
    # why: loop method, harness validity, gates, process — no workload regime by definition.
    **{i: _row() for i in (
        "kb-20260905-1f0727be", "kb-20260905-38efb550", "kb-20260905-521365f0",
        "kb-20260905-a474d802", "kb-20260905-ae5fd6c2", "kb-20260905-c1c2a5d5",
        "kb-20260905-c6904d17", "kb-20260905-d2a675d5", "kb-20260902-004",
        "kb-20260902-005", "kb-20260901-012", "kb-20260518-042", "kb-20260529-041",
        "kb-20260613-016",
    )},
    # why: instrumentation / metrics gaps, not findings about a workload.
    **{i: _row() for i in (
        "kb-20260906-22d4a185", "kb-20260915-2c4513a1", "kb-20260514-028", "kb-20260514-021",
    )},
    # why: code-structure decisions, latent bugs and deferred designs with no measurement.
    **{i: _row() for i in (
        "kb-20260514-018", "kb-20260514-019", "kb-20260514-020", "kb-20260514-022",
        "kb-20260514-023", "kb-20260514-024", "kb-20260514-026", "kb-20260514-027",
        "kb-20260514-043", "kb-20260514-044", "kb-20260514-045", "kb-20260514-046",
        "kb-20260514-047", "kb-20260611-038", "kb-20260611-039", "kb-20260611-040",
        "kb-20260612-034",
    )},
}


def apply(directory: Path = KNOWLEDGE_DIR) -> dict[str, Any]:
    """Write MAPPING onto the entries in `directory`; only entries that differ are rewritten."""
    entries = {e.id: e for e in load_entries(directory)}
    missing = sorted(set(MAPPING) - set(entries))
    if missing:
        raise SystemExit(f"MAPPING names entries not on disk (typo?): {missing}")
    written: list[str] = []
    for eid, row in MAPPING.items():
        e = entries[eid]
        if (e.regime, e.validity_range, e.mechanism) == (
                row["regime"], row["validity_range"], row["mechanism"]):
            continue
        e.regime, e.validity_range, e.mechanism = (
            row["regime"], dict(row["validity_range"]), row["mechanism"])
        save_entry(e, directory)
        written.append(eid)
    return {
        "written": written,
        "by_regime": Counter(r["regime"] for r in MAPPING.values() if r["regime"]),
        "unassigned": sorted(i for i, r in MAPPING.items() if r["regime"] is None),
        "not_in_table": sorted(set(entries) - set(MAPPING)),
    }


def main() -> int:
    s = apply()
    print(f"rewrote {len(s['written'])} of {len(MAPPING)} mapped entries")
    for regime, n in sorted(s["by_regime"].items()):
        print(f"  {regime}: {n}")
    print(f"  unassigned (deliberate, {len(s['unassigned'])}): {', '.join(s['unassigned'])}")
    if s["not_in_table"]:
        print(f"  NOT IN TABLE ({len(s['not_in_table'])}), left untouched: "
              f"{', '.join(s['not_in_table'])}")
    write_index()
    print("regenerated DECISIONS.md from knowledge/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
