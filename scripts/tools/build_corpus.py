"""Generate the frozen workload corpus ONCE, deterministically, then commit the result.

    PYTHONPATH=src python scripts/tools/build_corpus.py [--out corpus] [--seed 20260915]

Re-running with the same seed reproduces every byte (the test asserts it). Changing the seed, the
class table, or prompt_bank.py produces different traces — that is a NEW corpus version, and every
panel measured against the old one stops being comparable. Do not hand-edit a trace.

Arrivals are Poisson; prompts come from scripts/bench/prompt_bank.py with a unique per-request
preamble so the PrefixCache miss path is exercised (what load_test.py --workload realistic does).
Some sessions get a second turn whose prompt extends the first, so turn_index > 0 is the one
place a prefix hit is legitimately available.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import prompt_bank  # noqa: E402
from inference_server.research.corpus import (  # noqa: E402
    SPLITS,
    Manifest,
    TraceRequest,
    WorkloadClass,
    build_manifest,
    write_trace,
)

DEFAULT_SEED = 20260915

NOTES = ("SLO values are placeholders pending the plan's Phase 0 decision: steady_interactive "
         "carries bench_serving's current p95 TTFT 200ms / TPOT 50ms, cold_start 2000ms TTFT, "
         "long_context 1000ms TTFT. Built by scripts/tools/build_corpus.py with seed "
         f"{DEFAULT_SEED}; a rebuild with any other input is a new corpus version.")


@dataclass
class ClassSpec:
    cls: WorkloadClass
    n_sessions: int
    buckets: list[tuple[str, list[str], float]]   # prompt_bank mix: (name, prompts, weight)
    followup_p: float                              # share of sessions that get a second turn


def _specs() -> list[ClassSpec]:
    mix = prompt_bank.PROMPT_MIX
    short_medium = [b for b in mix if b[0] in ("short", "medium")]
    long_only = [b for b in mix if b[0] == "long"]
    return [
        ClassSpec(WorkloadClass(
            name="cold_start",
            description="The first requests to a fresh replica: few, sparse, mixed short/medium.",
            slo_ttft_ms=2000.0, slo_tpot_ms=None, arrival_rate_rps=0.5,
            seen="cold_start/seen.jsonl", heldout="cold_start/heldout.jsonl"),
            n_sessions=8, buckets=short_medium, followup_p=0.0),
        ClassSpec(WorkloadClass(
            name="steady_interactive",
            description="Warm replica, mixed lengths at a rate that fills the batch; some "
                        "sessions return for a second turn.",
            slo_ttft_ms=200.0, slo_tpot_ms=50.0, arrival_rate_rps=4.0,
            seen="steady_interactive/seen.jsonl", heldout="steady_interactive/heldout.jsonl"),
            n_sessions=100, buckets=mix, followup_p=0.3),
        ClassSpec(WorkloadClass(
            name="long_context",
            description="Long prompts only, high KV pressure; where eviction policy matters.",
            slo_ttft_ms=1000.0, slo_tpot_ms=None, arrival_rate_rps=1.0,
            seen="long_context/seen.jsonl", heldout="long_context/heldout.jsonl"),
            n_sessions=40, buckets=long_only, followup_p=0.0),
    ]


def _draw_prompt(rng: random.Random, buckets) -> tuple[str, int]:
    """Weighted bucket draw with a unique preamble, so no two prompts share a first block."""
    total = sum(w for _, _, w in buckets)
    r, cum = rng.random() * total, 0.0
    name, prompts = buckets[-1][0], buckets[-1][1]
    for bname, bprompts, w in buckets:
        cum += w
        if r <= cum:
            name, prompts = bname, bprompts
            break
    lo, hi = prompt_bank.MAX_TOKENS_RANGE[name]
    return f"Case {rng.getrandbits(48):012x}. {rng.choice(prompts)}", rng.randint(lo, hi)


def build_trace(spec: ClassSpec, seed: int) -> list[TraceRequest]:
    rng = random.Random(seed)
    reqs: list[TraceRequest] = []
    t = 0.0
    for i in range(spec.n_sessions):
        if i:
            t += rng.expovariate(spec.cls.arrival_rate_rps)
        sid = f"{spec.cls.name}-{seed}-{i:04d}"
        prompt, max_tokens = _draw_prompt(rng, spec.buckets)
        reqs.append(TraceRequest(round(t, 3), sid, 0, prompt, max_tokens))
        if rng.random() < spec.followup_p:
            follow = rng.choice(prompt_bank.SHORT_PROMPTS)
            reqs.append(TraceRequest(round(t + rng.uniform(2.0, 10.0), 3), sid, 1,
                                     f"{prompt}\n\nFollow-up: {follow}", max_tokens))
    reqs.sort(key=lambda r: (r.arrival_s, r.session_id, r.turn_index))
    return reqs


def build_corpus(out: Path, seed: int = DEFAULT_SEED) -> Manifest:
    """Write every (class, split) trace and the manifest. Split k of class c uses its own seed."""
    specs = _specs()
    for ci, spec in enumerate(specs):
        for si, split in enumerate(SPLITS):
            write_trace(out / spec.cls.trace_file(split),
                        build_trace(spec, seed + 100 * ci + si))
    m = build_manifest({s.cls.name: s.cls for s in specs}, out, notes=NOTES)
    (out / "manifest.json").write_text(json.dumps(m.to_dict(), indent=2) + "\n")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "corpus"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    m = build_corpus(Path(args.out), args.seed)
    print(f"corpus_version {m.corpus_version}")
    for name, c in m.classes.items():
        print(f"  {name:20s} ttft<{c.slo_ttft_ms:.0f}ms rate={c.arrival_rate_rps} rps")
