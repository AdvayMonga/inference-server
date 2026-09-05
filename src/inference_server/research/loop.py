"""CLI driver for the research loop. Every step of LOOP.md, deterministic where it can be.

    python -m inference_server.research.loop attribute runs/<id>.json
    python -m inference_server.research.loop screen <hypotheses.json>
    python -m inference_server.research.loop judge --hyp H.json --baseline A.json --treatment B.json
    python -m inference_server.research.loop index
    python -m inference_server.research.loop kb --status rejected

Step 2 (hypothesize) is deliberately NOT automated — it is the one place judgement belongs, and
it runs against research/HYPOTHESIZE.md with the knowledge base as input. Everything either side
of it is code, so the same inputs always give the same steps. "The agent decides; the code
measures."
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from inference_server.research.attribute import attribute
from inference_server.research.gates import judge
from inference_server.research.kb import (
    related,
    load_entries,
    query,
    save_experiment,
    write_index,
)
from inference_server.research.schemas import Arm, Experiment, Hypothesis, Vitals

TIER_NAMES = {1: "static/arithmetic (seconds)", 2: "CPU repro (seconds)",
              3: "single-GPU probe (5-10 min)", 4: "full sweep (15-30 min, real $)"}


def cmd_attribute(args) -> int:
    panel = Vitals.load(args.panel)
    gaps = attribute(panel)
    if not gaps:
        print("no gaps found — the panel is within SLO and shows no pressure signals")
        return 0
    print(f"Gaps from {panel.validity.harness} "
          f"(regime={panel.validity.workload_regime}, sha={panel.validity.engine_sha}):\n")
    for i, g in enumerate(gaps, 1):
        print(f"  {i}. [{g.id}] {g.title}")
        print(f"     prize: {g.magnitude}")
    if args.out:
        Path(args.out).write_text(json.dumps([g.to_dict() for g in gaps], indent=2))
        print(f"\nwrote {args.out}")
    return 0


def cmd_screen(args) -> int:
    """Order hypotheses cheapest-falsification-first and flag known dead ends."""
    raw = json.loads(Path(args.hypotheses).read_text())
    hyps = [Hypothesis(**h) for h in (raw if isinstance(raw, list) else [raw])]
    entries = load_entries()

    for h in hyps:
        h.validate()

    for h in sorted(hyps, key=lambda x: x.falsification_tier):
        hits = related(h.statement, entries, tags=h.tags)
        blocking = [e for _, e in hits if e.status in ("rejected", "resolved")]
        flag = "  <-- READ THE RELATED ENTRIES FIRST" if blocking else ""
        print(f"tier {h.falsification_tier} ({TIER_NAMES[h.falsification_tier]}){flag}")
        print(f"  {h.id}  {h.statement}")
        print(f"  predicts {h.predicted_metric} {h.predicted_direction} by "
              f"{h.predicted_magnitude}")
        print(f"  falsify with: {h.falsification_test}")
        for score, e in hits:
            print(f"    ~{score:5.1f} [{e.status}] {e.id}: {e.title[:66]}")
        if not hits:
            print("    (no related knowledge — genuinely new ground, or the tags are missing)")
        if not h.tags:
            print("    ! no tags on this hypothesis: knowledge lookup fell back to free text, "
                  "which is unreliable. Add tags.")
        uncited = [e.id for _, e in hits
                   if e.status in ("rejected", "resolved") and e.id not in h.kb_check]
        if uncited:
            print(f"    ! settled entries not cited in kb_check: {', '.join(uncited[:3])}")
        print()
    print("Run the cheapest tier first; never enter tier N+1 while tier N could still falsify.")
    return 0


def cmd_judge(args) -> int:
    hyp = Hypothesis(**json.loads(Path(args.hyp).read_text()))
    hyp.validate()
    baseline, treatment = Vitals.load(args.baseline), Vitals.load(args.treatment)

    j = judge(hyp, baseline, treatment, run_tests=not args.no_tests)
    print(f"hypothesis: {hyp.statement}")
    print(f"predicted:  {hyp.predicted_metric} {hyp.predicted_direction} "
          f"by {hyp.predicted_magnitude}\n")
    for name in ("validity", "significance", "correctness", "cost"):
        g = j.gates[name]
        print(f"  [{'PASS' if g.passed else 'FAIL'}] {name:13} {g.reason}")
    print(f"\nverdict: {j.verdict.upper()}")

    exp = Experiment(
        hypothesis_id=hyp.id,
        engine_sha_base=baseline.validity.engine_sha,
        branch=args.branch or "",
        arms=[Arm("baseline", baseline.validity.engine_sha, [baseline.validity.run_id]),
              Arm("treatment", treatment.validity.engine_sha, [treatment.validity.run_id])],
        verdict=j.verdict,
        gates=j.to_dict(),
        delta={hyp.predicted_metric:
               j.gates["significance"].evidence or {}},
    )
    path = save_experiment(exp)
    print(f"recorded {exp.id} -> {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    if j.verdict != "confirmed":
        print("\nThis is a RESULT, not a failure. Write it to knowledge/ so the loop does not "
              "re-propose it.")
    return 0


def cmd_index(args) -> int:
    p = write_index()
    print(f"regenerated {p} from knowledge/ ({len(load_entries())} entries)")
    return 0


def cmd_kb(args) -> int:
    hits = query(status=args.status, tags=args.tags or (), text=args.text)
    for e in hits:
        print(f"{e.status:9} {e.id}  {e.title}")
        if args.verbose:
            print(f"           tags: {', '.join(e.tags)}")
    print(f"\n{len(hits)} entr{'y' if len(hits) == 1 else 'ies'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research.loop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("attribute", help="panel -> ranked gaps (step 1)")
    a.add_argument("panel"); a.add_argument("--out")
    a.set_defaults(fn=cmd_attribute)

    s = sub.add_parser("screen", help="order hypotheses cheapest-first, flag dead ends (step 3)")
    s.add_argument("hypotheses"); s.set_defaults(fn=cmd_screen)

    j = sub.add_parser("judge", help="run the four gates and record an experiment (steps 5-6)")
    j.add_argument("--hyp", required=True)
    j.add_argument("--baseline", required=True)
    j.add_argument("--treatment", required=True)
    j.add_argument("--branch"); j.add_argument("--no-tests", action="store_true")
    j.set_defaults(fn=cmd_judge)

    i = sub.add_parser("index", help="regenerate DECISIONS.md from knowledge/")
    i.set_defaults(fn=cmd_index)

    k = sub.add_parser("kb", help="query the knowledge base")
    k.add_argument("--status"); k.add_argument("--tags", nargs="*")
    k.add_argument("--text"); k.add_argument("-v", "--verbose", action="store_true")
    k.set_defaults(fn=cmd_kb)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
