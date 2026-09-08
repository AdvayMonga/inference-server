"""CLI driver for the research loop. Every step of LOOP.md, deterministic where it can be.

    python -m inference_server.research.loop attribute runs/<id>.json
    python -m inference_server.research.loop screen <hypotheses.json> [--venues vast-4090]
    python -m inference_server.research.loop budget [--limit-usd 5] [--reset]
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
from pathlib import Path

from inference_server.research.attribute import attribute
from inference_server.research.budget import (
    VENUES,
    VENUES_ENV,
    Budget,
    Footprint,
    Unroutable,
    available_venues,
    preflight,
    route,
)
from inference_server.research.kb import (
    load_entries,
    query,
    related,
    write_index,
)
from inference_server.research.schemas import Hypothesis, Vitals
from inference_server.research.session import NotMeasurable, format_judgement, judge_group

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
    """Order hypotheses cheapest-falsification-first, place each on a venue, flag dead ends."""
    raw = json.loads(Path(args.hypotheses).read_text())
    hyps = [Hypothesis(**h) for h in (raw if isinstance(raw, list) else [raw])]
    entries = load_entries()
    venues = available_venues(args.venues)
    budget = Budget.load(limit_usd=args.limit_usd)
    footprint = Footprint.from_env()

    for h in hyps:
        h.validate()

    # Sort by what it actually costs here, not by tier alone: a tier-3 probe on local MPS is
    # free, a tier-3 probe that needs Triton is not.
    placed = []
    for h in hyps:
        try:
            placed.append((h, route(h, venues, footprint), None))
        except Unroutable as e:
            placed.append((h, None, str(e)))
    placed.sort(key=lambda t: (t[1].est_usd if t[1] else float("inf"), t[0].falsification_tier))

    sizes = ", ".join(f"{v.name} {v.mem_gb or '?'}GB" for v in venues)
    print(f"venues: {sizes}   "
          f"budget: ${budget.remaining_usd:.2f} of ${budget.limit_usd:.2f} left")
    print(f"footprint (from MODEL_NAME / CUSTOM_BACKEND_BLOCKS): {footprint.model}, "
          f"{footprint.blocks}x{footprint.block_size} blocks\n")
    for h, pl, err in placed:
        hits = related(h.statement, entries, tags=h.tags)
        blocking = [e for _, e in hits if e.status in ("rejected", "resolved")]
        flag = "  <-- READ THE RELATED ENTRIES FIRST" if blocking else ""
        print(f"tier {h.falsification_tier} ({TIER_NAMES[h.falsification_tier]}){flag}")
        print(f"  {h.id}  {h.statement}")
        print(f"  predicts {h.predicted_metric} {h.predicted_direction} by "
              f"{h.predicted_magnitude}")
        print(f"  falsify with: {h.falsification_test}")
        if pl is None:
            print(f"    ! NO VENUE: {err}")
        else:
            ok, why = budget.can_afford(pl.tier, usd_per_hour=pl.venue.usd_per_hour)
            print(f"  run on: {pl.venue.name} ({pl.reason})"
                  + ("" if ok else f"\n    ! OVER BUDGET: {why}"))
            if pl.tier >= 3:
                fit_ok, fit_why = preflight(pl.venue, footprint)
                print(f"  memory: {fit_why}" if fit_ok else f"    ! MEMORY: {fit_why}")
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


def cmd_budget(args) -> int:
    """Show the cap, the ledger, and which venues this session can route to."""
    budget = Budget.load(limit_usd=args.limit_usd)
    if args.reset:
        budget.reset(args.limit_usd).save()
        print("ledger reset")
    print(f"cap ${budget.limit_usd:.2f}   spent ${budget.spent_usd:.2f}   "
          f"remaining ${budget.remaining_usd:.2f}")
    for e in budget.entries:
        print(f"  tier {e['tier']}  ${e['usd']:.2f}  {e.get('venue') or '-':12} {e['label']}")
    avail = {v.name: v for v in available_venues(args.venues)}
    print(f"\nvenues (cloud is opt-in via {VENUES_ENV} or --venues):")
    for v in VENUES.values():
        v = avail.get(v.name, v)
        mark = "on " if v.name in avail else "off"
        print(f"  [{mark}] {v.name:12} ${v.usd_per_hour:.2f}/h  {str(v.mem_gb or '?'):>5} GB  "
              f"{', '.join(sorted(v.capabilities)) or 'cpu only'}")
    return 0


def cmd_judge(args) -> int:
    """Thin shim. The procedure lives in session.judge_group so the CLI cannot drift from it —
    this one did, silently: it took a single panel per arm long after single-run arms stopped
    being able to clear the significance gate."""
    hyp = Hypothesis(**json.loads(Path(args.hyp).read_text()))
    try:
        j, exp = judge_group(hyp, args.run_group, baseline=args.baseline,
                             treatment=args.treatment, branch=args.branch or "",
                             run_tests=not args.no_tests, check_drift=not args.no_drift_check)
    except NotMeasurable as e:
        print(f"NOT MEASURABLE: {e}")
        return 2
    print(format_judgement(j, hyp, exp))
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
    s.add_argument("hypotheses")
    s.add_argument("--venues",
                   help=f"cloud venues to allow, e.g. vast-4090 (default: ${VENUES_ENV})")
    s.add_argument("--limit-usd", type=float, default=5.0)
    s.set_defaults(fn=cmd_screen)

    b = sub.add_parser("budget", help="show the per-iteration cap, ledger and venues")
    b.add_argument("--venues")
    b.add_argument("--limit-usd", type=float, default=5.0)
    b.add_argument("--reset", action="store_true", help="start a fresh iteration")
    b.set_defaults(fn=cmd_budget)

    j = sub.add_parser("judge", help="run the five gates over a run group and record it (5-6)")
    j.add_argument("--hyp", required=True)
    j.add_argument("--run-group", required=True,
                   help="arms are the panels in runs/ carrying this run_group")
    j.add_argument("--baseline", default="baseline", help="arm= label of the control")
    j.add_argument("--treatment", default="treatment", help="arm= label of the change")
    j.add_argument("--branch"); j.add_argument("--no-tests", action="store_true")
    j.add_argument("--no-drift-check", action="store_true",
                   help="judge panels whose engine sha has since moved (you almost never want this)")
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
