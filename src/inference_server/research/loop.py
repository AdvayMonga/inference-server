"""CLI driver for the research loop. Every step of LOOP.md, deterministic where it can be.

    python -m inference_server.research.loop attribute runs/<id>.json
    python -m inference_server.research.loop screen <hypotheses.json>
    python -m inference_server.research.loop simulate --class steady_interactive --config '{"policy":"fair"}'
    python -m inference_server.research.loop judge --hyp H.json --baseline A.json --treatment B.json
    python -m inference_server.research.loop band --run-group <null run group>
    python -m inference_server.research.loop index
    python -m inference_server.research.loop kb --status rejected
    python -m inference_server.research.loop kb --suspended     # what cannot be measured right now
    python -m inference_server.research.loop no-claim --why "drops an unused import"

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
from inference_server.research.kb import (
    covers,
    format_scope,
    related,
    load_entries,
    query,
    suspends,
    suspensions,
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
    """Order hypotheses cheapest-falsification-first, flag known dead ends, block suspensions.

    Exit 1 if any hypothesis predicts a metric a live suspension says this tier cannot measure.
    Unlike the `related()` flags, which are keyword-scored and therefore advisory, a suspension
    is declared and exact (metric + tier), so it is safe to make it a verdict.
    """
    raw = json.loads(Path(args.hypotheses).read_text())
    hyps = [Hypothesis(**h) for h in (raw if isinstance(raw, list) else [raw])]
    entries = load_entries()

    for h in hyps:
        h.validate()

    # Checked against EVERY entry, not just the ones `related()` surfaces: a suspension is a
    # property of the instrument, not of the subject. A prefix-cache hypothesis that predicts
    # p95 TTFT at tier 1 shares no words with "the simulator has no termination model", and it
    # is exactly as unfalsifiable at that tier as a scheduling one.
    live = suspensions(entries)
    n_stopped = 0

    for h in sorted(hyps, key=lambda x: x.falsification_tier):
        hits = related(h.statement, entries, tags=h.tags)
        blocking = [e for _, e in hits if e.status in ("rejected", "resolved")]
        stopped = [e for e in live if suspends(e, h.predicted_metric, h.falsification_tier)]
        flag = ("  <-- SUSPENDED: THIS TIER CANNOT MEASURE THAT METRIC" if stopped
                else "  <-- READ THE RELATED ENTRIES FIRST" if blocking else "")
        n_stopped += bool(stopped)
        print(f"tier {h.falsification_tier} ({TIER_NAMES[h.falsification_tier]}){flag}")
        if h.falsification_tier <= 2 and not stopped:
            print("  policy hypothesis? `loop simulate --class <cls> --config '{...}'` falsifies "
                  "scheduling / admission / KV-sizing changes without a GPU")
        print(f"  {h.id}  {h.statement}")
        print(f"  predicts {h.predicted_metric} {h.predicted_direction} by "
              f"{h.predicted_magnitude}")
        print(f"  falsify with: {h.falsification_test}")
        for e in stopped:
            print(f"    !! SUSPENDED {h.predicted_metric} at tier {h.falsification_tier}: "
                  f"[{e.status}] {e.id}: {e.title[:60]}")
            print(f"       scope: {format_scope(e)}")
            print(f"       a result from this tier is NOT evidence about {h.predicted_metric}. "
                  f"Use a tier this entry does not suspend, predict a metric it does not, or "
                  f"lift it: fix the instrument and supersede the entry.")
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
    if n_stopped:
        print(f"{n_stopped} of {len(hyps)} hypotheses are SUSPENDED: the loop currently cannot "
              f"falsify them at the tier they name. Do not run them.")
        return 1
    return 0


def cmd_simulate(args) -> int:
    """Tier-1 falsifier: replay one corpus class through the simulator, one row per config."""
    from inference_server.research import harness as H
    from inference_server.research.corpus import load_trace
    from inference_server.research.simulator import (
        PLACEHOLDER_A100_E4B,
        SimConfig,
        TimingModel,
        simulate,
    )

    timing = TimingModel.from_json(args.timing) if args.timing else PLACEHOLDER_A100_E4B
    manifest, trace = load_trace(args.cls, args.split)
    cls = manifest.classes[args.cls]
    print(f"-- simulate {args.cls}/{args.split} ({len(trace)} requests, corpus "
          f"{manifest.corpus_version[:12]}, timing {timing.model}/{timing.hardware} "
          f"fitted_from={timing.fitted_from}) --")
    for raw in args.config or ["{}"]:
        cfg = SimConfig(**json.loads(raw))
        res = simulate(trace, cfg, timing, seed=args.seed)
        s = res.summary(cls)
        print(json.dumps(cfg.to_dict()))
        print("  " + " ".join(f"{k}={v}" for k, v in s.items()))
        if args.emit:
            panel = res.to_panel(cls, corpus_version=manifest.corpus_version, split=args.split,
                                 seed=args.seed)
            H.emit(panel, label=f"simulate {args.cls}/{args.split}",
                   runs_dir=Path(args.runs_dir) if args.runs_dir else None)
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


def cmd_band(args) -> int:
    """Reduce a NULL run group (same config as both arms) to a stored noise band.

    `arms_for` is reused rather than reimplemented: a band measured from unbalanced or
    block-ordered runs would understate the spread in exactly the way that makes the band
    dangerous, and those are the same checks an A/B has to pass.
    """
    from inference_server.research.noise import BAND_DIR, band_from_arms, format_band
    from inference_server.research.session import arms_for

    try:
        arms = arms_for(args.run_group)
    except NotMeasurable as e:
        print(f"NOT MEASURABLE: {e}")
        return 2
    band = band_from_arms(arms, drop_first_per_arm=not args.keep_warmup,
                          entry_id=args.entry or "", notes=args.notes or "")
    if not band.workload_class or not band.model or not band.hardware:
        print("the panels do not identify (workload_class, model, hardware); a band that cannot "
              "say where it applies would be applied everywhere", file=sys.stderr)
        return 1
    name = "-".join(_slug(x) for x in (band.harness, band.workload_class, band.model,
                                       band.hardware))
    out = Path(args.out) if args.out else BAND_DIR / f"{name}.json"
    band.to_json(out)
    print(f"-- noise band from {band.n_runs} run(s) of {band.identity()} --")
    print(format_band(band))
    print(f"\nwrote {out}")
    print("a delta inside this band is reported `inconclusive` by compare.significance_"
          "replicated and fails the significance gate")
    return 0


def _slug(text: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in text)
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")


def cmd_index(args) -> int:
    p = write_index()
    print(f"regenerated {p} from knowledge/ ({len(load_entries())} entries)")
    return 0


def cmd_noclaim(args) -> int:
    """Record that one commit changes no engine behaviour. The record vouches for that sha only."""
    import subprocess

    from inference_server.research.kb import save_experiment
    from inference_server.research.schemas import REPO_ROOT, Arm, Experiment

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()

    sha = git("rev-parse", args.sha)
    base = git("merge-base", args.base, sha)
    exp = Experiment(hypothesis_id="no-behaviour-change", engine_sha_base=base,
                     arms=[Arm("baseline", base), Arm("treatment", sha)],
                     branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                     verdict="confirmed", source="loop", no_behaviour_change=args.why)
    path = save_experiment(exp)
    print(f"recorded {exp.id}: {sha[:12]} claims no behaviour change — {args.why}")
    print(f"  {path}")
    print(f"  commit this file. The claim covers {sha[:12]} only; any engine change after it "
          f"needs its own record.")
    return 0


def parse_situation(spec: str | None) -> dict:
    """`--situation model=E4B,concurrency=8` -> dict; numbers become numbers so ranges compare."""
    out = {}
    for pair in (spec or "").split(","):
        if not pair.strip():
            continue
        key, eq, raw = pair.partition("=")
        if not eq or not key.strip():
            raise argparse.ArgumentTypeError(f"expected key=value, got {pair!r}")
        try:
            val = float(raw) if "." in raw else int(raw)
        except ValueError:
            val = raw
        out[key.strip()] = val
    return out


def cmd_kb(args) -> int:
    hits = query(status=args.status, tags=args.tags or (), text=args.text, regime=args.regime,
                 suspended=args.suspended)
    if args.situation:
        hits = [e for e in hits if covers(e, args.situation)]
    live = {e.id for e in suspensions()}
    for e in hits:
        # An entry with no validity_range is unscoped — it covers() everything by construction,
        # which is not the same as having been verified for this situation.
        scope = "" if e.validity_range or not args.situation else "  [unscoped]"
        print(f"{e.status:9} {e.id}  {e.title}{scope}")
        if e.id in live:
            tiers = ", ".join(str(t) for t in sorted(e.suspended_tiers)) or "all"
            print(f"           SUSPENDS {', '.join(e.suspended_metrics)} at tier(s) {tiers}")
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

    s = sub.add_parser("screen", help="order hypotheses cheapest-first, flag dead ends (step 3); "
                                      "exit 1 if one predicts a suspended metric")
    s.add_argument("hypotheses"); s.set_defaults(fn=cmd_screen)

    m = sub.add_parser("simulate", help="tier-1: replay a corpus class through the simulator, "
                                        "one row per --config (policy hypotheses, no GPU)")
    m.add_argument("--class", dest="cls", required=True)
    m.add_argument("--split", default="seen", choices=("seen", "heldout"))
    m.add_argument("--config", action="append",
                   help='JSON of SimConfig knobs, e.g. \'{"policy": "fair"}\'; repeatable')
    m.add_argument("--timing", help="TimingModel JSON; default is the A100/E4B placeholder")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--emit", action="store_true", help="write a panel per config via harness.emit")
    m.add_argument("--runs-dir")
    m.set_defaults(fn=cmd_simulate)

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

    b = sub.add_parser("band", help="reduce a null run group to a noise band in knowledge/noise/")
    b.add_argument("--run-group", required=True,
                   help="a group whose two arms ran the SAME config (replay_local.py --null)")
    b.add_argument("--out", help="default knowledge/noise/<harness>-<class>-<model>-<hw>.json")
    b.add_argument("--entry", help="knowledge entry id this band belongs to")
    b.add_argument("--notes", default="")
    b.add_argument("--keep-warmup", action="store_true",
                   help="do not drop each arm's first run (reports the raw spread instead of "
                        "the one the significance gate faces)")
    b.set_defaults(fn=cmd_band)

    i = sub.add_parser("index", help="regenerate DECISIONS.md from knowledge/")
    i.set_defaults(fn=cmd_index)

    k = sub.add_parser("kb", help="query the knowledge base")
    k.add_argument("--status"); k.add_argument("--tags", nargs="*")
    k.add_argument("--text"); k.add_argument("-v", "--verbose", action="store_true")
    k.add_argument("--regime",
                   help="only entries for this regime (cold_start, steady_interactive, long_context)")
    k.add_argument("--suspended", action="store_true",
                   help="only entries whose suspension is live: a metric the loop cannot "
                        "currently measure at the tiers named")
    k.add_argument("--situation", type=parse_situation,
                   help="key=value[,key=value]; keep entries whose validity_range covers it")
    k.set_defaults(fn=cmd_kb)

    n = sub.add_parser("no-claim",
                       help="record that a commit changes no engine behaviour (dead code, rename, comment)")
    n.add_argument("--why", required=True,
                   help="one line: why this diff cannot change what the engine does")
    n.add_argument("--sha", default="HEAD", help="the commit the claim covers")
    n.add_argument("--base", default="main")
    n.set_defaults(fn=cmd_noclaim)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
