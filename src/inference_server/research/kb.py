"""Knowledge base and experiment ledger — git-native, one JSON file per record.

Why files and not a database: the loop commits its own work, so its memory should live in the
same history as the code it changed. A record is then diffable, reviewable in a PR, greppable,
and survives any tooling rewrite. At this scale (hundreds of entries) an index is unnecessary.

DECISIONS.md becomes a GENERATED VIEW of `knowledge/`. Edit the JSON; regenerate the markdown.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from inference_server.research.schemas import (
    REPO_ROOT,
    Experiment,
    KnowledgeEntry,
    SchemaError,
)

KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RUNS_DIR = REPO_ROOT / "runs"

# Order matters in the generated view: what is still live comes first, what is settled last.
STATUS_ORDER = ("open", "deferred", "resolved", "rejected", "obsolete")
STATUS_BLURB = {
    "open": "Live — being worked, or waiting on a trigger.",
    "deferred": "Deliberately not doing this yet; each carries the trigger that would change that.",
    "resolved": "Settled. Kept because the reasoning still constrains new work.",
    "rejected": "**Tried, measured, does not work.** Do not retry blind — these are the entries "
                "that stop the loop re-treading dead ends.",
    "obsolete": "Superseded. Kept only so old commits remain readable.",
}


def _slug(text: str, limit: int = 48) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in text]
    s = "".join(keep)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")[:limit]


# --------------------------------------------------------------------------- knowledge

def save_entry(entry: KnowledgeEntry, directory: Path = KNOWLEDGE_DIR) -> Path:
    entry.validate()
    entry.updated_at = time.time()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry.id}.json"
    path.write_text(json.dumps(asdict(entry), indent=2, sort_keys=True))
    return path


def load_entries(directory: Path = KNOWLEDGE_DIR) -> list[KnowledgeEntry]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        try:
            out.append(KnowledgeEntry(**json.loads(p.read_text())))
        except TypeError as e:
            raise SchemaError(f"{p.name} is not a KnowledgeEntry: {e}") from e
    return out


def query(
    entries: Iterable[KnowledgeEntry] | None = None,
    *,
    tags: Iterable[str] = (),
    status: str | None = None,
    text: str | None = None,
    regime: str | None = None,
    suspended: bool = False,
) -> list[KnowledgeEntry]:
    """What the hypothesis step calls before proposing anything: has this been tried?"""
    items = list(entries) if entries is not None else load_entries()
    tagset = set(tags)
    if tagset:
        items = [e for e in items if tagset & set(e.tags)]
    if status:
        items = [e for e in items if e.status == status]
    if suspended:
        items = suspensions(items)
    if regime:
        items = [e for e in items if e.regime == regime]
    if text:
        t = text.lower()
        items = [e for e in items
                 if t in e.title.lower() or t in e.summary.lower()
                 or any(t in tag for tag in e.tags)]
    return items


def suspensions(entries: Iterable[KnowledgeEntry] | None = None) -> list[KnowledgeEntry]:
    """Entries whose suspension is still LIVE.

    A suspension is lifted by superseding the entry — file the record that shows the instrument
    measuring again, point `supersedes`/`superseded_by` at each other, and this stops returning
    the old one. That is how the same suspension was already lifted once by hand
    (kb-20260917-c07eb94b -> kb-20260918-9fc68282) and re-imposed
    (-> kb-20260919-94acfdb8). Deleting the fields instead would leave no record that the gate
    was ever there, which is the thing a loop with memory must not allow.
    """
    items = list(entries) if entries is not None else load_entries()
    return [e for e in items
            if e.suspended_metrics and e.superseded_by is None and e.status != "obsolete"]


def suspends(entry: KnowledgeEntry, metric: str, tier: int) -> bool:
    """Does this entry say `metric` cannot be measured at falsification tier `tier`?"""
    return (metric in entry.suspended_metrics
            and (not entry.suspended_tiers or tier in entry.suspended_tiers))


def format_scope(entry: KnowledgeEntry) -> str:
    """Where a suspension applies, from the fields that already say where an entry applies."""
    parts = [f"regime={entry.regime}"] if entry.regime else []
    parts += [f"{k}={v}" for k, v in sorted(entry.validity_range.items())]
    return ", ".join(parts) or "unscoped — no regime or validity_range, so it applies everywhere"


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def covers(entry: KnowledgeEntry, situation: dict[str, Any]) -> bool:
    """Does `situation` fall inside every bound the entry's validity_range states?

    Keys the range does not mention are unconstrained. A 2-list of numbers is an inclusive
    range `lo <= x <= hi` (a non-numeric x is outside it); any other list is membership
    ("one of"); a scalar is equality. Deliberately no "nearest entry": outside the measured bounds is uncovered,
    full stop — extrapolating a finding is the failure this field exists to stop.
    """
    for key, bound in entry.validity_range.items():
        if key not in situation:
            continue
        x = situation[key]
        if isinstance(bound, list):
            if len(bound) == 2 and all(map(_num, bound)):
                inside = _num(x) and bound[0] <= x <= bound[1]
            else:
                inside = x in bound
        else:
            inside = bound == x
        if not inside:
            return False
    return True


def related(statement: str, entries: Iterable[KnowledgeEntry] | None = None,
            limit: int = 5, tags: Iterable[str] = ()) -> list[tuple[float, KnowledgeEntry]]:
    """Entries worth reading before proposing `statement`, most relevant first.

    Advisory, and deliberately not a verdict. The first real iteration proved a verdict-shaped
    detector is worse than none: matching keywords against only the `rejected` set flagged the
    hypothesis that went on to find a 4.4x gap, while missing 'chunk long prefills across ticks'
    — a scheduling lever an already-`resolved` entry rules out. So: search EVERY status (a
    resolved finding rules things out just as hard as a rejected one), score by weighted term
    overlap, and let the caller read.
    """
    items = list(entries) if entries is not None else load_entries()
    terms = {w for w in _slug(statement).split("-") if len(w) > 3}
    if not terms:
        return []

    want_tags = set(tags)
    scored: list[tuple[float, KnowledgeEntry]] = []
    for e in items:
        # Tag overlap dominates: it is declared, not inferred, so it does not miss an entry
        # just because the wording differs.
        tag_hits = len(want_tags & set(e.tags))
        title_terms = {w for w in _slug(e.title).split("-") if len(w) > 3}
        body_terms = {w for w in _slug(e.summary).split("-") if len(w) > 3}
        tag_terms = {w for t in e.tags for w in _slug(t).split("-") if len(w) > 3}
        # A title or tag hit means the entry is ABOUT this; a body hit only mentions it.
        score = (6.0 * tag_hits
                 + 3.0 * len(terms & title_terms)
                 + 2.0 * len(terms & tag_terms)
                 + 1.0 * len(terms & body_terms) / max(len(body_terms), 1) * 4)
        # A settled finding constrains new work more than an open question does.
        if e.status in ("rejected", "resolved"):
            score *= 1.5
        if score >= 3.0:
            scored.append((round(score, 2), e))
    scored.sort(key=lambda kv: -kv[0])
    return scored[:limit]


def already_rejected(statement: str, entries: Iterable[KnowledgeEntry] | None = None,
                     ) -> list[KnowledgeEntry]:
    """Kept for callers that want only the hard 'this was tried and failed' set."""
    return [e for _, e in related(statement, entries, limit=10) if e.status == "rejected"]


# --------------------------------------------------------------------------- experiments

def save_experiment(exp: Experiment, directory: Path = EXPERIMENTS_DIR) -> Path:
    exp.validate()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{exp.id}.json"
    payload = asdict(exp)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_experiments(directory: Path = EXPERIMENTS_DIR) -> list[Experiment]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.glob("*.json")):
        d = json.loads(p.read_text())
        arms = d.pop("arms", [])
        exp = Experiment(**d)
        from inference_server.research.schemas import Arm
        exp.arms = [Arm(**a) for a in arms]
        out.append(exp)
    return out


def experiment_for_sha(sha: str, directory: Path = EXPERIMENTS_DIR) -> Experiment | None:
    """Used by the pre-merge gate: does a merge have an experiment backing it?"""
    for exp in load_experiments(directory):
        if any(arm.sha.startswith(sha) or sha.startswith(arm.sha) for arm in exp.arms):
            return exp
    return None


# --------------------------------------------------------------------------- generated views

def generate_index(entries: list[KnowledgeEntry] | None = None) -> str:
    items = entries if entries is not None else load_entries()
    by_status: dict[str, list[KnowledgeEntry]] = {s: [] for s in STATUS_ORDER}
    for e in items:
        by_status.setdefault(e.status, []).append(e)

    all_tags: dict[str, int] = {}
    for e in items:
        for t in e.tags:
            all_tags[t] = all_tags.get(t, 0) + 1

    lines = [
        "# Decisions Log",
        "",
        "> **Generated file — do not edit.** Source of truth is `knowledge/*.json`.",
        "> Regenerate with `python -m inference_server.research.loop index`.",
        "",
        f"{len(items)} entries. "
        f"Tags: {', '.join(f'`{t}`({n})' for t, n in sorted(all_tags.items(), key=lambda kv: -kv[1]))}",
        "",
        "Grep by tag or title rather than reading top-to-bottom.",
        "",
    ]
    # Regime-keyed section: the cheap lookup for "what is known about regime R". Entries with no
    # regime are counted, not listed — the status sections below already carry them.
    by_regime: dict[str, list[KnowledgeEntry]] = {}
    for e in items:
        by_regime.setdefault(e.regime or "unassigned", []).append(e)
    lines += ["## By regime", ""]
    for regime in sorted(by_regime, key=lambda r: (r == "unassigned", r)):
        group = by_regime[regime]
        if regime == "unassigned":
            lines.append(f"- **unassigned** ({len(group)}) — no `regime` field yet")
        else:
            refs = ", ".join(f"`{e.id}`" for e in sorted(group, key=lambda x: x.id))
            lines.append(f"- **{regime}** ({len(group)}): {refs}")
    lines.append("")
    # Live suspensions go near the top, above every status section: a metric the loop cannot
    # currently measure changes what is worth proposing, so it has to be visible before the
    # findings are. `loop screen` enforces this list; this section is the human-readable copy.
    live = suspensions(items)
    live_ids = {e.id for e in live}
    if live:
        lines += ["## Active suspensions", "",
                  "Metrics the loop currently **cannot measure** at the tiers named. `loop "
                  "screen` blocks a hypothesis that predicts one of them. Lifted by superseding "
                  "the entry with one that shows the instrument working again.", ""]
        for e in sorted(live, key=lambda x: x.id):
            tiers = (", ".join(f"tier {t}" for t in sorted(e.suspended_tiers))
                     or "every tier")
            lines.append(f"- `{', '.join(e.suspended_metrics)}` at {tiers} "
                         f"({format_scope(e)}) — `{e.id}`: {e.title}")
        lines.append("")
    for status in STATUS_ORDER:
        group = by_status.get(status) or []
        if not group:
            continue
        lines += [f"## {status.capitalize()} ({len(group)})", "", STATUS_BLURB[status], ""]
        for e in sorted(group, key=lambda x: -x.updated_at):
            # UTC, not localtime: DECISIONS.md is generated AND tracked, so the same
            # knowledge/ must render identically on a laptop and on a CI runner.
            date = time.strftime("%Y-%m-%d", time.gmtime(e.created_at))
            lines.append(f"### [{date}] {e.title}")
            if e.tags:
                lines.append(f"*tags: {', '.join(f'`{t}`' for t in e.tags)}* · `{e.id}`")
            lines += ["", e.summary.rstrip(), ""]
            if e.triggers:
                lines += ["**Revisit when:** " + "; ".join(e.triggers), ""]
            if e.evidence:
                ev = ", ".join(v for d in e.evidence for v in d.values())
                lines += [f"**Evidence:** {ev}", ""]
            if e.suspended_metrics:
                tiers = ", ".join(str(t) for t in sorted(e.suspended_tiers)) or "all"
                state = "LIVE" if e.id in live_ids else "lifted (superseded)"
                lines += [f"**Suspends:** `{', '.join(e.suspended_metrics)}` at tier(s) {tiers} "
                          f"— {state}", ""]
            if e.regime:
                lines += [f"**Regime:** `{e.regime}`", ""]
            if e.validity_range:
                lines += [f"**Valid over:** `{json.dumps(e.validity_range, sort_keys=True)}`", ""]
            if e.mechanism:
                lines += [f"**Mechanism:** {e.mechanism}", ""]
            if e.transfer_checked:
                lines += [f"**Transfer checked:** "
                          f"`{json.dumps(e.transfer_checked, sort_keys=True)}`", ""]
            if e.supersedes:
                lines += [f"**Supersedes:** `{e.supersedes}`", ""]
            if e.superseded_by:
                lines += [f"**Superseded by:** `{e.superseded_by}`", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_index(path: Path | None = None, entries: list[KnowledgeEntry] | None = None) -> Path:
    target = path or (REPO_ROOT / "archive" / "DECISIONS.md")
    target.parent.mkdir(exist_ok=True)
    target.write_text(generate_index(entries))
    return target
