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
from typing import Iterable

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
) -> list[KnowledgeEntry]:
    """What the hypothesis step calls before proposing anything: has this been tried?"""
    items = list(entries) if entries is not None else load_entries()
    tagset = set(tags)
    if tagset:
        items = [e for e in items if tagset & set(e.tags)]
    if status:
        items = [e for e in items if e.status == status]
    if text:
        t = text.lower()
        items = [e for e in items
                 if t in e.title.lower() or t in e.summary.lower()
                 or any(t in tag for tag in e.tags)]
    return items


def already_rejected(statement: str, entries: Iterable[KnowledgeEntry] | None = None,
                     ) -> list[KnowledgeEntry]:
    """Cheap guard against re-proposing a measured dead end. Keyword overlap, not semantics —
    it is a prompt to read, not an authority."""
    items = [e for e in (entries if entries is not None else load_entries())
             if e.status == "rejected"]
    words = {w for w in _slug(statement).split("-") if len(w) > 4}
    hits = []
    for e in items:
        ewords = {w for w in _slug(e.title + "-" + e.summary).split("-") if len(w) > 4}
        if len(words & ewords) >= 2:
            hits.append(e)
    return hits


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
    for status in STATUS_ORDER:
        group = by_status.get(status) or []
        if not group:
            continue
        lines += [f"## {status.capitalize()} ({len(group)})", "", STATUS_BLURB[status], ""]
        for e in sorted(group, key=lambda x: -x.updated_at):
            date = time.strftime("%Y-%m-%d", time.localtime(e.created_at))
            lines.append(f"### [{date}] {e.title}")
            if e.tags:
                lines.append(f"*tags: {', '.join(f'`{t}`' for t in e.tags)}* · `{e.id}`")
            lines += ["", e.summary.rstrip(), ""]
            if e.triggers:
                lines += ["**Revisit when:** " + "; ".join(e.triggers), ""]
            if e.evidence:
                ev = ", ".join(v for d in e.evidence for v in d.values())
                lines += [f"**Evidence:** {ev}", ""]
            if e.superseded_by:
                lines += [f"**Superseded by:** `{e.superseded_by}`", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_index(path: Path | None = None, entries: list[KnowledgeEntry] | None = None) -> Path:
    target = path or (REPO_ROOT / "DECISIONS.md")
    target.write_text(generate_index(entries))
    return target
