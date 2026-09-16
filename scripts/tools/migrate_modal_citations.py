#!/usr/bin/env python3
"""One-shot: re-point knowledge/ citations at scripts/archive/modal/ after the Modal move.

13 entries cite a `*_modal.py` instrument as the provenance for a measured number. Archiving
the instruments (2026-09-16) broke every path-qualified citation, which would have orphaned
the audit trail DECISIONS.md is generated from.

Rewrites `summary` and `evidence` only, by basename, so a bare `bench_serving_modal.py` in
prose becomes resolvable too. Idempotent: an already-migrated path matches and rewrites to
itself. Kept for provenance alongside the other one-shot migrations in this folder.

    PYTHONPATH=src python scripts/tools/migrate_modal_citations.py [--dry-run]
"""

from __future__ import annotations

import re
import sys

from inference_server.research import kb
from inference_server.research.schemas import REPO_ROOT

ARCHIVE = REPO_ROOT / "scripts" / "archive" / "modal"

# An optional directory prefix plus the basename, so `scripts/probes/x_modal.py`, a bare
# `x_modal.py`, and an already-migrated path all collapse to the same replacement.
CITATION = re.compile(r"(?:[\w.\-/]+/)?(?P<base>[\w.\-]+_modal\.py)")


def _index() -> dict[str, str]:
    """basename -> path relative to the repo root, for every archived instrument."""
    return {p.name: str(p.relative_to(REPO_ROOT)) for p in ARCHIVE.rglob("*_modal.py")}


def _rewrite(text: str, index: dict[str, str]) -> str:
    return CITATION.sub(lambda m: index.get(m["base"], m[0]), text)


def main(argv: list[str]) -> int:
    dry_run = "--dry-run" in argv
    index = _index()
    if not index:
        print(f"no archived instruments under {ARCHIVE} — nothing to migrate", file=sys.stderr)
        return 1

    changed = 0
    for entry in kb.load_entries():
        before = (entry.summary, [dict(d) for d in entry.evidence])
        entry.summary = _rewrite(entry.summary, index)
        entry.evidence = [{k: _rewrite(v, index) for k, v in d.items()} for d in entry.evidence]
        if before == (entry.summary, entry.evidence):
            continue
        changed += 1
        fields = []
        if before[0] != entry.summary:
            fields.append("summary")
        if before[1] != entry.evidence:
            fields.append(f"evidence={entry.evidence}")
        print(f"{entry.id}: {', '.join(fields)}")
        if not dry_run:
            kb.save_entry(entry)

    verb = "would update" if dry_run else "updated"
    print(f"{verb} {changed} entr{'y' if changed == 1 else 'ies'} "
          f"against {len(index)} archived instruments")
    if changed and not dry_run:
        print("now regenerate the view: python -m inference_server.research.loop index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
