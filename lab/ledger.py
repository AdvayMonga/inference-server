"""ledger: append-only record of every tool call, the raw knowledge base (format: lab/README.md)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = 1
ROOT = Path("lab/ledger")
KINDS = {"test", "equiv", "bench", "profile", "submit", "finding"}
WRITER_FIELDS = {"id", "at", "schema"}
HELDOUT_METRIC_KEYS = {"base", "new", "delta_pct", "band_pct", "verdict"}


class LedgerError(ValueError):
    pass


def _file(root: Path) -> Path:
    return Path(root) / "ledger.jsonl"


def _check(record: dict) -> None:
    """Integrity rules only: known kind, writer-owned fields, no held-out detail beyond one delta per metric."""
    if record.get("kind") not in KINDS:
        raise LedgerError(f"unknown kind {record.get('kind')!r}; one of {sorted(KINDS)}")
    if WRITER_FIELDS & record.keys():
        raise LedgerError(f"{sorted(WRITER_FIELDS & record.keys())} are set by the writer")
    if (record.get("config") or {}).get("split") != "heldout":
        return
    if record.get("raw"):
        raise LedgerError("held-out records carry no raw data")
    for name, m in (record.get("metrics") or {}).items():
        if not isinstance(m, dict) or not m.keys() <= HELDOUT_METRIC_KEYS:
            raise LedgerError(f"held-out metric {name!r} may only carry {sorted(HELDOUT_METRIC_KEYS)}")


def append(record: dict, root: Path = ROOT) -> dict:
    """Validate, stamp id/at/schema, append one line; return the stored record."""
    _check(record)
    stored = {"id": f"ev-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "schema": SCHEMA, **record}
    line = json.dumps(stored, default=str) + "\n"
    path = _file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)    # parallel sessions share one ledger
        f.write(line)
        f.flush()
    return stored


def put_blob(src: Path | bytes, root: Path = ROOT, suffix: str = "") -> str:
    """Store a file, directory or bytes by content hash; return its path relative to `root`."""
    h = hashlib.sha256()
    if isinstance(src, bytes):
        h.update(src)
    elif Path(src).is_dir():
        for p in sorted(Path(src).rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(src)).encode())
                h.update(p.read_bytes())
    else:
        h.update(Path(src).read_bytes())
    rel = f"blobs/{h.hexdigest()[:16]}{suffix}"
    dest = Path(root) / rel
    if dest.exists():
        return rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(src, bytes):
        dest.write_bytes(src)
    elif Path(src).is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copyfile(src, dest)
    return rel


def records(root: Path = ROOT, **match: Any) -> Iterator[dict]:
    """Every record in write order, optionally only those whose top-level fields equal `match`."""
    path = _file(root)
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if all(r.get(k) == v for k, v in match.items()):
                yield r


def seed(knowledge: Path, root: Path = ROOT) -> int:
    """Import hand-written findings as `finding` records, whole and unchanged; skip ones already seeded."""
    done = {r.get("source") for r in records(root, kind="finding")}
    n = 0
    for p in sorted(Path(knowledge).glob("*.json")):
        source = f"{Path(knowledge).name}/{p.name}"
        if source in done:
            continue
        append({"kind": "finding", "author": "human", "source": source,
                "claim": json.loads(p.read_text())}, root)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.ledger")
    ap.add_argument("--root", type=Path, default=ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="print records as JSONL")
    for k in ("kind", "session", "snapshot"):
        ls.add_argument(f"--{k}")
    sub.add_parser("show", help="pretty-print one record").add_argument("id")
    sub.add_parser("seed", help="import knowledge/*.json").add_argument(
        "--knowledge", type=Path, default=Path("knowledge"))
    a = ap.parse_args(argv)

    if a.cmd == "list":
        match = {k: getattr(a, k) for k in ("kind", "session", "snapshot") if getattr(a, k)}
        for r in records(a.root, **match):
            print(json.dumps(r))
    elif a.cmd == "show":
        hit = next(records(a.root, id=a.id), None)
        if hit is None:
            print(f"no record {a.id}", file=sys.stderr)
            return 1
        print(json.dumps(hit, indent=1))
    else:
        print(f"seeded {seed(a.knowledge, a.root)} findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
