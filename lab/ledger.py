"""ledger: append-only record of every tool call, the raw knowledge base (format: lab/README.md)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[1] / "lab" / "ledger"
KINDS = {"test", "equiv", "bench", "profile", "submit", "finding", "session", "note"}
WRITER_FIELDS = {"id", "at", "schema"}
# A held-out record is one aggregate per metric and nothing else: these fields, scalar values.
HELDOUT_FIELDS = {"kind", "config", "metrics", "run", "session", "snapshot", "snapshot_blob", "patch",
                  "tool", "args", "cost", "claim"}
HELDOUT_METRIC_KEYS = {"base", "new", "delta_pct", "band_pct", "verdict"}
SCALAR = (int, float, str, bool, type(None))


class LedgerError(ValueError):
    pass


def _file(root: Path) -> Path:
    return Path(root) / "ledger.jsonl"


@contextlib.contextmanager
def _lock(root: Path):
    """One lock file per ledger; parallel sessions share it. Held across whole read-then-write steps."""
    Path(root).mkdir(parents=True, exist_ok=True)
    with (Path(root) / "ledger.lock").open("a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _check(record: dict) -> None:
    """Integrity rules only: known kind, writer-owned fields, no held-out detail beyond one aggregate per metric."""
    if record.get("kind") not in KINDS:
        raise LedgerError(f"unknown kind {record.get('kind')!r}; one of {sorted(KINDS)}")
    if WRITER_FIELDS & record.keys():
        raise LedgerError(f"{sorted(WRITER_FIELDS & record.keys())} are set by the writer")
    config = record.get("config")
    if config is not None and not isinstance(config, dict):
        raise LedgerError("config must be an object")
    if (config or {}).get("split") != "heldout":
        return
    extra = record.keys() - HELDOUT_FIELDS
    if extra:
        raise LedgerError(f"held-out records may only carry {sorted(HELDOUT_FIELDS)}; got {sorted(extra)}")
    metrics = record.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise LedgerError("metrics must be an object")
    for name, m in metrics.items():
        if (not isinstance(m, dict) or not m.keys() <= HELDOUT_METRIC_KEYS
                or not all(isinstance(v, SCALAR) for v in m.values())):
            raise LedgerError(f"held-out metric {name!r} may only carry scalar "
                              f"{sorted(HELDOUT_METRIC_KEYS)}")


def _stamp(record: dict) -> dict:
    now = datetime.now(timezone.utc)
    return {"id": f"ev-{now:%Y%m%d}-{uuid.uuid4().hex[:12]}",
            "at": now.isoformat(timespec="seconds"), "schema": SCHEMA, **record}


def _write(stored: dict, root: Path) -> None:
    """One line, one os.write, so a reader never sees half a record. Caller holds the lock."""
    try:
        line = json.dumps(stored) + "\n"
    except TypeError as e:
        raise LedgerError(f"record is not JSON (lossless means no coercion): {e}")
    fd = os.open(_file(root), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def append(record: dict, root: Path = ROOT) -> dict:
    """Validate, stamp id/at/schema, append one line; return the stored record."""
    _check(record)
    stored = _stamp(record)
    with _lock(root):
        _write(stored, root)
    return stored


def _hash(src: Path | bytes) -> str:
    """Content hash; directories hash each file's path and sha with length prefixes, so layouts cannot collide."""
    h = hashlib.sha256()
    if isinstance(src, bytes):
        h.update(b"bytes\0" + src)
    elif Path(src).is_dir():
        h.update(b"dir\0")
        for p in sorted(Path(src).rglob("*")):
            if p.is_file():
                name = str(p.relative_to(src)).encode()
                h.update(f"{len(name)}\0".encode() + name + hashlib.sha256(p.read_bytes()).digest())
    else:
        h.update(b"file\0" + Path(src).read_bytes())
    return h.hexdigest()


def put_blob(src: Path | bytes, root: Path = ROOT, suffix: str = "") -> str:
    """Store a file, directory or bytes by content hash; return its path relative to `root`."""
    rel = f"blobs/{_hash(src)[:24]}{suffix}"
    dest = Path(root) / rel
    with _lock(root):
        if dest.exists():
            return rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp-{os.getpid()}")
        if isinstance(src, bytes):
            tmp.write_bytes(src)
        elif Path(src).is_dir():
            shutil.copytree(src, tmp)
        else:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dest)          # a blob exists whole or not at all
    return rel


def records(root: Path = ROOT, **match: Any) -> Iterator[dict]:
    """Every record in write order, optionally only those whose top-level fields equal `match`.

    A torn final line (a writer killed mid-append) is skipped with a warning; a bad line anywhere
    else is corruption and raises.
    """
    path = _file(root)
    if not path.exists():
        return
    lines = path.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for i, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"ledger: skipping torn final line in {path}", file=sys.stderr)
                return
            raise LedgerError(f"{path}:{i + 1} is not JSON")
        if all(r.get(k) == v for k, v in match.items()):
            yield r


def seed(knowledge: Path, root: Path = ROOT) -> int:
    """Import hand-written findings as `finding` records, whole and unchanged. A changed file seeds again."""
    with _lock(root):
        done = {(r.get("source"), r.get("content_sha")) for r in records(root, kind="finding")}
        n = 0
        for p in sorted(Path(knowledge).glob("*.json")):
            raw = p.read_bytes()
            key = (f"{Path(knowledge).name}/{p.name}", hashlib.sha256(raw).hexdigest())
            if key in done:
                continue
            record = {"kind": "finding", "author": "human", "source": key[0],
                      "content_sha": key[1], "claim": json.loads(raw)}
            _check(record)
            _write(_stamp(record), root)
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
        "--knowledge", type=Path, default=ROOT.parents[1] / "knowledge")
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
