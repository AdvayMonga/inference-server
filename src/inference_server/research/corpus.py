"""The workload corpus: frozen, hashed traces per workload class, split seen / held-out.

The corpus defines the landscape the loop searches — anything not in it is invisible. Traces are
data committed to `corpus/`, never generated at run time, and every file is hashed into a
`corpus_version` that travels in the panel's validity block so compare.py can refuse a comparison
across corpus drift. A modified trace is a new version, never a silent change: `load_manifest`
raises on any hash mismatch.

Stdlib only — the loop CI lane installs nothing else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inference_server.research.schemas import REPO_ROOT

CORPUS_SCHEMA_VERSION = 1
CORPUS_DIR = REPO_ROOT / "corpus"
SPLITS = ("seen", "heldout")


class CorpusError(ValueError):
    """The corpus on disk does not match its manifest. Always fatal."""


@dataclass
class TraceRequest:
    """One line of a trace. `arrival_s` is an offset from trace start on the replayer's clock."""

    arrival_s: float
    session_id: str
    turn_index: int
    prompt: str
    max_tokens: int
    sampling: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0, "top_p": 1.0,
                                                               "top_k": 0})
    expected_output_hash: str | None = None    # filled by a reference run; the correctness oracle
    expected_output_tokens: int | None = None  # ditto; the termination oracle. None = not measured

    def to_dict(self) -> dict[str, Any]:
        """`expected_output_tokens` is omitted when unset, so a trace written without it is byte-
        identical to one written before the field existed and the corpus_version does not move.
        `expected_output_hash` keeps serialising its null: it is already in every committed trace."""
        d = asdict(self)
        if d["expected_output_tokens"] is None:
            del d["expected_output_tokens"]
        return d


@dataclass
class WorkloadClass:
    name: str
    description: str
    slo_ttft_ms: float
    slo_tpot_ms: float | None
    arrival_rate_rps: float
    seen: str          # trace path, relative to the corpus dir
    heldout: str

    def trace_file(self, split: str) -> str:
        if split not in SPLITS:
            raise CorpusError(f"split must be one of {SPLITS}, got {split!r}")
        return getattr(self, split)

    def within_slo(self, ttft_p95_ms: float, tpot_p95_ms: float | None) -> bool:
        """The class SLO judges a run: p95 TTFT under its ceiling, and p95 TPOT when one is set."""
        if ttft_p95_ms >= self.slo_ttft_ms:
            return False
        if self.slo_tpot_ms is not None and (tpot_p95_ms is None or tpot_p95_ms >= self.slo_tpot_ms):
            return False
        return True


@dataclass
class Manifest:
    schema_version: int
    corpus_version: str
    classes: dict[str, WorkloadClass]
    files: dict[str, str]        # relative path -> sha256
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "corpus_version": self.corpus_version,
                "classes": {k: asdict(v) for k, v in self.classes.items()},
                "files": dict(sorted(self.files.items())), "notes": self.notes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        return cls(schema_version=d["schema_version"], corpus_version=d["corpus_version"],
                   classes={k: WorkloadClass(**v) for k, v in d["classes"].items()},
                   files=dict(d["files"]), notes=d.get("notes", ""))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_version(files: dict[str, str], classes: dict[str, WorkloadClass]) -> str:
    """One hash over the sorted (path, sha256) pairs AND the class table, so a changed SLO or
    rate moves the version as surely as a changed trace byte does."""
    lines = [f"{p} {h}" for p, h in sorted(files.items())]
    lines += [json.dumps(asdict(c), sort_keys=True) for _, c in sorted(classes.items())]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def write_trace(path: Path, requests: list[TraceRequest]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in requests:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")


def read_trace(path: Path) -> list[TraceRequest]:
    with open(path) as f:
        return [TraceRequest(**json.loads(line)) for line in f if line.strip()]


def build_manifest(classes: dict[str, WorkloadClass], corpus_dir: Path, notes: str = "") -> Manifest:
    """Hash every trace the classes reference and derive the corpus version from the hashes."""
    files = {}
    for c in classes.values():
        for split in SPLITS:
            rel = c.trace_file(split)
            files[rel] = file_sha256(corpus_dir / rel)
    return Manifest(schema_version=CORPUS_SCHEMA_VERSION,
                    corpus_version=corpus_version(files, classes),
                    classes=classes, files=files, notes=notes)


def load_manifest(corpus_dir: Path = CORPUS_DIR) -> Manifest:
    """Read `manifest.json` and verify every trace hash; refuse the corpus on any mismatch."""
    m = Manifest.from_dict(json.loads((corpus_dir / "manifest.json").read_text()))
    if m.schema_version != CORPUS_SCHEMA_VERSION:
        raise CorpusError(f"corpus schema_version {m.schema_version} != {CORPUS_SCHEMA_VERSION}")
    for rel, expected in m.files.items():
        p = corpus_dir / rel
        if not p.exists():
            raise CorpusError(f"trace {rel} listed in the manifest is missing")
        actual = file_sha256(p)
        if actual != expected:
            raise CorpusError(f"trace {rel} sha256 {actual[:12]} != manifest {expected[:12]}: a "
                              f"changed trace is a new corpus version, rebuild the manifest")
    if corpus_version(m.files, m.classes) != m.corpus_version:
        raise CorpusError("manifest corpus_version does not match its own file hashes and "
                          "class table: a changed SLO or rate is a new corpus version too")
    for c in m.classes.values():
        for split in SPLITS:
            if c.trace_file(split) not in m.files:
                raise CorpusError(f"class {c.name} {split} trace is not hashed in the manifest")
    return m


def load_trace(class_name: str, split: str,
               corpus_dir: Path = CORPUS_DIR) -> tuple[Manifest, list[TraceRequest]]:
    """Verified load of one (class, split). Returns the manifest too, for the version stamp."""
    m = load_manifest(corpus_dir)
    if class_name not in m.classes:
        raise CorpusError(f"unknown workload class {class_name!r}; have {sorted(m.classes)}")
    return m, read_trace(corpus_dir / m.classes[class_name].trace_file(split))
