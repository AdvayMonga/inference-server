"""Persisted Triton launch configs, read once at model load. No entry means today's launch.

Pure stdlib (no torch, no triton) so the table logic is testable on a CPU box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KIND = "triton_launch_table"

# (kernel, shape, head_dim, block_size) -> config, for the one GPU the table was filtered to.
_TABLE: dict[tuple[str, int, int, int], dict[str, Any]] = {}


def entry_key(kernel: str, shape: int, head_dim: int, block_size: int) -> tuple[str, int, int, int]:
    return (kernel, int(shape), int(head_dim), int(block_size))


def load(path: str | Path, gpu: str, model: str) -> int:
    """Replace the table with `path`'s entries for this GPU and model. Returns entries kept."""
    doc = json.loads(Path(path).read_text())
    if doc.get("kind") != KIND:
        raise ValueError(f"{path} is not a {KIND} (kind={doc.get('kind')!r})")
    _TABLE.clear()
    if doc.get("model") != model:
        return 0
    for e in doc.get("entries", []):
        if e["gpu"] == gpu:
            _TABLE[entry_key(e["kernel"], e["shape"], e["head_dim"], e["block_size"])] = dict(e["config"])
    return len(_TABLE)


def clear() -> None:
    _TABLE.clear()


def lookup(kernel: str, shape: int, head_dim: int, block_size: int) -> dict[str, Any] | None:
    """The tuned config for this launch, or None to launch exactly as before."""
    if not _TABLE:          # checked first so an empty table never hashes a traced shape
        return None
    return _TABLE.get(entry_key(kernel, shape, head_dim, block_size))


def compile_kwargs(cfg: dict[str, Any] | None) -> dict[str, int]:
    """Triton launch kwargs from a config; {} when there is none."""
    if not cfg:
        return {}
    return {k: int(cfg[k]) for k in ("num_warps", "num_stages") if k in cfg}
