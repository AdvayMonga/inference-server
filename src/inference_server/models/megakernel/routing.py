"""Which decode path a step takes, from a measured table (MEGAKERNEL_ROUTING, default routing.json beside this file):

    {"context_buckets": [1024, 4096, 16384],                 # longest row's context, upper bounds, ascending
     "rows": {"1": ["megakernel", "megakernel", "kernels"],  # per row count: a path per bucket
              "2": ["megakernel", "kernels", "kernels"]}}

"megakernel" is the one-launch decode step, "framework" the same step through the instruction interpreter
(framework.py, built only when the table names it), "kernels" the separate kernels (graphs or eager). A step whose row
count or context the table does not cover takes the MEGAKERNEL_MAX_ROWS rule. No file: the rule everywhere."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

PATHS = ("megakernel", "framework", "kernels")
DEFAULT = Path(__file__).with_name("routing.json")


@dataclass(frozen=True)
class Routing:
    context_buckets: tuple[int, ...]
    rows: dict[int, tuple[str, ...]]

    def path(self, n_rows: int, context: int) -> str | None:
        """The table's path for `n_rows` rows whose longest context is `context` tokens, or None outside it."""
        paths = self.rows.get(n_rows)
        i = bisect.bisect_left(self.context_buckets, context)
        return paths[i] if paths is not None and i < len(paths) else None

    def uses(self, path: str) -> bool:
        return any(path in paths for paths in self.rows.values())


def load(path: str | Path, max_rows: int) -> Routing | None:
    """The table at `path` (None if there is none), checked: buckets ascending, a known path per bucket, and the
    megakernel only at row counts it handles (`max_rows`)."""
    path = Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    buckets = tuple(int(b) for b in raw["context_buckets"])
    if list(buckets) != sorted(set(buckets)):
        raise ValueError(f"{path}: context_buckets must ascend: {buckets}")
    rows = {int(n): tuple(p) for n, p in raw["rows"].items()}
    for n, paths in rows.items():
        if len(paths) != len(buckets) or not set(paths) <= set(PATHS):
            raise ValueError(f"{path}: rows[{n}] must give one of {PATHS} per bucket: {paths}")
        if {"megakernel", "framework"} & set(paths) and not 1 <= n <= max_rows:
            raise ValueError(f"{path}: the megakernel handles 1-{max_rows} rows, not {n}")
    return Routing(buckets, rows)
