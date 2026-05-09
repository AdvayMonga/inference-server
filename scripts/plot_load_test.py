"""Plot load-test CSVs.

Usage:
  python scripts/plot_load_test.py benchmarks/load_short_*.csv [more.csv ...]

Produces three PNGs next to the first CSV: throughput, p95 TTFT, p95 TPOT vs N.
Each input CSV becomes one line on each plot, labeled by its `workload` column.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> tuple[str, list[dict]]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    label = rows[0]["workload"] if rows else path.stem
    return label, rows


def num(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def plot_metric(series: list[tuple[str, list[dict]]], y_field: str, ylabel: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rows in series:
        xs = [int(r["N"]) for r in rows]
        ys = [num(r[y_field]) for r in rows]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Concurrent users (N)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    series = [load(p) for p in paths]
    out_dir = paths[0].parent

    plot_metric(series, "tok_per_s", "Throughput (tok/s)", out_dir / "throughput.png")
    plot_metric(series, "ttft_p95", "p95 TTFT (ms)", out_dir / "ttft_p95.png")
    plot_metric(series, "tpot_p95", "p95 TPOT (ms)", out_dir / "tpot_p95.png")
    plot_metric(series, "total_p95", "p95 Total latency (ms)", out_dir / "total_p95.png")


if __name__ == "__main__":
    main()
