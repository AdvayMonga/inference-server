"""Open-loop replay of one corpus trace against a running server.

    PYTHONPATH=src python scripts/bench/replay_trace.py --class steady_interactive --split seen \\
        --base-url http://127.0.0.1:8000 [--rate-scale 1.0]

Every request fires at its trace `arrival_s` (divided by --rate-scale) on THIS process's clock,
whatever the server is doing — a slow server gets the same arrivals as a fast one, so it cannot
look better by receiving fewer requests. That is the property the closed-loop harness lacked for
a month. The client measures TTFT/TPOT itself; the engine's own stats are read afterwards for
the panel's pressure/cache fields only, never as evidence.

Emits `runs/<run_id>.json` (a Vitals panel stamped with the corpus version and class) plus
`runs/<run_id>.csv` with one row per request, which is what the loop re-slices later.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_serving import _pct, one_request  # noqa: E402
from inference_server.research import harness as H  # noqa: E402
from inference_server.research.corpus import (  # noqa: E402
    Manifest,
    TraceRequest,
    WorkloadClass,
    load_trace,
)
from inference_server.research.schemas import Vitals  # noqa: E402


@dataclass
class Row:
    """One replayed request: when it was meant to fire, when it did, what the client saw."""

    index: int
    session_id: str
    turn_index: int
    arrival_s: float
    fired_s: float
    ttft_ms: float | None
    tpot_ms: float | None
    out_tokens: int
    error: str | None


@dataclass
class ReplayResult:
    rows: list[Row] = field(default_factory=list)
    wall_s: float = 0.0          # first arrival -> last completion, drain included

    def summary(self, cls: WorkloadClass) -> dict[str, Any]:
        ok = [r for r in self.rows if r.error is None]
        ttfts = sorted(r.ttft_ms for r in ok)
        tpots = sorted(r.tpot_ms for r in ok if r.tpot_ms)
        window = self.wall_s or 1e-9
        p95_ttft = _pct(ttfts, 0.95)
        p95_tpot = _pct(tpots, 0.95) if tpots else None
        tok_s = sum(r.out_tokens for r in ok) / window
        return {
            "n_ok": len(ok), "n_err": len(self.rows) - len(ok),
            "achieved_rps": round(len(ok) / window, 2), "tok_per_s": round(tok_s, 1),
            "ttft_p50": round(_pct(ttfts, 0.50), 1), "ttft_p95": round(p95_ttft, 1),
            "tpot_p50": round(_pct(tpots, 0.50), 2) if tpots else None,
            "tpot_p95": round(p95_tpot, 2) if p95_tpot is not None else None,
            "within_slo": bool(ok) and cls.within_slo(p95_ttft, p95_tpot),
            "wall_s": round(self.wall_s, 2),
        }


async def run_replay(client: httpx.AsyncClient, trace: list[TraceRequest], *,
                     rate_scale: float = 1.0, drain_timeout_s: float = 300.0) -> ReplayResult:
    """Fire each request at arrival_s / rate_scale on our clock, then drain what is in flight."""
    res = ReplayResult()
    tasks: list[tuple[asyncio.Task, int, TraceRequest]] = []
    fired_at: dict[int, float] = {}
    t0 = time.perf_counter()

    async def fire(i: int, req: TraceRequest) -> None:
        fired = fired_at[i] = time.perf_counter() - t0
        # Sampling: one_request posts temperature 0 and the shim's default top_p/top_k, which
        # is what every trace carries; req.sampling is not yet threaded through.
        s = await one_request(client, req.prompt, req.max_tokens)
        res.rows.append(Row(i, req.session_id, req.turn_index, req.arrival_s, round(fired, 4),
                            round(s.ttft_s * 1000, 2) if s.error is None else None,
                            round(s.tpot_s * 1000, 3) if s.error is None else None,
                            s.out_tokens, s.error))

    for i, req in enumerate(trace):
        delay = t0 + req.arrival_s / rate_scale - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)          # the schedule, never a completion, gates this
        tasks.append((asyncio.create_task(fire(i, req)), i, req))
    if tasks:
        await asyncio.wait([t for t, _, _ in tasks], timeout=drain_timeout_s)
    # A request still open when the drain ends is a failure, not a request that never
    # happened: dropping it would inflate n_ok exactly under the overload this measures.
    late = [(t, i, req) for t, i, req in tasks if not t.done()]
    for t, i, req in late:
        t.cancel()
        res.rows.append(Row(i, req.session_id, req.turn_index, req.arrival_s,
                            round(fired_at[i], 4), None, None, 0, "drain_timeout"))
    if late:
        await asyncio.gather(*(t for t, _, _ in late), return_exceptions=True)
    res.wall_s = time.perf_counter() - t0
    res.rows.sort(key=lambda r: r.index)
    return res


async def fetch_stats(client: httpx.AsyncClient, path: str) -> dict[str, Any]:
    try:
        r = await client.get(path, timeout=5.0)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def build_panel(res: ReplayResult, cls: WorkloadClass, *, corpus_version: str, split: str,
                rate_scale: float, scheduler_stats: dict[str, Any],
                cache_stats: dict[str, Any]) -> Vitals:
    s = res.summary(cls)
    ttfts = [r.ttft_ms for r in res.rows if r.error is None]
    validity = H.build_validity(
        "replay_trace",
        {"workload_class": cls.name, "split": split, "corpus_version": corpus_version,
         "rate_scale": rate_scale, "arrival_rate_rps": cls.arrival_rate_rps * rate_scale,
         "n_requests": len(res.rows)},
        n_samples=s["n_ok"],
        workload_regime=H.infer_regime(cache_stats.get("hit_rate")),
        stderr_value=H.stderr(ttfts),
        concurrency_observed=scheduler_stats.get("active_high_water"),
        notes=f"open-loop trace replay, trial={H.trial_id()}",
    )
    validity.corpus_version = corpus_version
    validity.workload_class = cls.name
    return H.panel_from_stats(
        validity, scheduler_stats=scheduler_stats, cache_stats=cache_stats,
        tok_s_within_slo=s["tok_per_s"] if s["within_slo"] else None,
        slo_ttft_ms=cls.slo_ttft_ms, slo_tpot_ms=cls.slo_tpot_ms,
        ttft_p50=s["ttft_p50"], ttft_p95=s["ttft_p95"],
        tpot_p50=s["tpot_p50"], tpot_p95=s["tpot_p95"], wall_s=s["wall_s"],
    )


def write_rows(path: Path, rows: list[Row]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in Row.__dataclass_fields__.values()])
        w.writeheader()
        w.writerows(asdict(r) for r in rows)


def print_summary(cls: WorkloadClass, s: dict[str, Any]) -> None:
    tpot_slo = f", p95 TPOT < {cls.slo_tpot_ms:.0f}ms" if cls.slo_tpot_ms else ""
    print(f"\n  {cls.name}: SLO p95 TTFT < {cls.slo_ttft_ms:.0f}ms{tpot_slo}")
    print(f"  ok={s['n_ok']} err={s['n_err']} {s['achieved_rps']} req/s {s['tok_per_s']} tok/s | "
          f"TTFT p50/p95 {s['ttft_p50']}/{s['ttft_p95']} ms | "
          f"TPOT p50/p95 {s['tpot_p50']}/{s['tpot_p95']} ms | "
          f"{'within SLO' if s['within_slo'] else 'SLO BROKEN'} | wall {s['wall_s']}s")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", required=True)
    ap.add_argument("--split", default="seen", choices=("seen", "heldout"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rate-scale", type=float, default=1.0,
                    help="divide every arrival offset by this; 2.0 replays twice as fast")
    ap.add_argument("--drain-timeout-s", type=float, default=300.0,
                    help="after the last arrival, wait this long; still-open requests count as "
                         "errors (drain_timeout)")
    ap.add_argument("--runs-dir", default=None)
    args = ap.parse_args()

    manifest: Manifest
    manifest, trace = load_trace(args.cls, args.split)
    cls = manifest.classes[args.cls]
    print(f"-- replay {args.cls}/{args.split} ({len(trace)} requests, corpus "
          f"{manifest.corpus_version[:12]}, rate x{args.rate_scale}) against {args.base_url} --")
    async with httpx.AsyncClient(base_url=args.base_url,
                                 limits=httpx.Limits(max_connections=1024)) as client:
        res = await run_replay(client, trace, rate_scale=args.rate_scale,
                               drain_timeout_s=args.drain_timeout_s)
        sched = await fetch_stats(client, "/scheduler/stats")
        cache = await fetch_stats(client, "/cache/stats")

    s = res.summary(cls)
    print_summary(cls, s)
    if not s["n_ok"]:
        print("  no request succeeded; nothing to record")
        return 1
    panel = build_panel(res, cls, corpus_version=manifest.corpus_version, split=args.split,
                        rate_scale=args.rate_scale, scheduler_stats=sched, cache_stats=cache)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    path = H.emit(panel, label=f"replay {args.cls}/{args.split}", runs_dir=runs_dir)
    write_rows(path.with_suffix(".csv"), res.rows)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
