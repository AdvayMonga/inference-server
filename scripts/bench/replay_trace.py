"""Open-loop replay of one corpus trace against a running server.

    PYTHONPATH=src python scripts/bench/replay_trace.py --class steady_interactive --split seen \\
        --base-url http://127.0.0.1:8000 [--rate-scale 1.0] [--prompt-format chat|raw]

Every request fires at its trace `arrival_s` (divided by --rate-scale) on THIS process's clock,
whatever the server is doing — a slow server gets the same arrivals as a fast one, so it cannot
look better by receiving fewer requests. That is the property the closed-loop harness lacked for
a month. The client measures TTFT/TPOT itself; the engine's own stats are read afterwards for
the panel's pressure/cache fields only, never as evidence.

Each request carries the trace's `session_id` / `turn_index` as X-Session-Id / X-Turn-Index, so
a second turn lands on the same engine session as the first, and X-Trace-Id=<prefix>-<i> where
<prefix> is `<class>-<split>-<nonce>`, the nonce minted once per invocation. That id is the key a
telemetry SQLite row and a replay CSV row share; the nonce keeps two replays against one server
process (one telemetry file) from colliding, and the prefix is recorded in the panel's
harness_config so the two can be joined by prefix.

Prompts are posted to `/v1/chat/completions` by default, so the model's own chat template is
applied server-side; `--prompt-format raw` restores the old `/v1/completions` path. The trace
stores plain prompt text either way — templating a trace at build time would tie the corpus to
one model's template, and Phase 0 is still choosing the model. What that costs is an invisible
variable (which template, with which options), so a chat-route panel carries a `chat_template`
fingerprint in its validity block, verified against the prompt-token count the server reported,
and compare.py refuses across a changed one.

Emits `runs/<run_id>.json` (a Vitals panel stamped with the corpus version and class) plus
`runs/<run_id>.csv` with one row per request, which is what the loop re-slices later.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_serving import _pct, one_request  # noqa: E402
from inference_server.research import chat_template as CT  # noqa: E402
from inference_server.research import harness as H  # noqa: E402
from inference_server.research.corpus import (  # noqa: E402
    Manifest,
    TraceRequest,
    WorkloadClass,
    load_trace,
)
from inference_server.research.schemas import Vitals  # noqa: E402

# Corpus prompts go through the CHAT route by default. /v1/completions is spec-correct in not
# templating, but gemma-4-E2B-it answers a bare un-templated prompt with an immediate end-of-turn:
# cold_start/seen generated 220 tokens of a 626-token budget and six of eight cold_start/heldout
# prompts generated nothing at all (kb-20260919-6ef4e6bf). Templating is done SERVER-side rather
# than baked into the trace so the corpus stays model-independent — Phase 0 may swap the model.
# The price is that the tokens the model sees now depend on the tokenizer's bundled template, so
# every chat-route panel carries a verified `chat_template` fingerprint in its validity block.
DEFAULT_PROMPT_FORMAT = "chat"


@dataclass
class Row:
    """One replayed request: when it was meant to fire, when it did, what the client saw."""

    index: int
    session_id: str
    turn_index: int
    trace_id: str
    arrival_s: float
    fired_s: float
    ttft_ms: float | None
    tpot_ms: float | None
    out_tokens: int
    error: str | None
    prompt_tokens: int | None = None    # what the server encoded; templated on the chat route
    # The engine's own completion count, from the usage chunk. Higher than `out_tokens` whenever
    # a generated token decodes to "" — the shim emits no chunk for those, so the client cannot
    # see them. `out_tokens` stays the client's count because that is what TTFT/TPOT measured.
    server_out_tokens: int | None = None


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
                     rate_scale: float = 1.0, drain_timeout_s: float = 300.0,
                     trace_prefix: str = "replay",
                     prompt_format: str = DEFAULT_PROMPT_FORMAT) -> ReplayResult:
    """Fire each request at arrival_s / rate_scale on our clock, then drain what is in flight.
    Request i is sent as X-Trace-Id=<trace_prefix>-<i> on the trace's session/turn.

    `prompt_format` picks the surface each prompt is posted to — see DEFAULT_PROMPT_FORMAT."""
    res = ReplayResult()
    tasks: list[tuple[asyncio.Task, int, TraceRequest]] = []
    fired_at: dict[int, float] = {}
    t0 = time.perf_counter()

    def trace_id(i: int) -> str:
        return f"{trace_prefix}-{i}"

    async def fire(i: int, req: TraceRequest) -> None:
        fired = fired_at[i] = time.perf_counter() - t0
        headers = {"X-Session-Id": req.session_id, "X-Turn-Index": str(req.turn_index),
                   "X-Trace-Id": trace_id(i)}
        s = await one_request(client, req.prompt, req.max_tokens, headers=headers,
                              sampling=req.sampling, prompt_format=prompt_format)
        res.rows.append(Row(i, req.session_id, req.turn_index, trace_id(i), req.arrival_s,
                            round(fired, 4),
                            round(s.ttft_s * 1000, 2) if s.error is None else None,
                            round(s.tpot_s * 1000, 3) if s.error is None else None,
                            s.out_tokens, s.error, s.prompt_tokens, s.server_out_tokens))

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
        res.rows.append(Row(i, req.session_id, req.turn_index, trace_id(i), req.arrival_s,
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


def new_trace_prefix(cls_name: str, split: str) -> str:
    """`<class>-<split>-<nonce>`: unique per invocation so two replays never share a trace id."""
    return f"{cls_name}-{split}-{uuid.uuid4().hex[:6]}"


def template_stamp(prompt_format: str, model_name: str | None,
                   trace: list[TraceRequest] | None, rows: list[Row] | None) -> dict | None:
    """The chat_template fingerprint for this replay, verified against the server that served it.

    None on the raw route (no template is applied) and whenever the tokenizer cannot be read —
    an unknown stamp is an ordinary outcome, a WRONG one is what compare.py must refuse.
    """
    if prompt_format != "chat" or not model_name:
        return None
    fp = CT.fingerprint(model_name)
    if fp is None or not trace or not rows:
        return fp
    by_index = {r.index: r for r in rows}
    for i, req in enumerate(trace):
        row = by_index.get(i)
        if row is not None and row.prompt_tokens is not None:
            return CT.verify(fp, req.prompt, row.prompt_tokens)
    return fp


def build_panel(res: ReplayResult, cls: WorkloadClass, *, corpus_version: str, split: str,
                rate_scale: float, scheduler_stats: dict[str, Any],
                cache_stats: dict[str, Any], trace_prefix: str = "replay",
                prompt_format: str = DEFAULT_PROMPT_FORMAT, model_name: str | None = None,
                trace: list[TraceRequest] | None = None) -> Vitals:
    s = res.summary(cls)
    ttfts = [r.ttft_ms for r in res.rows if r.error is None]
    validity = H.build_validity(
        "replay_trace",
        {"workload_class": cls.name, "split": split, "corpus_version": corpus_version,
         "rate_scale": rate_scale, "arrival_rate_rps": cls.arrival_rate_rps * rate_scale,
         "n_requests": len(res.rows), "trace_prefix": trace_prefix,
         # compare.py refuses across this: the trace bytes are identical on both routes, so
         # corpus_version cannot tell a templated replay from an untemplated one.
         "prompt_format": prompt_format},
        n_samples=s["n_ok"],
        workload_regime=H.infer_regime(cache_stats.get("hit_rate")),
        stderr_value=H.stderr(ttfts),
        concurrency_observed=scheduler_stats.get("active_high_water"),
        notes=f"open-loop trace replay, trial={H.trial_id()}",
        chat_template=template_stamp(prompt_format, model_name, trace, res.rows),
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
    ap.add_argument("--prompt-format", default=DEFAULT_PROMPT_FORMAT, choices=("chat", "raw"),
                    help="chat: post through /v1/chat/completions so the model's chat template "
                         "is applied (default). raw: /v1/completions, the trace's bytes as-is — "
                         "an instruct model mostly answers those with nothing")
    args = ap.parse_args()

    manifest: Manifest
    manifest, trace = load_trace(args.cls, args.split)
    cls = manifest.classes[args.cls]
    trace_prefix = new_trace_prefix(args.cls, args.split)
    print(f"-- replay {args.cls}/{args.split} ({len(trace)} requests, corpus "
          f"{manifest.corpus_version[:12]}, rate x{args.rate_scale}, {args.prompt_format} route) "
          f"against {args.base_url}, trace ids {trace_prefix}-<i> --")
    async with httpx.AsyncClient(base_url=args.base_url,
                                 limits=httpx.Limits(max_connections=1024)) as client:
        models = await fetch_stats(client, "/v1/models")
        res = await run_replay(client, trace, rate_scale=args.rate_scale,
                               drain_timeout_s=args.drain_timeout_s,
                               trace_prefix=trace_prefix,
                               prompt_format=args.prompt_format)
        sched = await fetch_stats(client, "/scheduler/stats")
        cache = await fetch_stats(client, "/cache/stats")
    model_name = (models.get("data") or [{}])[0].get("id")

    s = res.summary(cls)
    print_summary(cls, s)
    if not s["n_ok"]:
        print("  no request succeeded; nothing to record")
        return 1
    panel = build_panel(res, cls, corpus_version=manifest.corpus_version, split=args.split,
                        rate_scale=args.rate_scale, scheduler_stats=sched, cache_stats=cache,
                        trace_prefix=trace_prefix, prompt_format=args.prompt_format,
                        model_name=model_name, trace=trace)
    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    path = H.emit(panel, label=f"replay {args.cls}/{args.split}", runs_dir=runs_dir)
    write_rows(path.with_suffix(".csv"), res.rows)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
