"""Concurrency-sweep load test against a running inference server.

Measures per-request TTFT / TPOT / total latency from the client side, sweeps
concurrency levels, dumps results to CSV. Run the server separately first:

  python -m uvicorn inference_server.server:app --port 8000

Then:
  python scripts/load_test.py --base-url http://127.0.0.1:8000
"""

import argparse
import asyncio
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Workload modes. short/long are uniform; mixed draws from discrete bins per request.
WORKLOADS = {
    "short":  {"prompt_tokens_target": 60,  "max_tokens": 100},
    "long":   {"prompt_tokens_target": 500, "max_tokens": 500},
    "mixed":  {"mixed": True},
}

# Discrete bins for "mixed" — chatbot-like distribution. (weight, prompt, max_tokens).
MIXED_BINS = [
    (0.70,  60, 100),   # short Q&A
    (0.20, 250, 300),   # medium
    (0.10, 800, 600),   # long
]

# A few seed sentences we tile/truncate to hit a target prompt size.
SEED = (
    "Explain how a computer works in simple terms. Walk through the CPU, memory, "
    "disk, and how they coordinate. Use plain language and concrete examples a "
    "high schooler could follow without prior background knowledge whatsoever. "
)


def build_prompt(target_tokens: int) -> str:
    # Roughly 4 chars per token for English — coarse but fine for load shaping.
    target_chars = target_tokens * 4
    n = max(1, (target_chars // len(SEED)) + 1)
    return (SEED * n)[:target_chars]


def sample_mixed(rng: random.Random) -> tuple[str, int]:
    """Draw a (prompt, max_tokens) sample from the mixed-workload bins."""
    r = rng.random()
    cum = 0.0
    for w, p_toks, mx in MIXED_BINS:
        cum += w
        if r <= cum:
            return build_prompt(p_toks), mx
    p_toks, mx = MIXED_BINS[-1][1], MIXED_BINS[-1][2]
    return build_prompt(p_toks), mx


@dataclass
class Sample:
    ttft_s: float
    tpot_s: float
    total_s: float
    tokens: int
    error: str | None = None


@dataclass
class LevelResult:
    workload: str
    concurrency: int
    duration_s: float
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.error is None]
        errs = [s for s in self.samples if s.error is not None]
        if not ok:
            return {
                "workload": self.workload, "N": self.concurrency,
                "n_ok": 0, "n_err": len(errs), "req_per_s": 0,
                "tok_per_s": 0, "ttft_p50": 0, "ttft_p95": 0,
                "tpot_p50": 0, "tpot_p95": 0, "total_p50": 0, "total_p95": 0,
            }
        ttfts = sorted(s.ttft_s for s in ok)
        tpots = sorted(s.tpot_s for s in ok if s.tpot_s > 0)
        totals = sorted(s.total_s for s in ok)
        n_tokens = sum(s.tokens for s in ok)
        return {
            "workload": self.workload,
            "N": self.concurrency,
            "n_ok": len(ok),
            "n_err": len(errs),
            "req_per_s": round(len(ok) / self.duration_s, 3),
            "tok_per_s": round(n_tokens / self.duration_s, 2),
            "ttft_p50": round(_pct(ttfts, 0.50) * 1000, 1),
            "ttft_p95": round(_pct(ttfts, 0.95) * 1000, 1),
            "tpot_p50": round(_pct(tpots, 0.50) * 1000, 2) if tpots else 0,
            "tpot_p95": round(_pct(tpots, 0.95) * 1000, 2) if tpots else 0,
            "total_p50": round(_pct(totals, 0.50) * 1000, 1),
            "total_p95": round(_pct(totals, 0.95) * 1000, 1),
        }


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def run_one_request(
    client: httpx.AsyncClient, base_url: str, prompt: str, max_tokens: int,
) -> Sample:
    """One streaming request. Times TTFT from request start to first token."""
    payload = {"text": prompt, "max_tokens": max_tokens, "stream": True, "thinking": False}
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    try:
        async with client.stream(
            "POST", f"{base_url}/generate",
            json=payload, timeout=httpx.Timeout(120.0),
        ) as r:
            if r.status_code != 200:
                body = await r.aread()
                return Sample(0, 0, 0, 0, error=f"http {r.status_code}: {body[:80]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    break
                # First framed chunk is JSON metadata; subsequent are tokens.
                if d.startswith("{") and "ttft_ms" in d:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                tokens += 1
    except Exception as e:
        return Sample(0, 0, 0, 0, error=type(e).__name__ + ": " + str(e)[:80])
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    tpot = (total - ttft) / (tokens - 1) if tokens > 1 else 0.0
    return Sample(ttft, tpot, total, tokens)


async def worker(
    client: httpx.AsyncClient, base_url: str, deadline: float,
    samples: list[Sample],
    fixed: tuple[str, int] | None,
    rng: random.Random | None,
) -> None:
    """Loop firing requests serially until the deadline.

    If `fixed` is set, every request uses the same (prompt, max_tokens).
    Otherwise draw from MIXED_BINS via `rng` per request.
    """
    while time.perf_counter() < deadline:
        if fixed is not None:
            prompt, mx = fixed
        else:
            prompt, mx = sample_mixed(rng)
        s = await run_one_request(client, base_url, prompt, mx)
        samples.append(s)


async def run_level(
    base_url: str, workload: str, concurrency: int, duration_s: float,
) -> LevelResult:
    cfg = WORKLOADS[workload]
    deadline = time.perf_counter() + duration_s
    result = LevelResult(workload=workload, concurrency=concurrency, duration_s=duration_s)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        if cfg.get("mixed"):
            fixed = None
            rngs = [random.Random(1000 + i) for i in range(concurrency)]
        else:
            fixed = (build_prompt(cfg["prompt_tokens_target"]), cfg["max_tokens"])
            rngs = [None] * concurrency
        workers = [
            asyncio.create_task(worker(client, base_url, deadline, result.samples, fixed, rngs[i]))
            for i in range(concurrency)
        ]
        await asyncio.gather(*workers)
    return result


async def fetch_server_stats(base_url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{base_url}/scheduler/stats")
            return r.json()
    except Exception:
        return {}


def print_row(s: dict) -> None:
    print(
        f"  {s['workload']:5s} N={s['N']:>3d} | "
        f"ok={s['n_ok']:>4d} err={s['n_err']:>2d} | "
        f"{s['req_per_s']:>6.2f} req/s {s['tok_per_s']:>7.1f} tok/s | "
        f"TTFT p50={s['ttft_p50']:>6.0f} p95={s['ttft_p95']:>6.0f} ms | "
        f"TPOT p50={s['tpot_p50']:>5.1f} p95={s['tpot_p95']:>5.1f} ms"
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--workload", default="short", choices=list(WORKLOADS.keys()))
    ap.add_argument("--levels", default="1,2,4,8,16",
                    help="Comma-separated concurrency levels")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Seconds per concurrency level")
    ap.add_argument("--output", default=None,
                    help="CSV output path; default benchmarks/load_<ts>.csv")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    out_path = Path(args.output) if args.output else (
        out_dir / f"load_{args.workload}_{int(time.time())}.csv"
    )

    # Sanity-check server is reachable first.
    s0 = await fetch_server_stats(args.base_url)
    if not s0:
        print(f"ERROR: cannot reach server at {args.base_url}/scheduler/stats")
        return
    print(f"Server OK. policy={s0.get('policy')} batch={s0.get('max_batch_size')} "
          f"kv_blocks={s0.get('kv_free_blocks')}")
    print(f"Workload: {args.workload}  duration/level: {args.duration}s  levels: {levels}")
    print()

    rows: list[dict] = []
    for N in levels:
        print(f"-- ramping to N={N} --")
        res = await run_level(args.base_url, args.workload, N, args.duration)
        # Snapshot server-side stats at end of level for cross-validation.
        srv = await fetch_server_stats(args.base_url)
        s = res.summary()
        s["srv_pending"] = srv.get("pending_depth", 0)
        s["srv_active"] = srv.get("active_size", 0)
        s["srv_kv_pressure"] = round(srv.get("kv_pressure", 0.0), 3)
        rows.append(s)
        print_row(s)
        # Brief cooldown so percentiles between levels don't bleed into each other
        # on the server's 60s window.
        await asyncio.sleep(2)

    # Write CSV
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")
    print("\nFinal table:")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
