"""Concurrency-sweep load test against a running inference server.

Measures per-request TTFT / TPOT / total latency from the client side, sweeps
concurrency levels, dumps results to CSV. Run the server separately first:

  python -m uvicorn inference_server.server:app --port 8000

Then:
  python scripts/bench/load_test.py --base-url http://127.0.0.1:8000

Prompts are unique per request by default, so a sweep exercises the PrefixCache miss
path. Pass --prefix-share full to measure the hit path instead. Whichever you pick is
written into the CSV, because a latency number without its cache regime is not evidence.
"""

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import prompt_bank  # noqa: E402  sibling module, not an installed package

# Workload modes. short/long are uniform; mixed draws from discrete bins; realistic
# draws distinct English prompts from prompt_bank.
WORKLOADS = {
    "short":     {"prompt_tokens_target": 60,  "max_tokens": 100},
    "long":      {"prompt_tokens_target": 500, "max_tokens": 500},
    "mixed":     {"mixed": True},
    "realistic": {"realistic": True},
}

# Discrete bins for "mixed" — chatbot-like distribution. (weight, prompt, max_tokens).
MIXED_BINS = [
    (0.70,  60, 100),   # short Q&A
    (0.20, 250, 300),   # medium
    (0.10, 800, 600),   # long
]

ERROR_BACKOFF_S = 0.25   # pause after a failed/rejected request; without it a 429 storm
                         # becomes a tight loop that counts thousands of errors per second

# A few seed sentences we tile/truncate to hit a target prompt size.
SEED = (
    "Explain how a computer works in simple terms. Walk through the CPU, memory, "
    "disk, and how they coordinate. Use plain language and concrete examples a "
    "high schooler could follow without prior background knowledge whatsoever. "
)


def build_prompt(target_tokens: int, rng: random.Random | None = None) -> str:
    """Tile SEED to a target size.

    With `rng`, a unique preamble goes first, so no two prompts share even their first
    block and the PrefixCache cannot match them. Without it every prompt is identical
    from the front and the whole sweep measures the cache-hit path.
    """
    # Roughly 4 chars per token for English — coarse but fine for load shaping.
    target_chars = target_tokens * 4
    head = "" if rng is None else f"Case {rng.getrandbits(48):012x}. "
    body_chars = max(0, target_chars - len(head))
    n = max(1, (body_chars // len(SEED)) + 1)
    return head + (SEED * n)[:body_chars]


def sample_mixed(rng: random.Random, unique: bool = True) -> tuple[str, int]:
    """Draw a (prompt, max_tokens) sample from the mixed-workload bins."""
    r = rng.random()
    cum = 0.0
    p_toks, mx = MIXED_BINS[-1][1], MIXED_BINS[-1][2]
    for w, bin_toks, bin_mx in MIXED_BINS:
        cum += w
        if r <= cum:
            p_toks, mx = bin_toks, bin_mx
            break
    return build_prompt(p_toks, rng if unique else None), mx


def sample_realistic(rng: random.Random) -> tuple[str, int]:
    """Draw a (prompt, max_tokens) sample from the prompt bank's weighted buckets.

    Prompts are distinct English text, but the bank is finite, so repeats do occur and
    do hit the cache. That is what real traffic looks like; --prefix-share does not
    apply here.
    """
    r, cum = rng.random(), 0.0
    name, prompts, _ = prompt_bank.PROMPT_MIX[-1]
    for bucket, bucket_prompts, weight in prompt_bank.PROMPT_MIX:
        cum += weight
        if r <= cum:
            name, prompts = bucket, bucket_prompts
            break
    lo, hi = prompt_bank.MAX_TOKENS_RANGE[name]
    return rng.choice(prompts), rng.randint(lo, hi)


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
    wall_s: float = 0.0   # start -> last completion; the window rates are measured over
    prefix_share: str = "none"
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.error is None]
        errs = [s for s in self.samples if s.error is not None]
        # Workers stop STARTING requests at the deadline but finish the one in flight, so
        # completions land up to one request-latency after it. Dividing by the nominal
        # duration counted that tail as if it happened inside the window and overstated
        # throughput most exactly where it matters — long requests near saturation.
        window = self.wall_s or self.duration_s
        if not ok:
            return {
                "workload": self.workload, "prefix_share": self.prefix_share,
                "N": self.concurrency,
                "n_ok": 0, "n_err": len(errs), "req_per_s": 0,
                "tok_per_s": 0, "ttft_p50": 0, "ttft_p95": 0,
                "tpot_p50": 0, "tpot_p95": 0, "total_p50": 0, "total_p95": 0,
                "wall_s": round(window, 2),
            }
        ttfts = sorted(s.ttft_s for s in ok)
        tpots = sorted(s.tpot_s for s in ok if s.tpot_s > 0)
        totals = sorted(s.total_s for s in ok)
        n_tokens = sum(s.tokens for s in ok)
        return {
            "workload": self.workload,
            "prefix_share": self.prefix_share,
            "N": self.concurrency,
            "n_ok": len(ok),
            "n_err": len(errs),
            "req_per_s": round(len(ok) / window, 3),
            "tok_per_s": round(n_tokens / window, 2),
            "ttft_p50": round(_pct(ttfts, 0.50) * 1000, 1),
            "ttft_p95": round(_pct(ttfts, 0.95) * 1000, 1),
            "tpot_p50": round(_pct(tpots, 0.50) * 1000, 2) if tpots else 0,
            "tpot_p95": round(_pct(tpots, 0.95) * 1000, 2) if tpots else 0,
            "total_p50": round(_pct(totals, 0.50) * 1000, 1),
            "total_p95": round(_pct(totals, 0.95) * 1000, 1),
            "wall_s": round(window, 2),
        }


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def run_one_request(
    client: httpx.AsyncClient, base_url: str, prompt: str, max_tokens: int,
    session_id: str = "default",
) -> Sample:
    """One streaming request. Times TTFT from request start to first token."""
    payload = {"text": prompt, "max_tokens": max_tokens, "stream": True, "thinking": False,
               "session_id": session_id}
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
    draw,
    session_id: str = "default",
) -> None:
    """Loop firing serial requests until the deadline — one closed-loop user.

    `draw()` yields one (prompt, max_tokens) per request. Each worker is its own session so
    per-session fairness and admission actually see N users, not one.
    """
    while time.perf_counter() < deadline:
        prompt, mx = draw()
        s = await run_one_request(client, base_url, prompt, mx, session_id)
        samples.append(s)
        if s.error is not None:
            await asyncio.sleep(ERROR_BACKOFF_S)   # a rejected user does not resubmit instantly


def make_draw(cfg: dict, rng: random.Random, prefix_share: str):
    """Zero-arg sampler for one worker.

    prefix_share="full" pins a single prompt for the whole run, so every request after
    the first is a PrefixCache hit. "none" gives each request a unique first block.
    """
    if cfg.get("realistic"):
        return partial(sample_realistic, rng)
    if cfg.get("mixed"):
        return partial(sample_mixed, rng, prefix_share != "full")
    target, mx = cfg["prompt_tokens_target"], cfg["max_tokens"]
    if prefix_share == "full":
        pinned = (build_prompt(target), mx)
        return lambda: pinned
    return lambda: (build_prompt(target, rng), mx)


async def run_level(
    base_url: str, workload: str, concurrency: int, duration_s: float,
    prefix_share: str = "none",
) -> LevelResult:
    cfg = WORKLOADS[workload]
    t_start = time.perf_counter()
    deadline = t_start + duration_s
    result = LevelResult(workload=workload, concurrency=concurrency, duration_s=duration_s,
                         prefix_share=prefix_share)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        workers = [
            asyncio.create_task(worker(
                client, base_url, deadline, result.samples,
                make_draw(cfg, random.Random(1000 + i), prefix_share),
                session_id=f"load-{i}"))
            for i in range(concurrency)
        ]
        await asyncio.gather(*workers)
    result.wall_s = time.perf_counter() - t_start
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
        f"  {s['workload']:9s} share={s['prefix_share']:4s} N={s['N']:>3d} | "
        f"ok={s['n_ok']:>4d} err={s['n_err']:>2d} | "
        f"{s['req_per_s']:>6.2f} req/s {s['tok_per_s']:>7.1f} tok/s | "
        f"TTFT p50={s['ttft_p50']:>6.0f} p95={s['ttft_p95']:>6.0f} ms | "
        f"TPOT p50={s['tpot_p50']:>5.1f} p95={s['tpot_p95']:>5.1f} ms"
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--workload", default="realistic", choices=list(WORKLOADS.keys()))
    ap.add_argument("--prefix-share", default="none", choices=["none", "full"],
                    help="none: unique prompt prefixes, measures the PrefixCache miss path "
                         "(default). full: one pinned prompt, measures the hit path. Ignored "
                         "by the realistic workload, which draws from a finite prompt bank.")
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
        out_dir / f"load_{args.workload}_share-{args.prefix_share}_{int(time.time())}.csv"
    )

    # Sanity-check server is reachable first.
    s0 = await fetch_server_stats(args.base_url)
    if not s0:
        print(f"ERROR: cannot reach server at {args.base_url}/scheduler/stats")
        return
    print(f"Server OK. policy={s0.get('policy')} batch={s0.get('max_batch_size')} "
          f"kv_blocks={s0.get('kv_free_blocks')}")
    print(f"Workload: {args.workload}  prefix-share: {args.prefix_share}  "
          f"duration/level: {args.duration}s  levels: {levels}")
    print()

    rows: list[dict] = []
    for N in levels:
        print(f"-- ramping to N={N} --")
        res = await run_level(args.base_url, args.workload, N, args.duration, args.prefix_share)
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
