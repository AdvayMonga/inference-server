"""Open-loop serving benchmark for /v1/completions — Poisson arrivals, realistic length
distribution, SLO-gated throughput.

The vLLM-style methodology our closed-loop load_test.py can't do: load is an INPUT (arrival
rate lambda), not an output (fixed concurrency). Requests arrive on a Poisson schedule
regardless of server state, so the queue can actually grow — which is how backpressure,
tail latency, and the real saturation point get exercised. We sweep lambda past saturation
and report the max throughput that still holds the SLO.

SLO (PLAN.md P1): p95 TTFT < 200 ms, p95 TPOT < 50 ms/token.

Authoritative vLLM head-to-head (P4) drives the SAME shim with guidellm:
    guidellm benchmark --target $URL --rate-type poisson --rate 2,4,8,16 \
        --data "prompt_tokens=240,output_tokens=150" --backend openai_http
This built-in driver is the fast dev-loop; guidellm is the published number.
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

SLO_TTFT_MS = 200.0   # PLAN.md P1
SLO_TPOT_MS = 50.0

# ShareGPT-ish length distribution (lognormal). Medians ~240 prompt / ~150 output, long-tailed.
# Override with a real ShareGPT jsonl via --data once we care about the published distribution.
PROMPT_MU, PROMPT_SIGMA = 5.48, 0.75   # exp(5.48) ~ 240 tokens
OUTPUT_MU, OUTPUT_SIGMA = 5.01, 0.65   # exp(5.01) ~ 150 tokens

SEED_TEXT = (
    "Explain how a computer works in simple terms. Walk through the CPU, memory, disk, and "
    "how they coordinate, using plain language and concrete examples a beginner could follow. "
)


def sample_lengths(rng: random.Random) -> tuple[int, int]:
    """Draw (prompt_tokens, output_tokens) from the length distribution."""
    p = int(min(2048, max(8, rng.lognormvariate(PROMPT_MU, PROMPT_SIGMA))))
    o = int(min(1024, max(8, rng.lognormvariate(OUTPUT_MU, OUTPUT_SIGMA))))
    return p, o


def build_prompt(target_tokens: int) -> str:
    """Tile the seed to ~target_tokens (≈4 chars/token)."""
    chars = target_tokens * 4
    n = max(1, chars // len(SEED_TEXT) + 1)
    return (SEED_TEXT * n)[:chars]


@dataclass
class Sample:
    ttft_s: float = 0.0
    tpot_s: float = 0.0
    out_tokens: int = 0
    error: str | None = None


async def one_request(client: httpx.AsyncClient, prompt: str, max_tokens: int) -> Sample:
    """One streaming /v1/completions call. TTFT = first token chunk; TPOT = mean ITL."""
    body = {"prompt": prompt, "max_tokens": max_tokens, "stream": True, "temperature": 0.0}
    t0 = time.perf_counter()
    ttft = None
    first_tok_t = last_tok_t = None
    n = 0
    try:
        async with client.stream("POST", "/v1/completions", json=body,
                                 timeout=httpx.Timeout(120.0)) as r:
            if r.status_code != 200:
                return Sample(error=f"http {r.status_code}")
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                choices = obj.get("choices")
                if choices and choices[0].get("text"):   # a token chunk (usage chunk has choices=[])
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                        first_tok_t = now
                    last_tok_t = now
                    n += 1
    except Exception as e:
        return Sample(error=type(e).__name__)
    if ttft is None:
        return Sample(error="no_tokens")
    tpot = (last_tok_t - first_tok_t) / (n - 1) if n > 1 else 0.0
    return Sample(ttft_s=ttft, tpot_s=tpot, out_tokens=n)


@dataclass
class RateResult:
    rate: float
    duration_s: float
    wall_s: float = 0.0   # first arrival -> last completion, drain included (vLLM convention)
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [s for s in self.samples if s.error is None]
        errs = [s for s in self.samples if s.error is not None]
        # Arrivals stop at the deadline; completions keep landing through the drain. Past
        # saturation that drain is long, and dividing by the nominal duration reported an
        # "achieved" rate that matched the offered rate however far behind the server fell.
        window = self.wall_s or self.duration_s
        ttfts = sorted(s.ttft_s for s in ok)
        tpots = sorted(s.tpot_s for s in ok if s.tpot_s > 0)
        tokens = sum(s.out_tokens for s in ok)
        p95_ttft = _pct(ttfts, 0.95) * 1000
        p95_tpot = _pct(tpots, 0.95) * 1000
        within = bool(ok) and p95_ttft < SLO_TTFT_MS and p95_tpot < SLO_TPOT_MS
        return {
            "rate": self.rate, "n_ok": len(ok), "n_err": len(errs),
            "achieved_rps": round(len(ok) / window, 2),
            "tok_per_s": round(tokens / window, 1),
            "ttft_p50": round(_pct(ttfts, 0.50) * 1000, 1),
            "ttft_p95": round(p95_ttft, 1),
            "tpot_p50": round(_pct(tpots, 0.50) * 1000, 2),
            "tpot_p95": round(p95_tpot, 2),
            "within_slo": within,
            "wall_s": round(window, 2),
        }


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


async def run_rate(client: httpx.AsyncClient, rate: float, duration_s: float,
                   rng: random.Random) -> RateResult:
    """Open-loop: spawn requests on a Poisson schedule for `duration_s`, then drain in-flight."""
    res = RateResult(rate=rate, duration_s=duration_s)
    tasks: list[asyncio.Task] = []
    t_start = time.perf_counter()
    deadline = t_start + duration_s

    async def fire():
        p_toks, o_toks = sample_lengths(rng)
        res.samples.append(await one_request(client, build_prompt(p_toks), o_toks))

    while time.perf_counter() < deadline:
        tasks.append(asyncio.create_task(fire()))
        await asyncio.sleep(rng.expovariate(rate))   # inter-arrival gap, independent of completions
    if tasks:
        await asyncio.wait(tasks, timeout=150.0)      # drain outstanding requests
    res.wall_s = time.perf_counter() - t_start
    return res


def report(rows: list[dict]) -> dict:
    """Print the sweep table and derive the SLO headline: max throughput within SLO + knee."""
    print(f"\n  SLO: p95 TTFT < {SLO_TTFT_MS:.0f}ms, p95 TPOT < {SLO_TPOT_MS:.0f}ms/tok\n")
    print(f"  {'rate':>5} {'ok':>4} {'err':>4} {'rps':>6} {'tok/s':>7} "
          f"{'TTFT p50/p95':>14} {'TPOT p50/p95':>14}  SLO")
    for s in rows:
        print(f"  {s['rate']:>5.1f} {s['n_ok']:>4} {s['n_err']:>4} {s['achieved_rps']:>6.1f} "
              f"{s['tok_per_s']:>7.1f} {s['ttft_p50']:>6.0f}/{s['ttft_p95']:<7.0f} "
              f"{s['tpot_p50']:>6.1f}/{s['tpot_p95']:<7.1f}  {'ok' if s['within_slo'] else 'X'}")
    within = [s for s in rows if s["within_slo"]]
    best = max(within, key=lambda s: s["tok_per_s"], default=None)
    knee = next((s for s in rows if not s["within_slo"]), None)
    print()
    if best:
        print(f"  MAX THROUGHPUT WITHIN SLO: {best['tok_per_s']:.0f} tok/s @ rate {best['rate']:.1f} req/s")
    else:
        print("  NO rate held the SLO — server is under its target even at the lowest rate.")
    if knee:
        print(f"  SATURATION KNEE: SLO first broken at rate {knee['rate']:.1f} req/s "
              f"(p95 TTFT {knee['ttft_p95']:.0f}ms, p95 TPOT {knee['tpot_p95']:.1f}ms)")
    return {"max_slo_tok_s": best["tok_per_s"] if best else 0,
            "knee_rate": knee["rate"] if knee else None}


async def sweep(base_url: str, rates: list[float], duration_s: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    async with httpx.AsyncClient(base_url=base_url,
                                 limits=httpx.Limits(max_connections=1024)) as client:
        for rate in rates:
            print(f"-- rate {rate} req/s for {duration_s}s --")
            res = await run_rate(client, rate, duration_s, rng)
            rows.append(res.summary())
            await asyncio.sleep(2)   # cooldown so tails don't bleed between rates
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rates", default="2,4,8,16,32", help="Poisson arrival rates (req/s)")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    rates = [float(x) for x in args.rates.split(",")]
    rows = await sweep(args.base_url, rates, args.duration, args.seed)
    report(rows)

    out = Path(args.output) if args.output else Path("benchmarks") / f"serving_{int(time.time())}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
