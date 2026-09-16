"""Discrete-event trace-replay simulator: the tier-1 falsifier for policy hypotheses.

Replays a corpus trace through a model of the scheduler with attention replaced by a fitted
`TimingModel`, so a scheduling / admission / KV-sizing hypothesis dies on a laptop in seconds
instead of on a GPU in an hour. It reproduces the real loop's iteration order — shed expired,
admit in policy order, prefill the wave, one decode step — and calls the SAME pure ordering
functions the engine does (`scheduling_policy.fcfs_order` / `fair_order` / `fair_initial_counter`
/ `fair_charge`); it never reimplements them. That import is the one sanctioned engine import in
`research/`: the module is stdlib-only pure code.

The timing model only has to RANK configurations the way hardware does (spec: 04-simulator.md),
so `rank_correlation` is the validation primitive; the hardware validation itself needs a GPU
and is deferred. Simulated panels carry `harness="simulator"` and are refused by compare.py
against hardware panels, which is correct.

NOT modelled in v1: preemption, chunked prefill, prefix-cache eviction, and cache-held blocks
counting against the pool. `prefill_mode` is "monolithic" (serial, the engine default) or
"batched" (one wave, the custom backend). Prefill stalls decode in both, as in the engine. The
KV gate reserves the engine's worst case, prompt + max_tokens, BEFORE any prefix lookup; a hit
only shrinks prefill cost. Tokens are `len(prompt) // 4` because traces are text.

    python -m inference_server.research.loop simulate --class steady_interactive \\
        --config '{"policy": "fcfs"}' --config '{"policy": "fair"}' [--timing t.json] [--emit]
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inference_server.research import harness as H
from inference_server.research.corpus import TraceRequest, WorkloadClass
from inference_server.research.schemas import Vitals
from inference_server.scheduling_policy import (
    fair_charge,
    fair_initial_counter,
    fair_order,
    fcfs_order,
)

CHARS_PER_TOKEN = 4


def n_tokens(prompt: str) -> int:
    """Token count stand-in for a text trace."""
    return max(1, len(prompt) // CHARS_PER_TOKEN)


# ------------------------------------------------------------------ timing model

@dataclass
class TimingModel:
    """Fitted linear costs; identity says which engine / hardware the fit describes."""

    prefill: tuple[float, float, float]       # s = a + b*prompt_tokens + c*batch_prompt_tokens
    decode: tuple[float, float, float]        # s = a + b*batch_size + c*total_kv_tokens
    model: str = "unknown"
    hardware: str = "unknown"
    engine_sha: str = "unknown"
    fitted_from: str = "placeholder"          # run id, or "placeholder"

    def prefill_s(self, prompt_tokens: int, batch_prompt_tokens: int) -> float:
        a, b, c = self.prefill
        return a + b * prompt_tokens + c * batch_prompt_tokens

    def decode_step_s(self, batch_size: int, total_kv_tokens: int) -> float:
        a, b, c = self.decode
        return a + b * batch_size + c * total_kv_tokens

    def identity(self) -> dict[str, Any]:
        return {"model": self.model, "hardware": self.hardware, "engine_sha": self.engine_sha,
                "fitted_from": self.fitted_from}

    def to_json(self, path: str | Path | None = None) -> str:
        blob = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(blob)
        return blob

    @classmethod
    def from_json(cls, path: str | Path) -> "TimingModel":
        d = json.loads(Path(path).read_text())
        return cls(prefill=tuple(d["prefill"]), decode=tuple(d["decode"]),
                   **{k: d[k] for k in ("model", "hardware", "engine_sha", "fitted_from") if k in d})


# Plan's measured A100 / E4B position: prefill ~203 ms at the mean prompt (~167 tokens),
# TPOT ~24 ms at 32 concurrent. Ranks policies; makes no absolute claim.
PLACEHOLDER_A100_E4B = TimingModel(
    prefill=(0.010, 0.00115, 0.0),
    decode=(0.010, 0.00014, 1.0e-6),
    model="gemma-4-e4b", hardware="A100-80GB", engine_sha="unknown", fitted_from="placeholder",
)


def _lstsq(xs: list[list[float]], ys: list[float]) -> list[float]:
    """Ordinary least squares via normal equations + Gaussian elimination. Stdlib only."""
    k = len(xs[0])
    a = [[sum(r[i] * r[j] for r in xs) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * y for r, y in zip(xs, ys)) for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            raise ValueError("singular fit: the normal matrix has a (near-)zero pivot, so the "
                             "predictors are collinear or constant")
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(k):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
                b[r] -= f * b[col]
    return [b[i] / a[i][i] for i in range(k)]


def _fit(rows: list[dict[str, Any]], y: str, x1: str, x2: str) -> tuple[float, float, float]:
    """Fit y = a + b*x1 + c*x2; an absent or all-zero x2 is dropped and gets c = 0."""
    use = [r for r in rows if r.get(y) is not None]
    if len(use) < 2:
        raise ValueError(f"need >= 2 rows with {y}, got {len(use)}")
    with_x2 = any(r.get(x2) for r in use)
    xs = [[1.0, float(r[x1])] + ([float(r.get(x2) or 0)] if with_x2 else []) for r in use]
    coef = _lstsq(xs, [float(r[y]) for r in use])
    return (coef[0], coef[1], coef[2] if with_x2 else 0.0)


def fit_timing_model(rows: list[dict[str, Any]], **identity: str) -> TimingModel:
    """Least-squares fit from replay / telemetry rows.

    Prefill rows carry `prompt_tokens`, `prefill_s` [, `batch_prompt_tokens`]; decode rows carry
    `batch_size`, `decode_step_s` [, `total_kv_tokens`]. One row may carry both.
    """
    return TimingModel(
        prefill=_fit(rows, "prefill_s", "prompt_tokens", "batch_prompt_tokens"),
        decode=_fit(rows, "decode_step_s", "batch_size", "total_kv_tokens"),
        **identity,
    )


# ------------------------------------------------------------------ config

POLICIES = ("fcfs", "fair")
PREFILL_MODES = ("monolithic", "batched")


@dataclass(frozen=True)
class SimConfig:
    """The knobs a hypothesis varies. Frozen, so it hashes and diffs."""

    max_batch_size: int = 32
    max_queue_size: int = 256
    max_queue_wait_s: float = 30.0          # <= 0 disables the admission deadline
    kv_blocks: int = 4096
    block_size: int = 16
    policy: str = "fcfs"
    rate_scale: float = 1.0
    prefill_mode: str = "monolithic"        # monolithic: serial; batched: one wave

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {self.policy!r}")
        if self.prefill_mode not in PREFILL_MODES:
            raise ValueError(f"prefill_mode must be one of {PREFILL_MODES}, "
                             f"got {self.prefill_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ state

@dataclass
class _Req:
    """One request's simulator record. Satisfies scheduling_policy.Candidate."""

    index: int
    session_id: str
    priority: int
    arrival_seq: int
    trace: TraceRequest
    arrival: float                  # sim clock (trace arrival / rate_scale)
    prompt_tokens: int
    matched_tokens: int = 0
    blocks: int = 0
    admit: float | None = None
    first_token: float | None = None
    finish: float | None = None
    generated: int = 0
    error: str | None = None


@dataclass
class Row:
    """replay_trace.Row's columns, plus the split and the cache hit the simulator can see."""

    index: int
    session_id: str
    turn_index: int
    arrival_s: float
    fired_s: float
    ttft_ms: float | None
    tpot_ms: float | None
    out_tokens: int
    error: str | None
    matched_tokens: int
    queue_ms: float | None
    prefill_ms: float | None


@dataclass
class SimResult:
    cfg: SimConfig
    timing: TimingModel
    rows: list[Row] = field(default_factory=list)
    wall_s: float = 0.0
    wave_sizes: dict[int, int] = field(default_factory=dict)
    kv_admit_blocked: int = 0
    total_rejected: int = 0
    total_expired: int = 0
    decode_steps: int = 0
    active_high_water: int = 0
    active_sum: int = 0
    cache_lookups: int = 0
    cache_hits: int = 0
    pool_free_end: int = 0          # free blocks when the run ends
    pool_free_min: int = 0          # the low-water mark, the useful one for a replay

    def summary(self, cls: WorkloadClass) -> dict[str, Any]:
        """Same keys replay_trace reports, plus the pressure signals the simulator can see."""
        ok = [r for r in self.rows if r.error is None]
        ttfts = [r.ttft_ms for r in ok]
        tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
        window = self.wall_s or 1e-9
        p95_ttft = H.pct(ttfts, 0.95)
        p95_tpot = H.pct(tpots, 0.95) if tpots else None
        return {
            "n_ok": len(ok), "n_err": len(self.rows) - len(ok),
            "achieved_rps": round(len(ok) / window, 2),
            "tok_per_s": round(sum(r.out_tokens for r in ok) / window, 1),
            "ttft_p50": round(H.pct(ttfts, 0.50), 1), "ttft_p95": round(p95_ttft, 1),
            "ttft_queue_p50": round(H.pct([r.queue_ms for r in ok], 0.50), 1),
            "ttft_queue_p95": round(H.pct([r.queue_ms for r in ok], 0.95), 1),
            "ttft_prefill_p50": round(H.pct([r.prefill_ms for r in ok], 0.50), 1),
            "ttft_prefill_p95": round(H.pct([r.prefill_ms for r in ok], 0.95), 1),
            "tpot_p50": round(H.pct(tpots, 0.50), 2) if tpots else None,
            "tpot_p95": round(p95_tpot, 2) if p95_tpot is not None else None,
            "within_slo": bool(ok) and cls.within_slo(p95_ttft, p95_tpot),
            "wall_s": round(self.wall_s, 2),
            "wave_sizes": dict(sorted(self.wave_sizes.items())),
            "kv_admit_blocked": self.kv_admit_blocked,
            "total_rejected": self.total_rejected, "total_expired": self.total_expired,
            "decode_steps": self.decode_steps,
            "active_high_water": self.active_high_water,
            "active_mean": (round(self.active_sum / self.decode_steps, 2)
                            if self.decode_steps else None),
            "pool_free_end": self.pool_free_end, "pool_free_min": self.pool_free_min,
            "cache_hit_rate": (round(self.cache_hits / self.cache_lookups, 3)
                               if self.cache_lookups else None),
        }

    def to_panel(self, cls: WorkloadClass, *, corpus_version: str | None = None,
                 split: str | None = None, seed: int = 0, notes: str = "") -> Vitals:
        """A Vitals panel that is unmistakably simulated: harness="simulator", config + timing
        identity in harness_config. compare.py refuses it against hardware panels."""
        s = self.summary(cls)
        ok = [r for r in self.rows if r.error is None]
        validity = H.build_validity(
            "simulator",
            {**self.cfg.to_dict(), "timing": self.timing.identity(), "seed": seed,
             "workload_class": cls.name, "split": split, "corpus_version": corpus_version,
             "n_requests": len(self.rows)},
            n_samples=s["n_ok"],
            workload_regime=H.infer_regime(s["cache_hit_rate"]),
            stderr_value=H.stderr(r.ttft_ms for r in ok),
            concurrency_observed=s["active_high_water"],
            notes=notes or "simulated: ranks policies, makes no absolute claim",
        )
        validity.corpus_version = corpus_version
        validity.workload_class = cls.name
        sched = {k: s[k] for k in ("wave_sizes", "kv_admit_blocked", "total_rejected",
                                   "total_expired", "decode_steps", "active_high_water",
                                   "active_mean")}
        cache = {"hit_rate": s["cache_hit_rate"], "lookups": self.cache_lookups,
                 "pool_total_blocks": self.cfg.kv_blocks, "pool_free_blocks": self.pool_free_end,
                 # peak, not end-of-run: the pressure a replay actually reached
                 "pool_utilization": round(1 - self.pool_free_min / self.cfg.kv_blocks, 3)}
        return H.panel_from_stats(
            validity, scheduler_stats={**sched, "total_preempted": 0, "total_iteration_errors": 0},
            cache_stats=cache,
            tok_s_within_slo=s["tok_per_s"] if s["within_slo"] else None,
            slo_ttft_ms=cls.slo_ttft_ms, slo_tpot_ms=cls.slo_tpot_ms,
            ttft_p50=s["ttft_p50"], ttft_p95=s["ttft_p95"],
            ttft_queue_p50=s["ttft_queue_p50"], ttft_queue_p95=s["ttft_queue_p95"],
            ttft_prefill_p50=s["ttft_prefill_p50"], ttft_prefill_p95=s["ttft_prefill_p95"],
            tpot_p50=s["tpot_p50"], tpot_p95=s["tpot_p95"], wall_s=s["wall_s"],
        )


# ------------------------------------------------------------------ the loop

class _PrefixCache:
    """PrefixCache.lookup semantics on text: longest block-aligned prefix already stored."""

    def __init__(self, block_size: int) -> None:
        self.chars = block_size * CHARS_PER_TOKEN
        self.block_size = block_size
        self.keys: set[tuple[int, str]] = set()
        self.lookups = 0
        self.hits = 0

    def _key(self, prompt: str, n: int) -> tuple[int, str]:
        return n, hashlib.sha1(prompt[: n * self.chars].encode()).hexdigest()

    def lookup(self, prompt: str) -> int:
        self.lookups += 1
        for n in range(len(prompt) // self.chars, 0, -1):
            if self._key(prompt, n) in self.keys:
                self.hits += 1
                return n * self.block_size
        return 0

    def store(self, prompt: str) -> None:
        for n in range(1, len(prompt) // self.chars + 1):
            self.keys.add(self._key(prompt, n))


def simulate(requests: list[TraceRequest], cfg: SimConfig, timing: TimingModel,
             seed: int = 0) -> SimResult:
    """Deterministic replay: admit -> prefill -> one decode step, per tick, on the trace clock.

    `seed` is recorded for provenance; v1 has no stochastic component.
    """
    order = sorted(range(len(requests)), key=lambda i: requests[i].arrival_s)
    reqs = [_Req(i, requests[i].session_id, 0, seq, requests[i],
                 requests[i].arrival_s / cfg.rate_scale, n_tokens(requests[i].prompt))
            for seq, i in enumerate(order)]
    res = SimResult(cfg=cfg, timing=timing)
    cache = _PrefixCache(cfg.block_size)
    counters: dict[str, float] = {}
    pending: list[_Req] = []
    active: list[_Req] = []
    free_blocks = free_min = cfg.kv_blocks
    now, next_arrival = 0.0, 0

    def reject(r: _Req, why: str, expired: bool = False) -> None:
        r.error = why
        res.total_rejected += 1
        res.total_expired += expired

    while next_arrival < len(reqs) or pending or active:
        while next_arrival < len(reqs) and reqs[next_arrival].arrival <= now:
            r = reqs[next_arrival]
            next_arrival += 1
            if len(pending) >= cfg.max_queue_size:
                reject(r, "429")
                continue
            if cfg.policy == "fair":
                counters[r.session_id] = fair_initial_counter(
                    r.session_id, counters, (p.session_id for p in pending))
            pending.append(r)
        if not pending and not active:
            now = reqs[next_arrival].arrival      # idle: jump to the next arrival
            continue

        # (a) shed expired, wherever they sit in the queue (see scheduler._shed_expired)
        if cfg.max_queue_wait_s > 0:
            for r in [p for p in pending if now - p.arrival > cfg.max_queue_wait_s]:
                pending.remove(r)
                reject(r, "expired", expired=True)

        # (b) admit in policy order while a slot and the KV reservation fit (HOL-wait on KV).
        # Worst-case reservation, before any prefix lookup, exactly as scheduler._admit_pending.
        ranked = (fair_order(pending, counters) if cfg.policy == "fair" else fcfs_order(pending))
        wave: list[_Req] = []
        for r in ranked:
            if len(active) + len(wave) >= cfg.max_batch_size:
                break
            r.blocks = math.ceil((r.prompt_tokens + r.trace.max_tokens) / cfg.block_size)
            if r.blocks > cfg.kv_blocks:
                pending.remove(r)
                reject(r, "kv_too_large")
                continue
            if r.blocks > free_blocks:
                res.kv_admit_blocked += 1
                break
            free_blocks -= r.blocks
            free_min = min(free_min, free_blocks)
            pending.remove(r)
            r.admit = now
            wave.append(r)

        # (c) prefill: decode stalls, first token at each prefill's end. A prefix hit shrinks
        # the work here only; the reservation above already happened.
        if wave:
            for r in wave:
                r.matched_tokens = cache.lookup(r.trace.prompt)
            batched = cfg.prefill_mode == "batched"
            if batched:
                res.wave_sizes[len(wave)] = res.wave_sizes.get(len(wave), 0) + 1
            wave_tokens = sum(r.prompt_tokens - r.matched_tokens for r in wave) if batched else 0
            for r in wave:
                now += timing.prefill_s(r.prompt_tokens - r.matched_tokens, wave_tokens)
                r.first_token = now
                r.generated = 1
                cache.store(r.trace.prompt)
                active.append(r)

        # (d) one decode step for every active row
        if active:
            kv = sum(r.prompt_tokens + r.generated for r in active)
            now += timing.decode_step_s(len(active), kv)
            res.decode_steps += 1
            res.active_sum += len(active)
            res.active_high_water = max(res.active_high_water, len(active))
            for r in active:
                if r.generated < r.trace.max_tokens:
                    r.generated += 1
                if cfg.policy == "fair":
                    fair_charge(counters, r.session_id, 1)
            for r in [a for a in active if a.generated >= a.trace.max_tokens]:
                active.remove(r)
                free_blocks += r.blocks
                r.finish = now
        elif pending and not wave:
            raise RuntimeError("simulator stalled: pending work, nothing admitted, nothing active")

    res.wall_s = now
    res.cache_lookups, res.cache_hits = cache.lookups, cache.hits
    res.pool_free_end, res.pool_free_min = free_blocks, free_min
    for r in sorted(reqs, key=lambda r: r.index):
        ok = r.error is None
        tpot = ((r.finish - r.first_token) / (r.generated - 1) * 1000
                if ok and r.generated > 1 else None)
        res.rows.append(Row(
            r.index, r.session_id, r.trace.turn_index, r.trace.arrival_s, round(r.arrival, 4),
            round((r.first_token - r.arrival) * 1000, 2) if ok else None,
            round(tpot, 3) if tpot is not None else None,
            r.generated if ok else 0, r.error, r.matched_tokens,
            round((r.admit - r.arrival) * 1000, 2) if ok else None,
            round((r.first_token - r.admit) * 1000, 2) if ok else None,
        ))
    return res


# ------------------------------------------------------------------ validation primitive

def _ranks(xs: list[float]) -> list[float]:
    """Average ranks (1-based), ties share their mean rank."""
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def rank_correlation(a: list[float], b: list[float]) -> float:
    """Spearman rho: does the simulator order configurations the way hardware does?"""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length lists of >= 2 values")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return cov / var if var else 0.0      # all ties on one side: no ordering to agree with
