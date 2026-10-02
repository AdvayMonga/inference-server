"""Build the frozen workload corpus from real public traces, deterministically, then commit it.

    uv sync --extra dev --extra corpus
    .venv/bin/python scripts/corpus/fetch_traces.py           # once: raw inputs into the cache
    .venv/bin/python scripts/corpus/build_corpus.py [--seed 20261001]

Timing and sessions come from BurstGPT (`BurstGPT_3.csv`): every arrival, its session (API-log
requests have none and become one-request sessions) and its `Request tokens`. Text comes from
WildChat-1M: real conversations with their stored assistant replies. Each BurstGPT session is
paired with one WildChat conversation, and each of its requests with the conversation turn
whose templated prompt length (the build tokenizer, Qwen3 by default, enable_thinking=False) is
closest to the trace's Request tokens, turns strictly increasing within a session. Turn k > 0
is a real `[user, assistant, ..., user]` message list. No conversation is used twice.

Same seed + same cached inputs + same tokenizer = same bytes (the test asserts it). Anything
else is a NEW corpus version. Do not hand-edit a trace.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_traces import DEFAULT_CACHE, QWEN3_REV, WILDCHAT_JSONL, WILDCHAT_REV  # noqa: E402
from lab.corpus import (  # noqa: E402
    SPLITS,
    Manifest,
    TraceRequest,
    WorkloadClass,
    build_manifest,
    write_trace,
)

DEFAULT_SEED = 20261001
DEFAULT_TOKENIZER = "tokenizers/Qwen3-30B-A3B"     # relative to the cache, or any HF id/path
TOKENIZER_LABEL = f"Qwen/Qwen3-30B-A3B@{QWEN3_REV[:12]}"
BURSTGPT_CSV = "burstgpt/BurstGPT_3.csv"
PROMPT_CAP = 16384       # templated prompt tokens; longer turns are never used
MAX_TOKENS = 2048        # generous, so answers end on their own; prompt + budget < 32k context
MIN_TARGET = 16          # Request tokens of 0 (failed logs) still need a prompt
WEEK_S = 7 * 86400       # even trace weeks are `seen`, odd weeks `heldout`
MAX_SESSION_TURNS = 16   # longer window sessions are split into consecutive conversations
CANDIDATES = 400         # conversations scored per multi-request session

NOTES_TMPL = (
    "Real-trace corpus. Timing + sessions: BurstGPT v2.0 BurstGPT_3.csv (HPMLL, CC-BY-4.0). "
    "Text: WildChat-1M @{wc} shards 0-1 (allenai, ODC-BY), English, not toxic/flagged/redacted, "
    "no conversation containing a credential-like string (SECRET_RE). "
    "Azure LLM inference trace 2024 (CC-BY-4.0) is fetched as a cross-source check only, not "
    "a split (no sessions; see corpus/README.md). Tokenizer {tok}, chat template "
    "enable_thinking=False, prompt cap {cap}, max_tokens {mt}. Seed {seed}. Splits: seen = even "
    "weeks of the trace, heldout = odd weeks. `dilation` per trace: arrival_s = real offset x "
    "dilation; replay with --rate-scale <dilation> for real timing. Each request's "
    "build_prompt_tokens is its templated length under this tokenizer. SLOs proposed "
    "2026-10-01, pending owner confirmation."
)


@dataclass
class ClassSpec:
    cls: WorkloadClass
    target_rps: float          # a window faster than this is dilated down to it
    min_tokens: int = 0        # keep only trace requests at least this long (long_context)


def _specs() -> list[ClassSpec]:
    def wc(name, desc, ttft, tpot, rps):
        return WorkloadClass(name=name, description=desc, slo_ttft_ms=ttft, slo_tpot_ms=tpot,
                             arrival_rate_rps=rps, seen=f"{name}/seen.jsonl",
                             heldout=f"{name}/heldout.jsonl")
    return [
        ClassSpec(wc("cold_start", "The first 60 requests after a >=10 min idle gap in the "
                     "trace, arriving within 20 min, with varied prompt lengths (not one "
                     "scripted client): what a replica woken from zero sees.",
                     5000.0, None, 0.5), target_rps=0.5),
        ClassSpec(wc("steady_interactive", "A steady busy 10-minute window (200-300 requests, "
                     "no minute above 2x the window mean): a warm replica under real load.",
                     1000.0, 100.0, 0.5), target_rps=0.5),
        ClassSpec(wc("long_context", "60 consecutive trace requests of >=3000 tokens, paired "
                     "with long WildChat prompts and histories: prefill and KV pressure.",
                     3000.0, None, 0.1), target_rps=0.1, min_tokens=3000),
        ClassSpec(wc("spike", "A real burst: a minute of 40-150 requests after ten quiet "
                     "minutes, with five minutes of lead-in and three of tail.",
                     2000.0, None, 2.5), target_rps=2.5),
    ]


# ------------------------------------------------------------------ inputs

@dataclass
class Arrival:
    t: float               # seconds from the trace's first row (BurstGPT has 1 s resolution)
    session: str           # "" for API-log requests
    tokens: int


def split_of(t: float) -> str:
    return SPLITS[int(t // WEEK_S) % 2]


@lru_cache(maxsize=2)
def load_burstgpt(cache: Path) -> tuple[Arrival, ...]:
    with open(cache / BURSTGPT_CSV) as f:
        rows = list(csv.DictReader(f))
    t0 = float(rows[0]["Timestamp"])
    # A failed request is logged with 0 request tokens: no length to pair, so it is dropped.
    return tuple(Arrival(float(r["Timestamp"]) - t0, r["Session ID"], int(r["Request tokens"]))
                 for r in rows if int(r["Request tokens"]) > 0)


# Credentials users pasted into chats, placeholders included (scanners flag those too).
SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_\w{40,}"
    r"|xox[abposr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def _keep(r: dict) -> bool:
    m = r["messages"]
    return (r["language"] == "English" and not r["toxic"] and not r["flagged"]
            and not r["redacted"] and len(m) >= 2 and len(m) % 2 == 0
            and all(x["role"] == ("user", "assistant")[i % 2] for i, x in enumerate(m))
            and all(x["content"].strip() for x in m)
            and not any(SECRET_RE.search(x["content"]) for x in m))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Pool:
    convs: list[list[dict]]            # message lists
    lens: list[list[int]]              # lens[c][k]: templated tokens of the prompt ending at turn k
    index: list[tuple[int, int, int]] = field(default_factory=list)   # sorted (len, c, k)


@lru_cache(maxsize=2)
def load_pool(cache: Path, tokenizer: str) -> Pool:
    """Filtered WildChat, de-duplicated by first user message, with per-turn templated lengths.
    The lengths are cached under `derived/`, keyed by every input that determines them."""
    convs, seen_first = [], set()
    with open(cache / WILDCHAT_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if _keep(r) and r["messages"][0]["content"] not in seen_first:
                seen_first.add(r["messages"][0]["content"])
                convs.append(r["messages"])
    tok_dir = cache / tokenizer
    key_src = [_sha(cache / WILDCHAT_JSONL), str(PROMPT_CAP), "v2"]   # bump when _keep changes
    key_src += [_sha(p) for p in sorted(tok_dir.iterdir())] if tok_dir.is_dir() else [tokenizer]
    key = hashlib.sha256("\n".join(key_src).encode()).hexdigest()[:16]
    lens_path = cache / "derived" / f"prefix_lens-{key}.json"
    if lens_path.exists():
        lens = json.loads(lens_path.read_text())
    else:
        lens = _prefix_lens(convs, str(tok_dir) if tok_dir.is_dir() else tokenizer)
        lens_path.parent.mkdir(parents=True, exist_ok=True)
        lens_path.write_text(json.dumps(lens))
    pool = Pool(convs, lens)
    pool.index = sorted((n, c, k) for c, ls in enumerate(lens) for k, n in enumerate(ls)
                        if n >= MIN_TARGET)
    return pool


def _prefix_lens(convs: list[list[dict]], tokenizer: str) -> list[list[int]]:
    """Templated prompt length of each turn, truncated before the first one over the cap."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(tokenizer)
    out: list[list[int]] = []
    for start in range(0, len(convs), 2000):
        texts, owner = [], []
        for c, m in enumerate(convs[start:start + 2000], start):
            for k in range(len(m) // 2):
                text = tk.apply_chat_template(m[:2 * k + 1], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
                if len(text) > 8 * PROMPT_CAP:      # far over the cap; so is every later turn
                    break
                texts.append(text)
                owner.append(c)
        lens: dict[int, list[int]] = defaultdict(list)
        for c, ids in zip(owner, tk(texts, add_special_tokens=False)["input_ids"]):
            lens[c].append(len(ids))
        for c in range(start, min(start + 2000, len(convs))):
            ls = lens.get(c, [])
            cut = next((i for i, n in enumerate(ls) if n > PROMPT_CAP), len(ls))
            out.append(ls[:cut])
    return out


# ------------------------------------------------------------------ windows

def varied(window: list[Arrival]) -> bool:
    """Mixed prompt lengths, not one scripted client: request-token CV >= 0.3 and at most 60% of
    requests within +-25% of the median. Most post-idle BurstGPT windows fail this."""
    tok = sorted(a.tokens for a in window)
    med, mean = tok[len(tok) // 2], sum(tok) / len(tok)
    cv = math.sqrt(sum((x - mean) ** 2 for x in tok) / len(tok)) / mean
    return cv >= 0.3 and sum(0.75 * med <= x <= 1.25 * med for x in tok) <= 0.6 * len(tok)


def _windows_cold_start(arr, spec):
    t = [a.t for a in arr]
    for i in range(1, len(arr) - 59):
        if t[i] - t[i - 1] >= 600 and t[i + 59] - t[i] <= 1200 and varied(arr[i:i + 60]):
            yield list(arr[i:i + 60])


def _windows_steady(arr, spec):
    t = [a.t for a in arr]
    for b in range(int(t[-1] // 600)):
        lo, hi = bisect.bisect_left(t, b * 600), bisect.bisect_left(t, (b + 1) * 600)
        if not 200 <= hi - lo <= 300:
            continue
        per_min = [0] * 10
        for x in t[lo:hi]:
            per_min[int(x // 60) - b * 10] += 1
        if max(per_min) <= 2 * (hi - lo) / 10:
            yield list(arr[lo:hi])


def _windows_long(arr, spec):
    longs = [a for a in arr if a.tokens >= spec.min_tokens]
    for i in range(0, len(longs) - 59, 30):
        if 600 <= longs[i + 59].t - longs[i].t <= 2400:
            yield longs[i:i + 60]


def _windows_spike(arr, spec):
    t = [a.t for a in arr]
    per_min = [0] * (int(t[-1] // 60) + 1)
    for x in t:
        per_min[int(x // 60)] += 1
    for i in range(10, len(per_min) - 2):
        before = sorted(per_min[i - 10:i])
        if (40 <= per_min[i] <= 150 and per_min[i] >= 6 * max(before[5], 1)
                and before[-1] < per_min[i] / 2):
            lo, hi = bisect.bisect_left(t, i * 60 - 300), bisect.bisect_left(t, i * 60 + 180)
            if 100 <= hi - lo <= 300:
                yield list(arr[lo:hi])


WINDOWS = {"cold_start": _windows_cold_start, "steady_interactive": _windows_steady,
           "long_context": _windows_long, "spike": _windows_spike}


def pick_window(arr, spec: ClassSpec, split: str, rng: random.Random) -> list[Arrival]:
    """One real window lying wholly inside one of `split`'s weeks, chosen by the seed."""
    cands = [w for w in WINDOWS[spec.cls.name](arr, spec)
             if split_of(w[0].t) == split and int(w[0].t // WEEK_S) == int(w[-1].t // WEEK_S)]
    if not cands:
        raise SystemExit(f"no {spec.cls.name} window in {split}")
    return rng.choice(cands)


def dilation(window: list[Arrival], spec: ClassSpec) -> float:
    """>= 1. Every inter-arrival is stretched by the same factor, so the burst SHAPE is kept."""
    if spec.cls.name == "spike":       # the peak minute, not the mean, is what must be servable
        per_min: dict[int, int] = defaultdict(int)
        for a in window:
            per_min[int(a.t // 60)] += 1
        rate = max(per_min.values()) / 60
    else:
        rate = (len(window) - 1) / max(window[-1].t - window[0].t, 1.0)
    return round(max(1.0, rate / spec.target_rps), 3)


# ------------------------------------------------------------------ pairing

def _cost(n: int, target: int) -> float:
    """Ratio distance; a turn shorter than MIN_TARGET is never chosen."""
    return abs(math.log(n / max(target, MIN_TARGET))) if n >= MIN_TARGET else float("inf")


def assign_turns(lens: list[int], targets: list[int]) -> tuple[float, list[int]]:
    """Strictly increasing turns minimising the summed |log(len / target)|. O(n * K)."""
    n, K, inf = len(targets), len(lens), float("inf")
    best = [[_cost(lens[k], targets[0]) for k in range(K)]]
    back = [[-1] * K]
    for j in range(1, n):
        row, brow, run, arg = [inf] * K, [-1] * K, inf, -1
        for k in range(1, K):
            if best[j - 1][k - 1] < run:
                run, arg = best[j - 1][k - 1], k - 1
            if arg >= 0:
                row[k], brow[k] = run + _cost(lens[k], targets[j]), arg
        best.append(row)
        back.append(brow)
    k = min(range(K), key=lambda x: (best[-1][x], x))
    total, turns = best[-1][k], [k]
    for j in range(n - 1, 0, -1):
        k = back[j][k]
        turns.append(k)
    return total, turns[::-1]


def pair_single(index: list[tuple[int, int, int]], target: int,
                used: set[int]) -> tuple[int, int]:
    """The unused (conversation, turn) in `index` nearest the target length, as a ratio."""
    target = min(max(target, MIN_TARGET), PROMPT_CAP)
    i = bisect.bisect_left(index, (target, -1, -1))
    lo, hi = i - 1, i
    while lo >= 0 and index[lo][1] in used:
        lo -= 1
    while hi < len(index) and index[hi][1] in used:
        hi += 1
    cand = [index[j] for j in (lo, hi) if 0 <= j < len(index)]
    _, c, k = min(cand, key=lambda x: (_cost(x[0], target), x[0]))
    return c, k


def pair_session(pool: Pool, targets: list[int], used: set[int],
                 rng: random.Random) -> tuple[int, list[int]]:
    """Best of a seeded sample of unused conversations long enough for the session."""
    eligible = [c for c, ls in enumerate(pool.lens) if len(ls) >= len(targets) and c not in used]
    sample = rng.sample(eligible, min(CANDIDATES, len(eligible)))
    scored = [(assign_turns(pool.lens[c], targets), c) for c in sample]
    (cost, turns), c = min(scored, key=lambda x: (x[0][0], x[1]))
    assert cost < float("inf"), "no candidate conversation fits the session"
    return c, turns


def flatten(messages: list[dict]) -> str:
    return "\n\n".join(m["content"] for m in messages)


def build_trace(window: list[Arrival], spec: ClassSpec, split: str, dil: float, pool: Pool,
                used: set[int], rng: random.Random) -> tuple[list[TraceRequest], list[int]]:
    """Group the window into sessions, pair each with a conversation, emit one request per
    arrival. Also returns each request's templated prompt length."""
    groups: dict[str, list[Arrival]] = {}
    for i, a in enumerate(window):
        groups.setdefault(a.session or f"api-{i}", []).append(a)
    chunks = [g[i:i + MAX_SESSION_TURNS] for g in groups.values()
              for i in range(0, len(g), MAX_SESSION_TURNS)]
    t0 = window[0].t
    out: list[tuple[TraceRequest, int]] = []
    for si, g in enumerate(chunks):        # first-arrival order: dicts keep insertion order
        targets = [a.tokens for a in g]
        if len(g) == 1:     # any turn: an API call's Request tokens may well include history
            c, k = pair_single(pool.index, targets[0], used)
            turns = [k]
        else:
            c, turns = pair_session(pool, targets, used, rng)
        used.add(c)
        sid = f"{spec.cls.name}-{split}-{si:04d}"
        for a, k in zip(g, turns):
            msgs = pool.convs[c][:2 * k + 1]
            n = pool.lens[c][k]
            assert MIN_TARGET <= n <= PROMPT_CAP, (sid, n)
            req = TraceRequest(round((a.t - t0) * dil, 3), sid, k,
                               flatten(msgs), MAX_TOKENS, messages=msgs if k else None,
                               build_prompt_tokens=n)
            out.append((req, n))
    out.sort(key=lambda x: (x[0].arrival_s, x[0].session_id, x[0].turn_index))
    return [r for r, _ in out], [n for _, n in out]


# ------------------------------------------------------------------ corpus

def build_corpus(out: Path, seed: int = DEFAULT_SEED, cache: Path = DEFAULT_CACHE,
                 tokenizer: str = DEFAULT_TOKENIZER, verbose: bool = False) -> Manifest:
    """Write every (class, split) trace and the manifest. Class c, split s uses seed + 100c + s;
    conversations are drawn in that order, so reordering classes is a new corpus version."""
    cache = Path(cache).expanduser()
    arr, pool = load_burstgpt(cache), load_pool(cache, tokenizer)
    used: set[int] = set()
    dils: dict[str, float] = {}
    specs = _specs()
    for ci, spec in enumerate(specs):
        for si, split in enumerate(SPLITS):
            rng = random.Random(seed + 100 * ci + si)
            window = pick_window(arr, spec, split, rng)
            dil = dilation(window, spec)
            trace, lens = build_trace(window, spec, split, dil, pool, used, rng)
            write_trace(out / spec.cls.trace_file(split), trace)
            dils[spec.cls.trace_file(split)] = dil
            if verbose:
                _report(spec, split, window, dil, trace, lens)
    tok = TOKENIZER_LABEL if tokenizer == DEFAULT_TOKENIZER else tokenizer
    notes = NOTES_TMPL.format(wc=WILDCHAT_REV[:12], tok=tok, cap=PROMPT_CAP, mt=MAX_TOKENS,
                              seed=seed)
    m = build_manifest({s.cls.name: s.cls for s in specs}, out, notes=notes, dilation=dils)
    (out / "manifest.json").write_text(json.dumps(m.to_dict(), indent=2) + "\n")
    return m


def _report(spec, split, window, dil, trace, lens) -> None:
    print(f"  {spec.cls.name:18s} {split:7s} n={len(trace):3d} "
          f"sessions={len({r.session_id for r in trace}):3d} "
          f"multi-turn={sum(r.messages is not None for r in trace):3d} "
          f"day={window[0].t / 86400:6.2f} span={trace[-1].arrival_s:7.1f}s dil=x{dil} "
          f"prompt_tok={min(lens)}-{max(lens)} (median {sorted(lens)[len(lens) // 2]}) "
          f"budget={min(r.max_tokens for r in trace)}-{max(r.max_tokens for r in trace)}")


def azure_check(cache: Path) -> None:
    """Cross-source sanity check: the Azure 2024 conversation trace beside BurstGPT."""
    import statistics as st
    from datetime import datetime

    def summary(name, ts, toks):
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        per_min: dict[int, int] = defaultdict(int)
        for x in ts:
            per_min[int((x - ts[0]) // 60)] += 1
        mins = sorted(per_min.get(i, 0) for i in range(int((ts[-1] - ts[0]) // 60) + 1))
        q = sorted(toks)
        print(f"  {name:9s} n={len(ts)} mean_rps={len(ts) / (ts[-1] - ts[0]):.2f} "
              f"interarrival_cv={st.pstdev(gaps) / st.mean(gaps):.2f} "
              f"per_min p50/p99/max={mins[len(mins) // 2]}/{mins[int(len(mins) * .99)]}/"
              f"{mins[-1]} prompt_tok p50/p90/p99={q[len(q) // 2]}/{q[int(len(q) * .9)]}/"
              f"{q[int(len(q) * .99)]}")

    arr = load_burstgpt(cache)
    summary("burstgpt", [a.t for a in arr], [a.tokens for a in arr])
    ts, toks = [], []
    with open(cache / "azure/AzureLLMInferenceTrace_conv_1week.csv") as f:
        for r in csv.DictReader(f):
            ts.append(datetime.fromisoformat(r["TIMESTAMP"]).timestamp())
            toks.append(int(r["ContextTokens"]))
    summary("azure", ts, toks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "corpus"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="tokenizer dir relative to the cache, or an HF id; a different one is "
                         "a new corpus version")
    ap.add_argument("--azure-check", action="store_true",
                    help="print BurstGPT vs Azure 2024 arrival/length statistics and exit")
    args = ap.parse_args()
    if args.azure_check:
        sys.exit(azure_check(Path(args.cache).expanduser()))
    m = build_corpus(Path(args.out), args.seed, Path(args.cache), args.tokenizer, verbose=True)
    print(f"corpus_version {m.corpus_version}")
    for name, c in m.classes.items():
        print(f"  {name:20s} ttft<{c.slo_ttft_ms:.0f}ms rate={c.arrival_rate_rps} rps")
