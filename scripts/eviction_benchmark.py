"""Eviction policy benchmark: compare LRU vs AttentionSinkLRU vs H2O.

Spawns a fresh server per policy with a small KV budget that forces eviction,
runs a controlled workload, and prints a markdown comparison table.

Usage:
    python scripts/eviction_benchmark.py
"""

import datetime
import html as html_lib
import os
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

PORT = 8766
SERVER_URL = f"http://127.0.0.1:{PORT}"
NUM_BLOCKS = 16  # very tight — forces aggressive eviction
BLOCK_SIZE = 16
POLICIES = ["lru", "attention_sink_lru", "h2o"]

# Each system prompt is intentionally unique (no shared prefix between them)
# and long enough to occupy several blocks. This kills cross-prompt sharing
# and forces the cache to either evict or refuse new blocks.
SYSTEM_PROMPTS = [
    "Imagine you are an expert in ancient Roman history teaching a graduate seminar. Be meticulous, cite primary sources when possible, and use formal academic prose. Question: ",
    "Pretend you are a senior structural engineer explaining steel frame design to a junior colleague. Use industry terminology and reference relevant codes. Question: ",
    "Act as a marine biologist studying deep sea ecosystems near hydrothermal vents. Discuss findings with technical accuracy and scientific rigor. Question: ",
    "You are a master pastry chef from Lyon training apprentices in classical French technique. Speak with authority, precision, and culinary detail. Question: ",
    "Imagine you are a quantum physicist working on entanglement experiments at a national lab. Use proper notation and explain phenomena rigorously. Question: ",
    "Act as a constitutional lawyer reviewing landmark Supreme Court opinions for first year students. Reference cases by name and year. Question: ",
    "You are a veteran wildlife photographer who has spent decades in Patagonia. Describe technical and artistic aspects vividly. Question: ",
    "Pretend you are a renaissance composer discussing counterpoint with a fellow musician. Use period appropriate musical vocabulary. Question: ",
]

SUFFIXES = [
    "What is the most important consideration here?",
    "How would you explain this to a complete beginner?",
    "What are the most common mistakes people make?",
]


def build_workload() -> list[str]:
    """Phase 1: fill far past capacity. Phase 2: revisit older items to test eviction recovery."""
    prompts: list[str] = []
    # Phase 1: 24 unique full prompts (8 systems × 3 suffixes)
    for sys_prompt in SYSTEM_PROMPTS:
        for suffix in SUFFIXES:
            prompts.append(sys_prompt + suffix)
    # Phase 2: revisits — mix of oldest, middle, newest to differentiate policies
    revisits = [0, 1, 6, 7, 12, 13, 18, 19, 0, 6, 12, 18]
    for i in revisits:
        prompts.append(prompts[i])
    return prompts


def wait_for_ready(timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/cache/stats", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Server didn't become ready")


@dataclass
class PolicyResult:
    policy: str
    total_requests: int
    cold_ttft_ms: float       # avg total time on first occurrence of each prompt
    warm_ttft_ms: float       # avg total time on revisits (phase 2)
    revisit_hit_tokens_avg: float  # avg cache_hit_tokens during phase 2
    final_hit_rate: float
    final_evictions: int
    final_utilization: float


def run_workload(policy: str) -> PolicyResult:
    env = os.environ.copy()
    env["EVICTION_POLICY"] = policy
    env["KV_CACHE_NUM_BLOCKS"] = str(NUM_BLOCKS)
    env["KV_CACHE_BLOCK_SIZE"] = str(BLOCK_SIZE)

    log_path = f"/tmp/eviction_bench_{policy}.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        ["uvicorn", "inference_server.server:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )
    try:
        wait_for_ready()
        prompts = build_workload()
        cold_count = len(SYSTEM_PROMPTS) * len(SUFFIXES)

        cold_times: list[float] = []
        warm_times: list[float] = []
        warm_hit_tokens: list[int] = []

        with httpx.Client(timeout=120.0) as client:
            for i, p in enumerate(prompts):
                t0 = time.perf_counter()
                r = client.post(f"{SERVER_URL}/generate", json={
                    "text": p, "max_tokens": 8, "stream": False, "thinking": False,
                })
                elapsed = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    print(f"  req {i} ({policy}) failed: {r.status_code} {r.text[:200]}", file=sys.stderr)
                    continue
                body = r.json()
                if i < cold_count:
                    cold_times.append(elapsed)
                else:
                    warm_times.append(elapsed)
                    warm_hit_tokens.append(body.get("cache_hit_tokens", 0))

        stats = httpx.get(f"{SERVER_URL}/cache/stats").json()

        return PolicyResult(
            policy=policy,
            total_requests=len(prompts),
            cold_ttft_ms=sum(cold_times) / max(len(cold_times), 1),
            warm_ttft_ms=sum(warm_times) / max(len(warm_times), 1),
            revisit_hit_tokens_avg=sum(warm_hit_tokens) / max(len(warm_hit_tokens), 1),
            final_hit_rate=stats["hit_rate"],
            final_evictions=stats["eviction_count"],
            final_utilization=stats["utilization"],
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def print_markdown_table(results: list[PolicyResult]) -> None:
    print()
    print(f"# Eviction Policy Comparison")
    print()
    print(f"- Workload: {results[0].total_requests} requests "
          f"({len(SYSTEM_PROMPTS)} system prompts × {len(SUFFIXES)} suffixes + 8 revisits)")
    print(f"- KV budget: {NUM_BLOCKS} blocks × {BLOCK_SIZE} tokens = {NUM_BLOCKS * BLOCK_SIZE} cacheable tokens")
    print(f"- Model: gemma (whatever MODEL_NAME is set to)")
    print()
    print("| Policy | Cold avg ms | Warm avg ms | Avg revisit hit tokens | Hit rate | Evictions | Util |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r.policy} | {r.cold_ttft_ms:.0f} | {r.warm_ttft_ms:.0f} "
              f"| {r.revisit_hit_tokens_avg:.1f} "
              f"| {r.final_hit_rate * 100:.1f}% | {r.final_evictions} "
              f"| {r.final_utilization * 100:.0f}% |")
    print()


def _bar_chart_svg(title: str, label_value_pairs: list[tuple[str, float]],
                   unit: str, lower_is_better: bool) -> str:
    """One small SVG bar chart. label_value_pairs is in policy order."""
    if not label_value_pairs:
        return ""
    width = 360
    bar_h = 28
    gap = 14
    left_pad = 140
    right_pad = 80
    top_pad = 36
    height = top_pad + len(label_value_pairs) * (bar_h + gap)
    max_v = max(v for _, v in label_value_pairs) or 1.0
    best_v = min(v for _, v in label_value_pairs) if lower_is_better \
        else max(v for _, v in label_value_pairs)
    bars = []
    bars.append(f'<text class="title" x="14" y="22">{html_lib.escape(title)}</text>')
    for i, (label, value) in enumerate(label_value_pairs):
        y = top_pad + i * (bar_h + gap)
        bar_w = max(2.0, (value / max_v) * (width - left_pad - right_pad))
        is_best = value == best_v
        fill = "#3fb950" if is_best else "#79c0ff"
        bars.append(
            f'<text x="14" y="{y + bar_h * 0.65:.0f}" class="lbl">{html_lib.escape(label)}</text>'
            f'<rect x="{left_pad}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{fill}" opacity="0.85"/>'
            f'<text x="{left_pad + bar_w + 6:.0f}" y="{y + bar_h * 0.65:.0f}" class="mono">{value:.2f}{unit}</text>'
        )
    arrow = "↓ lower better" if lower_is_better else "↑ higher better"
    bars.append(f'<text x="{width - right_pad}" y="22" class="small" text-anchor="end">{arrow}</text>')
    inner = "\n      ".join(bars)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html_lib.escape(title)}">\n'
        f'      {inner}\n    </svg>'
    )


def write_html_report(results: list[PolicyResult], out_path: pathlib.Path) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    workload_line = (
        f"{results[0].total_requests} requests "
        f"({len(SYSTEM_PROMPTS)} system prompts × {len(SUFFIXES)} suffixes + 8 revisits)"
    )
    budget_line = f"{NUM_BLOCKS} blocks × {BLOCK_SIZE} tokens = {NUM_BLOCKS * BLOCK_SIZE} cacheable tokens"
    model_line = f"MODEL_NAME={os.environ.get('MODEL_NAME', '(default)')}"

    charts = [
        _bar_chart_svg("Cold avg latency", [(r.policy, r.cold_ttft_ms) for r in results], " ms", True),
        _bar_chart_svg("Warm (revisit) avg latency", [(r.policy, r.warm_ttft_ms) for r in results], " ms", True),
        _bar_chart_svg("Avg cache-hit tokens on revisits", [(r.policy, r.revisit_hit_tokens_avg) for r in results], "", False),
        _bar_chart_svg("Final cache hit rate", [(r.policy, r.final_hit_rate * 100) for r in results], "%", False),
    ]

    rows = "\n".join(
        f"  <tr><td>{html_lib.escape(r.policy)}</td>"
        f"<td>{r.cold_ttft_ms:.0f}</td>"
        f"<td>{r.warm_ttft_ms:.0f}</td>"
        f"<td>{r.revisit_hit_tokens_avg:.1f}</td>"
        f"<td>{r.final_hit_rate * 100:.1f}%</td>"
        f"<td>{r.final_evictions}</td>"
        f"<td>{r.final_utilization * 100:.0f}%</td></tr>"
        for r in results
    )

    chart_blocks = "\n".join(f'    <div>{c}</div>' for c in charts)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Eviction policy benchmark — {timestamp}</title>
<link rel="stylesheet" href="arch.css" />
<style>
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
  .charts > div {{ background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }}
</style>
</head>
<body>
<div class="crumbs"><a href="architecture.html">← Architecture map</a> · Benchmark</div>
<h1>Eviction policy benchmark</h1>
<p class="lead">Generated {timestamp}. Compares <code>lru</code> · <code>attention_sink_lru</code> · <code>h2o</code> on the same workload.</p>

<div class="panel">
  <h3>Run config</h3>
  <ul class="compact" style="margin:0;padding-left:20px;">
    <li>Workload: {workload_line}</li>
    <li>KV budget: {budget_line}</li>
    <li>Model: {model_line}</li>
  </ul>
</div>

<h2>Results</h2>
<table>
  <tr><th>Policy</th><th>Cold avg ms</th><th>Warm avg ms</th><th>Avg revisit hit tokens</th><th>Hit rate</th><th>Evictions</th><th>Util</th></tr>
{rows}
</table>

<h2>Charts</h2>
<div class="charts">
{chart_blocks}
</div>

<p class="footnote">Green bar = best for that metric. H2O currently behaves like LRU because the backend doesn't extract attention weights yet — see <a href="arch-cache.html#eviction">cache page</a>.</p>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    results = []
    for policy in POLICIES:
        print(f"=== Running {policy} ===", file=sys.stderr)
        results.append(run_workload(policy))
        print(f"  done: hit_rate={results[-1].final_hit_rate:.2%}, "
              f"evictions={results[-1].final_evictions}", file=sys.stderr)
    print_markdown_table(results)

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = repo_root / "docs" / f"bench-eviction-{stamp}.html"
    write_html_report(results, out)
    print(f"HTML report written to {out.relative_to(repo_root)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
