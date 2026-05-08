"""Rolling-window metrics aggregator for scheduler observability.

Tracks per-request TTFT / TPOT / total latency and throughput counters in a
60-second sliding window. Exposes p50/p95/p99 percentiles via snapshot().
"""

import threading
import time
from collections import deque


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class MetricsTracker:
    """Time-windowed rolling aggregates. Thread-safe; pruned on read."""

    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self._lock = threading.Lock()
        self._ttft: deque[tuple[float, float]] = deque()
        self._tpot: deque[tuple[float, float]] = deque()
        self._total: deque[tuple[float, float]] = deque()
        self._completions: deque[tuple[float, int]] = deque()  # (ts, n_tokens)

    def record(self, ttft: float, tpot: float, total: float, n_tokens: int) -> None:
        now = time.time()
        with self._lock:
            self._ttft.append((now, ttft))
            if n_tokens > 1:
                self._tpot.append((now, tpot))
            self._total.append((now, total))
            self._completions.append((now, n_tokens))

    def _prune(self, dq: deque, cutoff: float) -> None:
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def snapshot(self) -> dict:
        now = time.time()
        cutoff = now - self.window
        with self._lock:
            for dq in (self._ttft, self._tpot, self._total, self._completions):
                self._prune(dq, cutoff)
            ttft_vals = sorted(v for _, v in self._ttft)
            tpot_vals = sorted(v for _, v in self._tpot)
            total_vals = sorted(v for _, v in self._total)
            n_completions = len(self._completions)
            n_tokens = sum(n for _, n in self._completions)

        def pct(vals: list[float]) -> dict:
            return {
                "p50": round(_percentile(vals, 0.50) * 1000, 2),
                "p95": round(_percentile(vals, 0.95) * 1000, 2),
                "p99": round(_percentile(vals, 0.99) * 1000, 2),
            }

        return {
            "window_seconds": self.window,
            "ttft_ms": pct(ttft_vals),
            "tpot_ms": pct(tpot_vals),
            "total_ms": pct(total_vals),
            "throughput": {
                "req_per_s": round(n_completions / self.window, 3),
                "tok_per_s": round(n_tokens / self.window, 2),
                "samples": n_completions,
            },
        }
