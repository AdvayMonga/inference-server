"""Prometheus metrics surface (pull-based, scraped at /metrics).

Aggregate by design — NO `session_id` label. session_id is unbounded in production, and a
per-session label would explode Prometheus cardinality (one time series per session). Per-session
detail stays in `/scheduler/stats` (JSON, pull-on-demand, not stored as time series). See
DECISIONS [2026-06-13]. Only low-cardinality labels are used (outcome, HTTP method/endpoint/status).

Counters/histograms are observed at event sites (request completion, rejection, prefill chunk,
HTTP request). Gauges reflect live scheduler state and are refreshed at scrape time via
`set_scheduler_gauges()`.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# --- Counters (monotonic) ---
REQUESTS = Counter("inference_requests_total", "Requests by terminal outcome", ["outcome"])  # completed|rejected|failed
TOKENS_GENERATED = Counter("inference_tokens_generated_total", "Output tokens generated")
PREFILL_CHUNKS = Counter("inference_prefill_chunks_total", "Prefill chunks processed (chunked mode)")

# --- Histograms (seconds) ---
TTFT = Histogram("inference_ttft_seconds", "Time to first token",
                 buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10))
TPOT = Histogram("inference_tpot_seconds", "Per-output-token latency (decode)",
                 buckets=(.005, .01, .025, .05, .1, .25, .5, 1))
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "End-to-end request latency",
                            buckets=(.05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60))
HTTP_DURATION = Histogram("inference_http_request_duration_seconds", "HTTP request duration",
                          ["method", "endpoint", "status"],
                          buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5))

# --- Gauges (live scheduler state; refreshed at scrape) ---
ACTIVE_BATCH = Gauge("inference_active_batch_size", "Rows currently decoding")
PENDING_DEPTH = Gauge("inference_pending_queue_depth", "Requests waiting in the pending queue")
PREFILLING_DEPTH = Gauge("inference_prefilling_depth", "Requests mid-prefill (chunked mode)")
KV_PRESSURE = Gauge("inference_kv_pressure_ratio", "KV cache pressure (0..1)")
ACTIVE_KV_RESERVED = Gauge("inference_active_kv_reserved_tokens", "Reserved KV tokens across in-flight rows")


def record_completion(ttft_s: float, tpot_s: float, total_s: float, n_tokens: int) -> None:
    REQUESTS.labels(outcome="completed").inc()
    TOKENS_GENERATED.inc(n_tokens)
    TTFT.observe(ttft_s)
    if n_tokens > 1:
        TPOT.observe(tpot_s)
    REQUEST_LATENCY.observe(total_s)


def record_rejection(failed: bool = False) -> None:
    REQUESTS.labels(outcome="failed" if failed else "rejected").inc()


def record_prefill_chunk() -> None:
    PREFILL_CHUNKS.inc()


def set_scheduler_gauges(stats: dict) -> None:
    """Refresh live gauges from a scheduler `stats()` snapshot (called at scrape time)."""
    ACTIVE_BATCH.set(stats.get("active_size", 0))
    PENDING_DEPTH.set(stats.get("pending_depth", 0))
    PREFILLING_DEPTH.set(stats.get("prefilling_depth", 0))
    KV_PRESSURE.set(stats.get("kv_pressure", 0.0))
    ACTIVE_KV_RESERVED.set(stats.get("active_kv_reserved", 0))


def render() -> tuple[bytes, str]:
    """(body, content_type) for the /metrics response."""
    return generate_latest(), CONTENT_TYPE_LATEST
