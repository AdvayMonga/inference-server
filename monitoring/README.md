# Monitoring

The server exposes Prometheus metrics at **`/metrics`** (pull-based, scrapable). Metrics are
**aggregate** — no `session_id` label (it's unbounded → cardinality blowup); per-session detail
lives in `/scheduler/stats` (JSON).

- `prometheus.yml` — scrape config (point `targets` at the server host:port).
- `grafana_dashboard.json` — import into Grafana (TTFT/TPOT percentiles, throughput, request
  rate by outcome, batch/queue depth, KV pressure).

The full docker-compose stack (Prometheus + Grafana wired together) lands in Phase 10.

## Metrics

| metric | type | meaning |
|---|---|---|
| `inference_requests_total{outcome}` | counter | requests by outcome (completed/rejected/failed) |
| `inference_tokens_generated_total` | counter | output tokens generated |
| `inference_prefill_chunks_total` | counter | prefill chunks processed (chunked mode) |
| `inference_ttft_seconds` | histogram | time to first token |
| `inference_tpot_seconds` | histogram | per-output-token latency (decode) |
| `inference_request_latency_seconds` | histogram | end-to-end request latency |
| `inference_http_request_duration_seconds{method,endpoint,status}` | histogram | HTTP timing (middleware) |
| `inference_active_batch_size` | gauge | rows currently decoding |
| `inference_pending_queue_depth` | gauge | requests waiting to be admitted |
| `inference_prefilling_depth` | gauge | requests mid-prefill (chunked) |
| `inference_kv_pressure_ratio` | gauge | KV cache pressure (0..1) |
| `inference_active_kv_reserved_tokens` | gauge | reserved KV tokens across in-flight rows |
