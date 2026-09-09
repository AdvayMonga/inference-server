# Monitoring

The server exposes Prometheus metrics at **`/metrics`** (pull-based, scrapable). Metrics are
**aggregate** — no `session_id` label (it's unbounded → cardinality blowup); per-session detail
lives in `/scheduler/stats` (JSON).

## Running it

```bash
docker compose -f monitoring/docker-compose.yml up -d
open http://localhost:3000     # Grafana, admin/admin, dashboard already provisioned
open http://localhost:9090     # Prometheus, check the target reads UP
```

Start the server first, separately. It is deliberately not part of this stack: the thing being
measured and the thing doing the measuring stay in different processes.

| file | role |
|---|---|
| `docker-compose.yml` | Prometheus + Grafana. Scrapes the host, does not run the server. |
| `prometheus.yml` | Scrape config. Target is `host.docker.internal:8000`; use `localhost:8000` when running Prometheus straight on the host, or the Modal hostname plus `scheme: https` when deployed. |
| `grafana_dashboard.json` | TTFT/TPOT percentiles, throughput, request rate by outcome, batch/queue depth, KV pressure. |
| `grafana/provisioning/` | Wires the Prometheus datasource and auto-loads the dashboard, so there is nothing to import by hand. |

Override the Grafana login with `GRAFANA_USER` / `GRAFANA_PASSWORD` in the environment.
Both Prometheus and Grafana keep their state in named volumes, so a restart does not lose history.

This is the only place aggregate numbers live. The chat page shows the request you just sent
and nothing more, on purpose — see `docs/arch-server.html`.

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
