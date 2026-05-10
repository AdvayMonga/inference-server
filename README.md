# Inference Server

LLM inference server built from scratch — paged KV cache, continuous batching, fair scheduling, backpressure. The engine layer of a vLLM-class system.

## Architecture

Visual map of every component, how they talk to each other, and what state each one owns:

- **[`docs/architecture.html`](docs/architecture.html)** — top-level overview. Click any block to drill into its detail page.
- [`docs/arch-server.html`](docs/arch-server.html) — HTTP endpoints, tokenizer, SSE streaming, load simulator, dashboard
- [`docs/arch-scheduler.html`](docs/arch-scheduler.html) — queues, per-tick loop, FCFS / Fair (VTC) policies, backpressure gates, chunked prefill
- [`docs/arch-cache.html`](docs/arch-cache.html) — paged blocks, radix tree, exact-match index, eviction policies
- [`docs/arch-backend.html`](docs/arch-backend.html) — InferenceBackend interface, prefill / decode primitives, hardware detection

Open `docs/architecture.html` in a browser. The diagrams are real SVG, so text stays sharp at any zoom.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Configuration

All settings are configured via environment variables. See `.env.example` for the full list.
