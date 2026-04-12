# Inference Server

Production LLM inference server built from scratch in Python. Features async batching, KV cache management, continuous batching, and streaming — the same problems solved by vLLM, TGI, and SGLang.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # edit with your settings
```

## Configuration

All settings are configured via environment variables. See `.env.example` for the full list.
