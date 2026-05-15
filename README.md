# Inference Server

LLM inference server

## Architecture

- docs/architecture.html

## Setup

- `python -m venv venv && source venv/bin/activate`
- `pip install -e ".[dev]"`
- `cp .env.example .env`
-  Downlaod and configure model via huggingface

## Configuration

- All settings via env vars in `.env` — see `.env.example` for the full list

## Run

- `uvicorn inference_server.server:app --host 0.0.0.0 --port 8000`
- Open `http://localhost:8000` for testing, dashboard