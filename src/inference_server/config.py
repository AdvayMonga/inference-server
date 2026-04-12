"""Server configuration — all tunables in one place, read from env vars."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Immutable server configuration loaded from environment variables."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Batching
    max_batch_size: int = 32
    batch_timeout_ms: float = 50.0
    max_queue_size: int = 1000

    # Model
    model_name: str = "google/gemma-4-E2B"
    backend: str = "mps"  # "mps", "cuda", or "cpu"
    max_tokens: int = 512
    context_window: int = 8192

    # KV Cache
    kv_cache_memory_fraction: float = 0.9  # fraction of free memory for KV cache

    # Streaming
    stream_by_default: bool = False

    # Observability
    log_level: str = "INFO"
    metrics_port: int = 9090


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to defaults."""
    return Settings(
        host=os.environ.get("HOST", Settings.host),
        port=int(os.environ.get("PORT", Settings.port)),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", Settings.max_batch_size)),
        batch_timeout_ms=float(os.environ.get("BATCH_TIMEOUT_MS", Settings.batch_timeout_ms)),
        max_queue_size=int(os.environ.get("MAX_QUEUE_SIZE", Settings.max_queue_size)),
        model_name=os.environ.get("MODEL_NAME", Settings.model_name),
        backend=os.environ.get("BACKEND", Settings.backend),
        max_tokens=int(os.environ.get("MAX_TOKENS", Settings.max_tokens)),
        context_window=int(os.environ.get("CONTEXT_WINDOW", Settings.context_window)),
        kv_cache_memory_fraction=float(os.environ.get("KV_CACHE_MEMORY_FRACTION", Settings.kv_cache_memory_fraction)),
        stream_by_default=os.environ.get("STREAM_BY_DEFAULT", str(Settings.stream_by_default)).lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", Settings.log_level),
        metrics_port=int(os.environ.get("METRICS_PORT", Settings.metrics_port)),
    )


settings = load_settings()
