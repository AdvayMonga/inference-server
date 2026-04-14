"""Server configuration — all tunables in one place, read from env vars."""

import os
from dataclasses import dataclass


def _detect_device() -> str:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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
    model_name: str = "google/gemma-4-E2B-it"
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    max_tokens: int = 512
    context_window: int = 8192

    # KV Cache
    kv_cache_memory_fraction: float = 0.9
    kv_cache_block_size: int = 16
    eviction_policy: str = "lru"  # "lru", "attention_sink_lru", "h2o"

    # Streaming
    stream_by_default: bool = False

    # Single-user optimization
    keep_model_warm: bool = True  # never unload model between requests
    compile_model: bool = False  # torch.compile the generation loop

    # Observability
    log_level: str = "INFO"
    metrics_port: int = 9090

    # Reserved for multi-user extension (ignored for now)
    max_concurrent_sessions: int = 1
    per_session_memory_limit_mb: int = 0  # 0 = unlimited
    scheduler_policy: str = "single_user"  # "single_user", "fair", "priority"

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            return _detect_device()
        return self.device

    @property
    def backend(self) -> str:
        """Map device to backend name for create_backend()."""
        return self.resolved_device


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to defaults."""
    return Settings(
        host=os.environ.get("HOST", Settings.host),
        port=int(os.environ.get("PORT", Settings.port)),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", Settings.max_batch_size)),
        batch_timeout_ms=float(os.environ.get("BATCH_TIMEOUT_MS", Settings.batch_timeout_ms)),
        max_queue_size=int(os.environ.get("MAX_QUEUE_SIZE", Settings.max_queue_size)),
        model_name=os.environ.get("MODEL_NAME", Settings.model_name),
        device=os.environ.get("DEVICE", Settings.device),
        max_tokens=int(os.environ.get("MAX_TOKENS", Settings.max_tokens)),
        context_window=int(os.environ.get("CONTEXT_WINDOW", Settings.context_window)),
        kv_cache_memory_fraction=float(os.environ.get("KV_CACHE_MEMORY_FRACTION", Settings.kv_cache_memory_fraction)),
        kv_cache_block_size=int(os.environ.get("KV_CACHE_BLOCK_SIZE", Settings.kv_cache_block_size)),
        eviction_policy=os.environ.get("EVICTION_POLICY", Settings.eviction_policy),
        stream_by_default=os.environ.get("STREAM_BY_DEFAULT", str(Settings.stream_by_default)).lower() == "true",
        keep_model_warm=os.environ.get("KEEP_MODEL_WARM", str(Settings.keep_model_warm)).lower() == "true",
        compile_model=os.environ.get("COMPILE_MODEL", str(Settings.compile_model)).lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", Settings.log_level),
        metrics_port=int(os.environ.get("METRICS_PORT", Settings.metrics_port)),
        max_concurrent_sessions=int(os.environ.get("MAX_CONCURRENT_SESSIONS", Settings.max_concurrent_sessions)),
        per_session_memory_limit_mb=int(os.environ.get("PER_SESSION_MEMORY_LIMIT_MB", Settings.per_session_memory_limit_mb)),
        scheduler_policy=os.environ.get("SCHEDULER_POLICY", Settings.scheduler_policy),
    )


def print_hardware_summary(settings: Settings) -> None:
    """Print startup hardware summary."""
    import torch

    device = settings.resolved_device
    print("=" * 50)
    print("HARDWARE SUMMARY")
    print("=" * 50)
    print(f"  Device:    {device}")
    print(f"  Backend:   {settings.backend}")
    print(f"  Model:     {settings.model_name}")

    if device == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU Mem:   {mem:.1f} GB")
    elif device == "mps":
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        mem_gb = int(result.stdout.strip()) / 1e9
        print(f"  Unified Memory: {mem_gb:.0f} GB (shared CPU/GPU)")
    else:
        print(f"  CPU only — no GPU acceleration")

    print(f"  KV Cache:  {settings.kv_cache_memory_fraction * 100:.0f}% of free memory")
    print(f"  Eviction:  {settings.eviction_policy}")
    print(f"  Compile:   {settings.compile_model}")
    print("=" * 50)


settings = load_settings()
