"""Server configuration — all tunables in one place, read from env vars."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env if present (local-dev convenience). No-op in prod where the
# orchestrator (Modal, Docker, k8s) injects env vars directly.
load_dotenv()


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
    # Active-KV budget — cap on (prompt_len + max_tokens) summed across in-flight rows.
    # 0 = derive from max_batch_size * context_window (effectively unbounded).
    max_active_kv_tokens: int = 0
    # Prefill strategy: "monolithic" (one forward on admit) or "chunked" (V-A: interleave one
    # prefill chunk + one decode step per iter, kills HOL blocking). "" = derive from chunk_size
    # (back-compat: chunk_size>0 → chunked). Future strategies: mixed_batch, disaggregated.
    prefill_mode: str = ""
    # Chunked prefill — split admitting request's uncached suffix into chunks of this size.
    # 0 = disabled (monolithic prefill on admit). Typical: 256–512.
    prefill_chunk_size: int = 0
    # Batched-prefill wave planning: free-slot-multiples of the pending queue the scheduler may
    # reorder within to group similar prompt lengths. 0 = strict policy order (default; measured
    # inert below the queued regime — 83-91% of waves are K=1 there).
    wave_window_mult: int = 0
    # Cross-session prefix cache. It holds refcounts on real KV blocks, so it needs BOTH a
    # count cap (host memory for the key tuples) and a share-of-pool cap (device blocks) —
    # otherwise a warm cache starves live requests and alloc() fails on the request path.
    # Admission deadline: reject requests that have queued longer than this instead of letting
    # overload become unbounded latency. 0 disables. Tune to your TTFT SLO.
    max_queue_wait_s: float = 30.0
    prefix_cache_max_entries: int = 1024
    prefix_cache_block_fraction: float = 0.5

    # Model
    model_name: str = "google/gemma-4-E2B-it"
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    backend_name: str = ""  # explicit backend override (e.g. "custom-cuda"); empty = derive from device
    max_tokens: int = 512
    context_window: int = 8192

    # KV Cache
    kv_cache_memory_fraction: float = 0.9
    kv_cache_block_size: int = 16
    kv_cache_num_blocks: int = 256
    eviction_policy: str = "lru"  # "lru", "attention_sink_lru", "h2o"

    # Streaming
    stream_by_default: bool = False

    # Single-user optimization
    keep_model_warm: bool = True  # never unload model between requests
    compile_model: bool = False  # torch.compile the generation loop

    # Observability
    log_level: str = "INFO"
    log_format: str = "text"  # "text" (dev) or "json" (structured, for prod/Modal log aggregators)
    metrics_port: int = 9090

    # Reserved for multi-user extension (ignored for now)
    max_concurrent_sessions: int = 1
    per_session_memory_limit_mb: int = 0  # 0 = unlimited
    scheduling_policy: str = "fcfs"  # "fcfs"; future: "fair", "priority"

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            return _detect_device()
        return self.device

    @property
    def backend(self) -> str:
        """Backend name for create_backend(). Explicit BACKEND override wins; else device."""
        return self.backend_name or self.resolved_device


def load_settings() -> Settings:
    """Build Settings from environment variables, falling back to defaults."""
    return Settings(
        host=os.environ.get("HOST", Settings.host),
        port=int(os.environ.get("PORT", Settings.port)),
        max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", Settings.max_batch_size)),
        batch_timeout_ms=float(os.environ.get("BATCH_TIMEOUT_MS", Settings.batch_timeout_ms)),
        max_queue_size=int(os.environ.get("MAX_QUEUE_SIZE", Settings.max_queue_size)),
        max_active_kv_tokens=int(os.environ.get("MAX_ACTIVE_KV_TOKENS", Settings.max_active_kv_tokens)),
        prefill_mode=os.environ.get("PREFILL_MODE", Settings.prefill_mode),
        prefill_chunk_size=int(os.environ.get("PREFILL_CHUNK_SIZE", Settings.prefill_chunk_size)),
        wave_window_mult=int(os.environ.get("WAVE_WINDOW_MULT", Settings.wave_window_mult)),
        max_queue_wait_s=float(os.environ.get("MAX_QUEUE_WAIT_S", Settings.max_queue_wait_s)),
        prefix_cache_max_entries=int(os.environ.get(
            "PREFIX_CACHE_MAX_ENTRIES", Settings.prefix_cache_max_entries)),
        prefix_cache_block_fraction=float(os.environ.get(
            "PREFIX_CACHE_BLOCK_FRACTION", Settings.prefix_cache_block_fraction)),
        model_name=os.environ.get("MODEL_NAME", Settings.model_name),
        device=os.environ.get("DEVICE", Settings.device),
        backend_name=os.environ.get("BACKEND", Settings.backend_name),
        max_tokens=int(os.environ.get("MAX_TOKENS", Settings.max_tokens)),
        context_window=int(os.environ.get("CONTEXT_WINDOW", Settings.context_window)),
        kv_cache_memory_fraction=float(os.environ.get("KV_CACHE_MEMORY_FRACTION", Settings.kv_cache_memory_fraction)),
        kv_cache_block_size=int(os.environ.get("KV_CACHE_BLOCK_SIZE", Settings.kv_cache_block_size)),
        kv_cache_num_blocks=int(os.environ.get("KV_CACHE_NUM_BLOCKS", Settings.kv_cache_num_blocks)),
        eviction_policy=os.environ.get("EVICTION_POLICY", Settings.eviction_policy),
        stream_by_default=os.environ.get("STREAM_BY_DEFAULT", str(Settings.stream_by_default)).lower() == "true",
        keep_model_warm=os.environ.get("KEEP_MODEL_WARM", str(Settings.keep_model_warm)).lower() == "true",
        compile_model=os.environ.get("COMPILE_MODEL", str(Settings.compile_model)).lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", Settings.log_level),
        log_format=os.environ.get("LOG_FORMAT", Settings.log_format),
        metrics_port=int(os.environ.get("METRICS_PORT", Settings.metrics_port)),
        max_concurrent_sessions=int(os.environ.get("MAX_CONCURRENT_SESSIONS", Settings.max_concurrent_sessions)),
        per_session_memory_limit_mb=int(os.environ.get("PER_SESSION_MEMORY_LIMIT_MB", Settings.per_session_memory_limit_mb)),
        scheduling_policy=os.environ.get("SCHEDULING_POLICY", Settings.scheduling_policy),
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
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
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
