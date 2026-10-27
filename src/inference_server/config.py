"""Server configuration — all tunables in one place, read from env vars."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from inference_server import profiles

# Load .env if present (local-dev convenience). No-op in prod where the
# orchestrator (Modal, Docker, k8s) injects env vars directly.
load_dotenv()
# The launch profile (profiles.toml) over the launch env, before anything reads it.
profiles.apply()


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
    # Rows decoded together; the KV pools are sized from free memory. 256: H200 2026-10-19 with fp8 experts, saturated
    # 4.6 req/s at 128 (clients beyond 128 wait in the queue and miss TTFT), 6.3 at 256, 6.2 at 512.
    max_batch_size: int = 1024      # 256 -> 1024 (2026-10-21): long answers 10.6 -> 22.5 req/s, saturated 8.3 -> 10.4
    batch_timeout_ms: float = 50.0
    max_queue_size: int = 1000
    # Active-KV budget — cap on (prompt_len + max_tokens) summed across in-flight rows.
    # 0 = derive from max_batch_size * context_window (effectively unbounded).
    max_active_kv_tokens: int = 0
    # Prefill strategy: "monolithic" (one forward on admit), "chunked" (V-A: one request's chunk + one decode
    # step per iter), "batched" (a wave of whole prompts, then decode) or "mixed" (every waiting prefill advances
    # by up to prefill_step_tokens in ONE forward per iter, then a decode step: prefill never stalls decode for
    # more than one bounded chunk, and is not one request at a time). "" = derive: chunk_size>0 → chunked, else
    # mixed on custom-* backends (H200, 2026-10-16: saturated 1.11 → 1.87 req/s with async decode), else monolithic.
    prefill_mode: str = ""
    # Mixed mode: the most prefill tokens fed between two decode steps (all rows together; H200 2026-10-17 with
    # FlashInfer: 8192 gives long_prompt 9.0 vs 8.4 req/s at 2048, saturated unchanged), and the most rows one step
    # may carry (0 = no cap; a cap of 32 lost saturated 3.58 -> 2.35 req/s, TPOT p99 over the limit at 128 clients).
    prefill_step_tokens: int = 2048    # 8192 -> 2048 (2026-10-26): bursty TPOT p99 26.7 -> 19.8 ms, long prompt and saturated unchanged
    prefill_step_rows: int = 0
    # Decode rows join the packed prefill forward (one step for chunks and tokens). Gate PASS 2026-10-18; its
    # throughput is not measured yet, so off until it is (PREFILL_JOINT=1).
    prefill_joint: bool = False
    # Batched prefill: the most (right-padded) tokens one wave may carry, rows x the longest prompt. Bounds the
    # activation peak, which the KV pools are sized around (a wave of long prompts OOMed with no cap). 0 = none.
    max_prefill_tokens: int = 32768
    # Decode step n launched with step n-1's tokens still on the device; n-1's tokens processed while n runs
    # (H200, 2026-10-16: single_stream TPOT p99 9.5 → 9.24 ms, decode step at batch 128 33.9 → 27.8 ms).
    async_decode: bool = True
    # The backend and scheduler in a child process (engine_process.py), HTTP and streams in this one: no shared GIL.
    engine_process: bool = True      # on since 2026-10-21: long_prompt 43 -> 50 req/s alone; off = one process, one GIL
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
    # "radix" shares at every block boundary (a shared system prompt with different tails hits);
    # "dict" only matches complete previously-stored prompts. Kept switchable for A/B.
    prefix_cache_impl: str = "radix"
    prefix_cache_max_entries: int = 1024
    prefix_cache_block_fraction: float = 0.9   # cached blocks are reclaimed on demand (BlockPool.alloc)

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    backend_name: str = ""  # custom-cuda | custom-mps | custom-cpu; empty = custom-<device>
    max_tokens: int = 512
    context_window: int = 8192

    # KV Cache
    kv_cache_memory_fraction: float = 0.9

    # Streaming
    stream_by_default: bool = False

    # Observability
    log_level: str = "INFO"
    log_format: str = "text"  # "text" (dev) or "json" (structured, for prod/Modal log aggregators)
    metrics_port: int = 9090
    # Per-request telemetry rows (SQLite, one file per run). Empty = off, no filesystem writes.
    telemetry_dir: str = ""
    # Engine event timeline (JSONL, one file per run). Empty = off.
    timeline_dir: str = ""

    scheduling_policy: str = "fair"  # "fair" (VTC: least-served session first) or "fcfs"

    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device == "auto":
            return _detect_device()
        return self.device

    @property
    def backend(self) -> str:
        """Backend name for create_backend(). Explicit BACKEND override wins; else custom-<device>."""
        return self.backend_name or f"custom-{self.resolved_device}"


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
        prefill_step_tokens=int(os.environ.get("PREFILL_STEP_TOKENS", Settings.prefill_step_tokens)),
        prefill_step_rows=int(os.environ.get("PREFILL_STEP_ROWS", Settings.prefill_step_rows)),
        prefill_joint=os.environ.get("PREFILL_JOINT", str(Settings.prefill_joint)).lower() in ("1", "true"),
        prefill_chunk_size=int(os.environ.get("PREFILL_CHUNK_SIZE", Settings.prefill_chunk_size)),
        max_prefill_tokens=int(os.environ.get("MAX_PREFILL_TOKENS", Settings.max_prefill_tokens)),
        wave_window_mult=int(os.environ.get("WAVE_WINDOW_MULT", Settings.wave_window_mult)),
        max_queue_wait_s=float(os.environ.get("MAX_QUEUE_WAIT_S", Settings.max_queue_wait_s)),
        async_decode=os.environ.get("ASYNC_DECODE", str(Settings.async_decode)).lower() in ("1", "true"),
        engine_process=os.environ.get("ENGINE_PROCESS", str(Settings.engine_process)).lower() in ("1", "true"),
        prefix_cache_impl=os.environ.get("PREFIX_CACHE_IMPL", Settings.prefix_cache_impl),
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
        stream_by_default=os.environ.get("STREAM_BY_DEFAULT", str(Settings.stream_by_default)).lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", Settings.log_level),
        log_format=os.environ.get("LOG_FORMAT", Settings.log_format),
        metrics_port=int(os.environ.get("METRICS_PORT", Settings.metrics_port)),
        telemetry_dir=os.environ.get("TELEMETRY_DIR", Settings.telemetry_dir),
        timeline_dir=os.environ.get("TIMELINE_DIR", Settings.timeline_dir),
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
        print("  CPU only — no GPU acceleration")

    print(f"  KV Cache:  {settings.kv_cache_memory_fraction * 100:.0f}% of free memory")
    print("=" * 50)


settings = load_settings()
