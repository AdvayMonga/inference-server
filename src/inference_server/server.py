"""FastAPI server — accepts text, generates LLM responses via the backend."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from inference_server import prometheus_metrics
from inference_server.backends import create_backend
from inference_server.config import settings, print_hardware_summary
from inference_server.logging_config import setup_logging
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.sampling import SamplingParams
from inference_server.scheduler import (
    ContinuousBatchScheduler,
    QueueFullError,
    ScheduledRequest,
)
from inference_server.scheduling_policy import create_scheduling_policy
from inference_server.openai_shim import router as openai_router
from inference_server.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    text: str
    max_tokens: int = settings.max_tokens
    stream: bool = settings.stream_by_default
    thinking: bool = True
    session_id: str = "default"
    priority: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


class GenerateResponse(BaseModel):
    """Response body for non-streaming /generate requests."""
    text: str
    tokens_generated: int
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    cache_hit_tokens: int


@asynccontextmanager
async def lifespan(app):
    """Load model, tokenizer, cache, and batcher at startup."""
    setup_logging(settings.log_format, getattr(logging, settings.log_level.upper(), logging.INFO))
    app.state.ready = False
    print_hardware_summary(settings)

    backend = create_backend(settings.backend)
    tokenizer = Tokenizer(settings.model_name, settings.context_window)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, backend.load_model, settings.model_name)

    layer_shapes = backend.kv_shape_per_layer() if hasattr(backend, "kv_shape_per_layer") else None
    kv_dtype = backend.kv_dtype if hasattr(backend, "kv_dtype") else None
    cache_manager = CacheManager(
        num_blocks=settings.kv_cache_num_blocks,
        block_size=settings.kv_cache_block_size,
        eviction_policy=settings.eviction_policy,
        layer_shapes=layer_shapes,
        device=str(getattr(backend, "device", "cpu")),
        dtype=kv_dtype,
    )
    backend.set_cache_adapter(cache_manager)

    scheduler = ContinuousBatchScheduler(
        backend,
        max_batch_size=settings.max_batch_size,
        max_queue_size=settings.max_queue_size,
        max_active_kv_tokens=settings.max_active_kv_tokens,
        prefill_chunk_size=settings.prefill_chunk_size,
        prefill_mode=settings.prefill_mode or None,
        wave_window_mult=settings.wave_window_mult,
        max_queue_wait_s=settings.max_queue_wait_s,
        policy=create_scheduling_policy(settings.scheduling_policy),
    )
    scheduler.start()

    app.state.backend = backend
    app.state.tokenizer = tokenizer
    app.state.model_name = settings.model_name
    app.state.scheduler = scheduler
    app.state.cache_adapter = cache_manager
    app.state.ready = True

    yield

    app.state.ready = False
    await scheduler.stop()


app = FastAPI(lifespan=lifespan)
app.include_router(openai_router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """Liveness — process is up. Cheap; touches no state."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness — 200 once model + scheduler are up, 503 otherwise."""
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


async def event_stream(
    req: ScheduledRequest, tokenizer,
    prompt_token_count: int, start_time: float,
) -> AsyncGenerator[str, None]:
    """SSE stream sourced from an already-enqueued request's token_queue."""
    first = True
    try:
        while True:
            tok_id = await req.token_queue.get()
            if tok_id is None:
                break
            text = tokenizer.decode_token(tok_id)
            if first:
                ttft = (time.perf_counter() - start_time) * 1000
                meta = {
                    "ttft_ms": round(ttft, 1),
                    "prompt_tokens": prompt_token_count,
                    "cache_hit_tokens": req.cache_hit_tokens,
                }
                yield f"data: {json.dumps(meta)}\n\n"
                first = False
            if text:
                yield f"data: {text}\n\n"
        if req.future.done() and req.future.exception():
            raise req.future.exception()  # type: ignore[misc]
    finally:
        yield "data: [DONE]\n\n"


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text — routes through the continuous-batch scheduler."""
    scheduler = app.state.scheduler
    tokenizer = app.state.tokenizer
    loop = asyncio.get_running_loop()

    token_ids = await loop.run_in_executor(None, tokenizer.encode_chat, request.text, request.thinking)

    sampling = SamplingParams(
        temperature=request.temperature, top_p=request.top_p, top_k=request.top_k,
    )

    if request.stream:
        token_queue: asyncio.Queue = asyncio.Queue()
        req = ScheduledRequest(
            token_ids=token_ids, max_tokens=request.max_tokens,
            session_id=request.session_id, future=loop.create_future(),
            token_queue=token_queue, priority=request.priority,
            sampling=sampling,
        )
        try:
            scheduler.enqueue(req)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e))
        start_time = time.perf_counter()
        return StreamingResponse(
            event_stream(req, tokenizer, len(token_ids), start_time),
            media_type="text/event-stream",
        )

    req = ScheduledRequest(
        token_ids=token_ids, max_tokens=request.max_tokens,
        session_id=request.session_id, future=loop.create_future(),
        priority=request.priority, sampling=sampling,
    )
    start_time = time.perf_counter()
    try:
        generated_ids = await scheduler.submit(req)
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e))
    total_time = time.perf_counter() - start_time

    output_text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)

    return GenerateResponse(
        text=output_text,
        tokens_generated=len(generated_ids),
        ttft_ms=0,
        total_ms=total_time * 1000,
        prompt_tokens=len(token_ids),
        cache_hit_tokens=req.cache_hit_tokens,
    )


@app.get("/cache/stats")
async def cache_stats():
    """KV cache statistics. The custom backend caches through its own PrefixCache + block
    pools rather than the HF-format CacheManager, so report that when it is the one serving —
    otherwise the endpoint silently describes a cache the engine is not using."""
    backend = getattr(app.state, "backend", None)
    prefix_cache = getattr(backend, "prefix_cache", None)
    if prefix_cache is not None:
        return {"backend": type(backend).__name__, "prefix_cache": prefix_cache.stats()}
    return app.state.cache_adapter.hit_rate_info


@app.get("/scheduler/stats")
async def scheduler_stats():
    """Return scheduler depth, throughput counters, and rejection count."""
    return app.state.scheduler.stats()


@app.get("/metrics")
async def metrics():
    """Prometheus exposition (pull-based, scrapable). Aggregate metrics; per-session detail is
    in /scheduler/stats. Refreshes live gauges from the scheduler before rendering."""
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        prometheus_metrics.set_scheduler_gauges(scheduler.stats())
    body, content_type = prometheus_metrics.render()
    return Response(content=body, media_type=content_type)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Time every HTTP request → per-endpoint histogram + structured log line. Uses the matched
    route template (not the raw path) as the label to keep cardinality low."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)
    if endpoint != "/metrics":  # don't let scrapes pollute their own histogram
        prometheus_metrics.HTTP_DURATION.labels(
            method=request.method, endpoint=endpoint, status=response.status_code,
        ).observe(duration)
        logger.info("http_request", extra={
            "method": request.method, "endpoint": endpoint,
            "status": response.status_code, "duration_ms": round(duration * 1000, 2),
        })
    return response


