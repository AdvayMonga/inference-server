"""FastAPI server — accepts text, generates LLM responses via the backend."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_server.backends import create_backend
from inference_server.batcher import BatchProcessor
from inference_server.config import settings, print_hardware_summary
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.tokenizer import Tokenizer


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    text: str
    max_tokens: int = settings.max_tokens
    stream: bool = settings.stream_by_default


class GenerateResponse(BaseModel):
    """Response body for non-streaming /generate requests."""
    text: str
    tokens_generated: int
    ttft_ms: float
    total_ms: float


@asynccontextmanager
async def lifespan(app):
    """Load model, tokenizer, cache, and batcher at startup."""
    print_hardware_summary(settings)

    backend = create_backend(settings.backend)
    tokenizer = Tokenizer(settings.model_name, settings.context_window)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, backend.load_model, settings.model_name)

    # Initialize KV cache manager and attach to backend
    cache_manager = CacheManager(
        num_blocks=1024,
        block_size=settings.kv_cache_block_size,
        eviction_policy=settings.eviction_policy,
    )
    backend.set_cache_manager(cache_manager)

    batcher = BatchProcessor(backend)
    batcher.start()

    app.state.backend = backend
    app.state.tokenizer = tokenizer
    app.state.batcher = batcher
    app.state.cache_manager = cache_manager

    yield

    await batcher.stop()


app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


async def event_stream(
    backend, tokenizer, token_ids: list[int], max_tokens: int
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks — one per generated token."""
    loop = asyncio.get_event_loop()
    first_token = True
    start_time = time.perf_counter()

    queue: asyncio.Queue = asyncio.Queue()

    def _run_stream():
        for token_id in backend.stream(token_ids, max_tokens):
            text = tokenizer.decode_token(token_id)
            queue.put_nowait(text)
        queue.put_nowait(None)

    loop.run_in_executor(None, _run_stream)

    while True:
        text = await queue.get()
        if text is None:
            break

        if first_token:
            ttft = (time.perf_counter() - start_time) * 1000
            yield f"data: {{\"ttft_ms\": {ttft:.1f}}}\n\n"
            first_token = False

        yield f"data: {text}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text — batched for non-streaming, direct for streaming."""
    backend = app.state.backend
    tokenizer = app.state.tokenizer
    batcher = app.state.batcher
    loop = asyncio.get_event_loop()

    token_ids = await loop.run_in_executor(None, tokenizer.encode_chat, request.text)

    if request.stream:
        return StreamingResponse(
            event_stream(backend, tokenizer, token_ids, request.max_tokens),
            media_type="text/event-stream",
        )

    # Single-user: call generate() directly for cache support
    # Batcher is available for multi-user extension
    start_time = time.perf_counter()
    generated_ids = await loop.run_in_executor(
        None, backend.generate, token_ids, request.max_tokens
    )
    total_time = time.perf_counter() - start_time

    output_text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)

    return GenerateResponse(
        text=output_text,
        tokens_generated=len(generated_ids),
        ttft_ms=0,
        total_ms=total_time * 1000,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Return KV cache statistics — useful for monitoring and the web UI."""
    return app.state.cache_manager.hit_rate_info
