"""FastAPI server — accepts text, generates LLM responses via the backend."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference_server.backends import create_backend
from inference_server.batcher import BatchProcessor
from inference_server.config import settings
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
    """Load model, tokenizer, and batcher at startup."""
    backend = create_backend(settings.backend)
    tokenizer = Tokenizer(settings.model_name, settings.context_window)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, backend.load_model, settings.model_name)

    batcher = BatchProcessor(backend)
    batcher.start()

    app.state.backend = backend
    app.state.tokenizer = tokenizer
    app.state.batcher = batcher

    yield

    await batcher.stop()


app = FastAPI(lifespan=lifespan)


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

    # Streaming bypasses the batcher (needs per-token delivery)
    if request.stream:
        return StreamingResponse(
            event_stream(backend, tokenizer, token_ids, request.max_tokens),
            media_type="text/event-stream",
        )

    # Non-streaming goes through the batcher for throughput
    start_time = time.perf_counter()
    generated_ids = await batcher.submit(token_ids, request.max_tokens)
    total_time = time.perf_counter() - start_time

    output_text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)

    return GenerateResponse(
        text=output_text,
        tokens_generated=len(generated_ids),
        ttft_ms=0,
        total_ms=total_time * 1000,
    )
