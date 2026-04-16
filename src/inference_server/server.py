"""FastAPI server — accepts text, generates LLM responses via the backend."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from inference_server.backends import create_backend
from inference_server.batcher import BatchProcessor
from inference_server.config import settings, print_hardware_summary
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.tokenizer import Tokenizer

DEFAULT_PROMPTS = [
    "Explain how a computer works in simple terms",
    "What is the capital of France?",
    "Write a short poem about the ocean",
    "Summarize the theory of relativity in three sentences",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis",
    "What happened during the French Revolution?",
    "Explain recursion to a five year old",
]


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


class SimulateRequest(BaseModel):
    """Request body for /simulate/start."""
    num_users: int = 4
    max_tokens: int = 64


@dataclass
class SimulationState:
    """Tracks a running background simulation."""
    running: bool = False
    num_users: int = 0
    tasks: list[asyncio.Task] = field(default_factory=list)
    requests_completed: int = 0
    total_tokens: int = 0
    ttft_sum: float = 0.0
    tpot_sum: float = 0.0
    tpot_count: int = 0
    start_time: float = 0.0


@asynccontextmanager
async def lifespan(app):
    """Load model, tokenizer, cache, and batcher at startup."""
    print_hardware_summary(settings)

    backend = create_backend(settings.backend)
    tokenizer = Tokenizer(settings.model_name, settings.context_window)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, backend.load_model, settings.model_name)

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
    app.state.simulation = SimulationState()

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
    """Yield SSE-formatted chunks, skipping thinking tokens."""
    loop = asyncio.get_event_loop()
    first_visible_token = True
    in_thinking = False
    start_time = time.perf_counter()
    accumulated = ""

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

        accumulated += text

        # Detect thinking start/end
        if "<think>" in accumulated and not in_thinking:
            in_thinking = True
        if "</think>" in accumulated and in_thinking:
            in_thinking = False
            # Strip everything up to and including </think>
            idx = accumulated.index("</think>") + len("</think>")
            accumulated = accumulated[idx:].lstrip()
            if not accumulated:
                continue

        if in_thinking:
            continue

        # Emit visible text
        if accumulated:
            if first_visible_token:
                ttft = (time.perf_counter() - start_time) * 1000
                yield f"data: {{\"ttft_ms\": {ttft:.1f}}}\n\n"
                first_visible_token = False

            yield f"data: {accumulated}\n\n"
            accumulated = ""

    yield "data: [DONE]\n\n"


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate text — direct call for non-streaming, SSE for streaming."""
    backend = app.state.backend
    tokenizer = app.state.tokenizer
    loop = asyncio.get_event_loop()

    token_ids = await loop.run_in_executor(None, tokenizer.encode_chat, request.text)

    if request.stream:
        return StreamingResponse(
            event_stream(backend, tokenizer, token_ids, request.max_tokens),
            media_type="text/event-stream",
        )

    start_time = time.perf_counter()
    generated_ids = await loop.run_in_executor(
        None, backend.generate, token_ids, request.max_tokens
    )
    total_time = time.perf_counter() - start_time

    raw_text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)
    output_text = tokenizer.strip_thinking(raw_text)

    return GenerateResponse(
        text=output_text,
        tokens_generated=len(generated_ids),
        ttft_ms=0,
        total_ms=total_time * 1000,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Return KV cache statistics."""
    return app.state.cache_manager.hit_rate_info


# --- Load Simulator ---

async def _simulated_user(user_id: int, sim: SimulationState, max_tokens: int, port: int):
    """One simulated user — loops sending non-streaming requests until cancelled."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        prompt_idx = user_id
        while True:
            prompt = DEFAULT_PROMPTS[prompt_idx % len(DEFAULT_PROMPTS)]
            prompt_idx += 1

            try:
                t0 = time.perf_counter()
                resp = await client.post(
                    f"http://127.0.0.1:{port}/generate",
                    json={"text": prompt, "max_tokens": max_tokens, "stream": False},
                )
                total = (time.perf_counter() - t0) * 1000

                data = resp.json()
                tokens = data.get("tokens_generated", 0)
                ttft = data.get("total_ms", total)  # approximate TTFT as total for non-streaming
                tpot = total / max(tokens, 1)

                sim.requests_completed += 1
                sim.total_tokens += tokens
                sim.ttft_sum += ttft
                sim.tpot_sum += tpot
                sim.tpot_count += 1

            except asyncio.CancelledError:
                return
            except Exception:
                pass

            await asyncio.sleep(0.2)


@app.post("/simulate/start")
async def simulate_start(request: SimulateRequest):
    """Start background traffic simulation."""
    sim = app.state.simulation
    if sim.running:
        return {"error": "Simulation already running"}, 409

    sim.running = True
    sim.num_users = request.num_users
    sim.requests_completed = 0
    sim.total_tokens = 0
    sim.ttft_sum = 0.0
    sim.tpot_sum = 0.0
    sim.tpot_count = 0
    sim.start_time = time.time()
    sim.tasks = []

    for i in range(request.num_users):
        task = asyncio.create_task(
            _simulated_user(i, sim, request.max_tokens, settings.port)
        )
        sim.tasks.append(task)

    return {"status": "started", "num_users": request.num_users}


@app.post("/simulate/stop")
async def simulate_stop():
    """Stop background traffic simulation."""
    sim = app.state.simulation
    if not sim.running:
        return {"status": "not_running"}

    for task in sim.tasks:
        task.cancel()
    await asyncio.gather(*sim.tasks, return_exceptions=True)

    sim.running = False
    sim.tasks = []
    return {"status": "stopped"}


@app.get("/simulate/status")
async def simulate_status():
    """Return live simulation stats."""
    sim = app.state.simulation
    elapsed = time.time() - sim.start_time if sim.running and sim.start_time > 0 else 0
    return {
        "running": sim.running,
        "num_users": sim.num_users,
        "requests_completed": sim.requests_completed,
        "total_tokens": sim.total_tokens,
        "avg_ttft_ms": sim.ttft_sum / max(sim.requests_completed, 1),
        "avg_tpot_ms": sim.tpot_sum / max(sim.tpot_count, 1),
        "tokens_per_sec": sim.total_tokens / max(elapsed, 0.001),
        "elapsed_s": elapsed,
    }
