"""FastAPI server — accepts text, generates LLM responses via the backend."""

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from inference_server.backends import create_backend
from inference_server.config import settings, print_hardware_summary
from inference_server.kv_cache.cache_manager import CacheManager
from inference_server.scheduler import (
    ContinuousBatchScheduler,
    QueueFullError,
    ScheduledRequest,
)
from inference_server.scheduling_policy import create_scheduling_policy
from inference_server.simulator_prompts import MAX_TOKENS_RANGE, PROMPT_MIX
from inference_server.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def _sample_prompt_and_max_tokens(rng: random.Random, max_tokens_cap: int) -> tuple[str, int, str]:
    """Weighted pick from PROMPT_MIX, then random max_tokens within the bucket's range."""
    r = rng.random()
    cumulative = 0.0
    for bucket_name, prompts, weight in PROMPT_MIX:
        cumulative += weight
        if r <= cumulative:
            prompt = rng.choice(prompts)
            lo, hi = MAX_TOKENS_RANGE[bucket_name]
            mt = rng.randint(lo, min(hi, max_tokens_cap))
            return prompt, mt, bucket_name
    # Fallback (weights don't sum to 1)
    prompt = rng.choice(PROMPT_MIX[0][1])
    return prompt, max_tokens_cap, PROMPT_MIX[0][0]


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    text: str
    max_tokens: int = settings.max_tokens
    stream: bool = settings.stream_by_default
    thinking: bool = True
    session_id: str = "default"
    priority: int = 0


class GenerateResponse(BaseModel):
    """Response body for non-streaming /generate requests."""
    text: str
    tokens_generated: int
    ttft_ms: float
    total_ms: float
    prompt_tokens: int
    cache_hit_tokens: int


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
        num_blocks=settings.kv_cache_num_blocks,
        block_size=settings.kv_cache_block_size,
        eviction_policy=settings.eviction_policy,
    )
    backend.set_cache_adapter(cache_manager)

    scheduler = ContinuousBatchScheduler(
        backend,
        max_batch_size=settings.max_batch_size,
        max_queue_size=settings.max_queue_size,
        max_active_kv_tokens=settings.max_active_kv_tokens,
        prefill_chunk_size=settings.prefill_chunk_size,
        policy=create_scheduling_policy(settings.scheduling_policy),
    )
    scheduler.start()

    app.state.backend = backend
    app.state.tokenizer = tokenizer
    app.state.scheduler = scheduler
    app.state.cache_adapter = cache_manager
    app.state.simulation = SimulationState()

    yield

    await scheduler.stop()


app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


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

    if request.stream:
        token_queue: asyncio.Queue = asyncio.Queue()
        req = ScheduledRequest(
            token_ids=token_ids, max_tokens=request.max_tokens,
            session_id=request.session_id, future=loop.create_future(),
            token_queue=token_queue, priority=request.priority,
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
        priority=request.priority,
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
    """Return KV cache statistics."""
    return app.state.cache_adapter.hit_rate_info


@app.get("/scheduler/stats")
async def scheduler_stats():
    """Return scheduler depth, throughput counters, and rejection count."""
    return app.state.scheduler.stats()


# --- Load Simulator ---

async def _simulated_user(user_id: int, sim: SimulationState, max_tokens: int, port: int):
    """One simulated user — loops sending streaming requests until cancelled.

    Per request: picks weighted from PROMPT_MIX (short/medium/long), draws random
    max_tokens from the bucket's range, and jitters think-time between requests.
    Per-user RNG seeded by user_id so reruns are reproducible.
    """
    rng = random.Random(user_id)
    async with httpx.AsyncClient(timeout=300.0) as client:
        while True:
            prompt, mt, _bucket = _sample_prompt_and_max_tokens(rng, max_tokens)

            try:
                t0 = time.perf_counter()
                ttft_ms: float | None = None
                tokens = 0
                async with client.stream(
                    "POST",
                    f"http://127.0.0.1:{port}/generate",
                    json={"text": prompt, "max_tokens": mt, "stream": True,
                          "thinking": False, "session_id": f"sim-{user_id}"},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.warning("sim user %d got %d: %s", user_id, resp.status_code, body[:200])
                        await asyncio.sleep(1.0)
                        continue
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        if ttft_ms is None:
                            try:
                                meta = json.loads(payload)
                                if isinstance(meta, dict) and "ttft_ms" in meta:
                                    ttft_ms = float(meta["ttft_ms"])
                                    continue
                            except json.JSONDecodeError:
                                pass
                        tokens += 1
                total = (time.perf_counter() - t0) * 1000

                if ttft_ms is None or tokens == 0:
                    continue

                sim.requests_completed += 1
                sim.total_tokens += tokens
                sim.ttft_sum += ttft_ms
                if tokens > 1:
                    sim.tpot_sum += (total - ttft_ms) / (tokens - 1)
                    sim.tpot_count += 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("sim user %d error: %s", user_id, e)
                await asyncio.sleep(1.0)

            # Jittered think-time so users don't synchronize.
            await asyncio.sleep(rng.uniform(0.05, 0.4))


@app.post("/simulate/start")
async def simulate_start(request: SimulateRequest):
    """Start background traffic simulation."""
    sim = app.state.simulation
    if sim.running:
        raise HTTPException(status_code=409, detail="Simulation already running")

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
