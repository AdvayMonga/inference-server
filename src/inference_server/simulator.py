"""Background traffic simulator. N async users hammer /generate via localhost."""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from inference_server.config import settings
from inference_server.simulator_prompts import MAX_TOKENS_RANGE, PROMPT_MIX

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulate", tags=["simulator"])


class SimulateRequest(BaseModel):
    """Request body for /simulate/start. max_tokens caps all buckets."""
    num_users: int = 4
    max_tokens: int = 500


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


def _sample_prompt_and_max_tokens(rng: random.Random, max_tokens_cap: int) -> tuple[str, int, str]:
    """Weighted pick from PROMPT_MIX, then random max_tokens within the bucket's range."""
    r = rng.random()
    cumulative = 0.0
    for bucket_name, prompts, weight in PROMPT_MIX:
        cumulative += weight
        if r <= cumulative:
            prompt = rng.choice(prompts)
            lo, hi = MAX_TOKENS_RANGE[bucket_name]
            hi = min(hi, max_tokens_cap)
            lo = min(lo, hi)
            mt = rng.randint(lo, hi)
            return prompt, mt, bucket_name
    prompt = rng.choice(PROMPT_MIX[0][1])
    return prompt, max_tokens_cap, PROMPT_MIX[0][0]


async def _simulated_user(user_id: int, sim: SimulationState, max_tokens: int, port: int):
    """One simulated user — loops sending streaming requests until cancelled."""
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

                if ttft_ms is not None and tokens > 0:
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

            # Jittered think-time so users don't synchronize. Always applied,
            # even after a degenerate response, to avoid tight retry loops.
            await asyncio.sleep(rng.uniform(0.05, 0.4))


@router.post("/start")
async def simulate_start(request: SimulateRequest, http_request: Request):
    """Start background traffic simulation.

    Returns a `warning` field if num_users >= max_batch_size — when the sim
    fills every batch slot, real requests from the user queue behind it and
    feel unresponsive. Headroom of at least one slot is recommended for
    live testing under load.
    """
    sim: SimulationState = http_request.app.state.simulation
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

    resp: dict = {"status": "started", "num_users": request.num_users}
    if request.num_users >= settings.max_batch_size:
        msg = (
            f"num_users ({request.num_users}) >= max_batch_size "
            f"({settings.max_batch_size}); your own requests will queue "
            f"behind sim traffic. Lower num_users for headroom."
        )
        logger.warning(msg)
        resp["warning"] = msg
    return resp


@router.post("/stop")
async def simulate_stop(http_request: Request):
    """Stop background traffic simulation."""
    sim: SimulationState = http_request.app.state.simulation
    if not sim.running:
        return {"status": "not_running"}

    for task in sim.tasks:
        task.cancel()
    await asyncio.gather(*sim.tasks, return_exceptions=True)

    sim.running = False
    sim.tasks = []
    sim.num_users = 0
    return {"status": "stopped"}


@router.get("/status")
async def simulate_status(http_request: Request):
    """Return live simulation stats."""
    sim: SimulationState = http_request.app.state.simulation
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
