"""M2.4: ContinuousBatchScheduler driving the CustomTorchBackend end-to-end.

Verifies the scheduler-facing primitives (prefill / decode_step_batched / splice /
remove / kv_length) produce correct tokens via paged KV, and that the eviction path
frees blocks back to the pool (no leak across requests). CPU-only (bf16 parity).
"""

from __future__ import annotations

import os

import pytest
import torch

from inference_server.backends.custom_torch_backend import CustomTorchBackend
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest

MODEL_NAME = "google/gemma-4-E2B-it"
PROMPT_IDS = [2, 651, 6037, 576, 6081, 603]  # arbitrary short token prompt
MAX_TOKENS = 6


@pytest.fixture(scope="module")
def backend():
    os.environ["CUSTOM_BACKEND_BLOCKS"] = "64"
    os.environ["CUSTOM_BACKEND_BLOCK_SIZE"] = "2"  # 6-token prompt → 3 full blocks (sharing engages)
    b = CustomTorchBackend(device="cpu", model_name=MODEL_NAME)
    b.load_model(MODEL_NAME)
    return b


def _manual_greedy(backend, token_ids, max_tokens):
    """Reference: prefill + greedy decode with a fresh paged cache, no think-filtering.
    Mirrors exactly what the scheduler collects (greedy first token, then decode)."""
    from inference_server.models.paged_kv_cache import PagedKVCache

    cache = PagedKVCache(pools=backend.pools)
    try:
        with torch.no_grad():
            logits = backend.model(torch.tensor([token_ids], device=backend.device), kv_cache=cache)
            tok = int(logits[:, -1, :].argmax(-1).item())
            out = []
            for _ in range(max_tokens):
                if backend.is_eos(tok):
                    break
                out.append(tok)
                if len(out) >= max_tokens:
                    break
                logits = backend.model(torch.tensor([[tok]], device=backend.device), kv_cache=cache)
                tok = int(logits[:, -1, :].argmax(-1).item())
        return out
    finally:
        cache.free_all()


async def _run_through_scheduler(backend, prompts, max_tokens):
    import asyncio

    sched = ContinuousBatchScheduler(backend, max_batch_size=8)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [
            ScheduledRequest(
                token_ids=list(p), max_tokens=max_tokens,
                session_id=f"s{i}", future=loop.create_future(),
            )
            for i, p in enumerate(prompts)
        ]
        return await asyncio.gather(*(sched.submit(r) for r in reqs))
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_output_matches_manual_greedy(backend):
    """Scheduler-driven custom backend yields the same tokens as a manual greedy decode."""
    expected = _manual_greedy(backend, PROMPT_IDS, MAX_TOKENS)
    (out,) = await _run_through_scheduler(backend, [PROMPT_IDS], MAX_TOKENS)
    assert out == expected, f"scheduler {out} != manual {expected}"
    assert len(out) >= 1


def _free_counts(backend):
    return [p.free_count if p is not None else None for p in backend.pools]


@pytest.mark.asyncio
async def test_eviction_frees_blocks_no_leak(backend):
    """Repeating a request must not monotonically drain the pool: the eviction path
    frees each session's blocks (only the prefix-cache entry stays resident)."""
    await _run_through_scheduler(backend, [PROMPT_IDS], MAX_TOKENS)  # warms prefix cache
    after_first = _free_counts(backend)
    await _run_through_scheduler(backend, [PROMPT_IDS], MAX_TOKENS)
    after_second = _free_counts(backend)
    assert after_first == after_second, (
        f"pool free counts drifted across identical requests (leak): {after_first} -> {after_second}"
    )


@pytest.mark.asyncio
async def test_concurrent_shared_prefix_hits_and_frees(backend):
    """Two sessions sharing a prefix both complete; prefix cache records hits; no leak."""
    before = _free_counts(backend)
    hits_before = backend.prefix_cache.hits
    outs = await _run_through_scheduler(backend, [PROMPT_IDS, PROMPT_IDS], MAX_TOKENS)
    assert all(len(o) >= 1 for o in outs)
    assert outs[0] == outs[1]  # identical prompts → identical greedy output
    assert backend.prefix_cache.hits > hits_before
    assert _free_counts(backend) == before, "blocks leaked after concurrent shared-prefix run"
