"""Minimal OpenAI /v1/completions shim — benchmark-compatibility surface for guidellm / vLLM
benchmark_serving. NOT a public API: raw-prompt text completion over the same scheduler as
/generate. No auth, no chat template, no registry. See PLAN.md P2, DECISIONS [2026-05-18]."""

import asyncio
import itertools
import json
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference_server.sampling import SamplingParams
from inference_server.scheduler import QueueFullError, ScheduledRequest

router = APIRouter()
_ids = itertools.count(1)  # request-id / session-id counter (no RNG needed)


class CompletionRequest(BaseModel):
    """OpenAI text-completion body. Unknown fields (stop, n, logprobs, ...) are ignored."""
    prompt: str | list[str]
    model: str = "inference-server"
    max_tokens: int = 128
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0


def _prompt_str(p: str | list[str]) -> str:
    """Normalize prompt to a single string (batched prompts unsupported — take the first)."""
    return p[0] if isinstance(p, list) else p


def _chunk(cid: str, created: int, model: str, text: str, finish: str | None) -> str:
    """One streaming SSE chunk in OpenAI text_completion format."""
    payload = {
        "id": cid, "object": "text_completion", "created": created, "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream(req: ScheduledRequest, tokenizer, cid: str, created: int, model: str,
                  prompt_tokens: int, max_tokens: int) -> AsyncGenerator[str, None]:
    """Stream one OpenAI chunk per generated token, then finish_reason + usage + [DONE]."""
    n = 0
    try:
        while True:
            tok_id = await req.token_queue.get()
            if tok_id is None:
                break
            text = tokenizer.decode_token(tok_id)
            n += 1
            if text:
                yield _chunk(cid, created, model, text, None)
        if req.future.done() and req.future.exception():
            raise req.future.exception()  # type: ignore[misc]
    finally:
        finish = "length" if n >= max_tokens else "stop"
        yield _chunk(cid, created, model, "", finish)
        usage_payload = {
            "id": cid, "object": "text_completion", "created": created, "model": model,
            "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                     "total_tokens": prompt_tokens + n},
        }
        yield f"data: {json.dumps(usage_payload)}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    """Text completion over the continuous-batch scheduler — the open-loop bench surface."""
    scheduler = request.app.state.scheduler
    tokenizer = request.app.state.tokenizer
    loop = asyncio.get_running_loop()

    prompt = _prompt_str(body.prompt)
    try:
        token_ids = await loop.run_in_executor(None, tokenizer.encode, prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    sampling = SamplingParams(temperature=body.temperature, top_p=body.top_p, top_k=body.top_k)
    cid = f"cmpl-{next(_ids)}"
    session_id = f"bench-{next(_ids)}"
    created = int(time.time())

    if body.stream:
        req = ScheduledRequest(
            token_ids=token_ids, max_tokens=body.max_tokens, session_id=session_id,
            future=loop.create_future(), token_queue=asyncio.Queue(), sampling=sampling,
        )
        try:
            scheduler.enqueue(req)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e))
        return StreamingResponse(
            _stream(req, tokenizer, cid, created, body.model, len(token_ids), body.max_tokens),
            media_type="text/event-stream",
        )

    req = ScheduledRequest(
        token_ids=token_ids, max_tokens=body.max_tokens, session_id=session_id,
        future=loop.create_future(), sampling=sampling,
    )
    try:
        generated_ids = await scheduler.submit(req)
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e))
    text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)
    n = len(generated_ids)
    return {
        "id": cid, "object": "text_completion", "created": created, "model": body.model,
        "choices": [{"text": text, "index": 0, "logprobs": None,
                     "finish_reason": "length" if n >= body.max_tokens else "stop"}],
        "usage": {"prompt_tokens": len(token_ids), "completion_tokens": n,
                  "total_tokens": len(token_ids) + n},
    }


@router.get("/v1/models")
async def models(request: Request):
    """Model list — some harnesses probe this before benchmarking."""
    model_id = getattr(request.app.state, "model_name", "inference-server")
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}
