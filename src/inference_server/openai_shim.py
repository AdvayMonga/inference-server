"""Minimal OpenAI shim — compatibility surface for guidellm / vLLM benchmark_serving and for
chat frontends such as Open WebUI. NOT a public API: /v1/completions is raw-prompt text
completion, /v1/chat/completions applies the model's chat template, and both run over the same
scheduler as /generate. No auth, no registry, no tools, no multimodal content parts.
Optional X-Session-Id / X-Turn-Index / X-Trace-Id request headers let a harness pin the engine
session and correlate rows; the trace id is echoed back as X-Trace-Id. See PLAN.md P2,
DECISIONS [2026-05-18]."""

import asyncio
import itertools
import json
import logging
import time
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference_server.sampling import SamplingParams
from inference_server.scheduler import QueueFullError, RequestTooLargeError, ScheduledRequest

logger = logging.getLogger(__name__)

router = APIRouter()
_ids = itertools.count(1)  # request-id / session-id counter (no RNG needed)


class ChatMessage(BaseModel):
    """One turn. `content` is text only — OpenAI's list-of-parts form is rejected upstream."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat body. Unknown fields (tools, response_format, ...) are ignored."""
    messages: list[ChatMessage]
    model: str = "inference-server"
    max_tokens: int = 128
    max_completion_tokens: int | None = None  # the field that replaced max_tokens
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0

    @property
    def output_budget(self) -> int:
        return self.max_completion_tokens or self.max_tokens


class CompletionRequest(BaseModel):
    """OpenAI text-completion body; `prompt_logprobs: k` (vLLM's) scores the prompt instead of generating."""
    prompt: str | list[str] | list[int]
    model: str = "inference-server"
    max_tokens: int = 128
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    prompt_logprobs: int | None = None


def _correlation(request: Request, auto_prefix: str) -> tuple[str, int, str]:
    """(session_id, turn_index, trace_id) from optional harness headers; absent -> auto ids."""
    h = request.headers
    session_id = h.get("x-session-id") or f"{auto_prefix}-{next(_ids)}"
    try:
        turn_index = int(h.get("x-turn-index", 0))
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Turn-Index must be an integer")
    return session_id, turn_index, h.get("x-trace-id", "")


def _prompt_str(p: str | list[str]) -> str:
    """Normalize prompt to a single string (batched prompts unsupported — take the first)."""
    if isinstance(p, list) and not p:
        raise HTTPException(status_code=400, detail="prompt is empty")
    return p[0] if isinstance(p, list) else p


def _token_ids(ids: list[int], tokenizer) -> list[int]:
    """A pre-tokenized prompt gets the checks tokenizer.encode would have made."""
    if not ids:
        raise HTTPException(status_code=400, detail="prompt is empty")
    vocab = getattr(tokenizer, "n_tokens", None)   # incl. added special tokens; vocab_size excludes them
    if vocab is not None and any(t < 0 or t >= vocab for t in ids):
        raise HTTPException(status_code=400, detail=f"token id outside the vocabulary of {vocab}")
    window = getattr(tokenizer, "context_window", None)
    if window is not None and len(ids) > window:
        raise HTTPException(status_code=400, detail=f"{len(ids)} tokens exceed the context window of {window}")
    return list(ids)


def check_context(prompt_tokens: int, max_tokens: int, tokenizer) -> None:
    """400 when prompt + completion would exceed the context window (vLLM's check and wording)."""
    window = getattr(tokenizer, "context_window", None)
    if window is not None and prompt_tokens + max_tokens > window:
        raise HTTPException(status_code=400, detail=(
            f"This model's maximum context length is {window} tokens. However, you requested "
            f"{prompt_tokens + max_tokens} tokens ({prompt_tokens} in the messages, {max_tokens} in "
            f"the completion). Please reduce the length of the messages or completion."))


def _chunk(cid: str, created: int, model: str, text: str, finish: str | None) -> str:
    """One streaming SSE chunk in OpenAI text_completion format."""
    payload = {
        "id": cid, "object": "text_completion", "created": created, "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _first_token(req: ScheduledRequest) -> int | None:
    """Wait for the first token before any byte of the response goes out, so a request that fails
    before it (expired in the queue, shed, preempted) is still an HTTP status rather than a 200
    whose stream is cut off. None: it ended without generating anything."""
    try:
        tok = await req.token_queue.get()
    except asyncio.CancelledError:
        req.cancelled = True
        raise
    if tok is None:
        await req.future           # raises the request's failure, whatever order its signals came in
    return tok


async def _token_batches(req: ScheduledRequest, first: int | None):
    """The generated token ids from `first`, as lists of every id already queued (one chunk each: the scheduler hands a
    step's tokens over at once, and a stream that fell behind catches up in one write). Ends normally on completion; on
    a failure after the first token it ends too, leaving `req.future` holding the error."""
    if first is None:
        return
    batch = [first]
    while True:
        ended = False
        while not req.token_queue.empty():
            tok = req.token_queue.get_nowait()
            if tok is None:
                ended = True
                break
            batch.append(tok)
        yield batch
        if ended:
            return
        tok = await req.token_queue.get()
        if tok is None:
            return
        batch = [tok]


def _failed_midway(req: ScheduledRequest, n: int) -> bool:
    """A failure after tokens were sent: the stream must end without a finish reason or [DONE],
    so the client sees it cut short instead of a complete answer."""
    exc = req.future.exception() if req.future.done() else None
    if exc is not None:
        logger.warning("request %s failed after %d tokens: %s", req.trace_id, n, exc)
    return exc is not None


async def _stream(req: ScheduledRequest, first: int | None, tokenizer, cid: str, created: int,
                  model: str, prompt_tokens: int, max_tokens: int) -> AsyncGenerator[str, None]:
    """Stream one OpenAI chunk per generated token, then finish_reason + usage + [DONE]."""
    n = 0
    ended = False
    detok = tokenizer.incremental()
    try:
        async for ids in _token_batches(req, first):
            text = detok.add(ids)
            n += len(ids)
            if text:
                yield _chunk(cid, created, model, text, None)
        if text := detok.flush():
            yield _chunk(cid, created, model, text, None)
        ended = True
    finally:
        req.cancelled = not ended      # client went away mid-stream: the scheduler stops the row
    if _failed_midway(req, n):
        return
    yield _chunk(cid, created, model, "", "length" if n >= max_tokens else "stop")
    usage_payload = {
        "id": cid, "object": "text_completion", "created": created, "model": model,
        "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                 "total_tokens": prompt_tokens + n},
    }
    yield f"data: {json.dumps(usage_payload)}\n\n"
    yield "data: [DONE]\n\n"


async def _score(request: Request, body: CompletionRequest, token_ids: list[int], loop) -> dict:
    """Teacher-forced scoring (vLLM `prompt_logprobs` shape) for the output-equivalence check."""
    backend = getattr(request.app.state, "backend", None)
    if backend is None or not hasattr(backend, "score_logprobs"):
        raise HTTPException(status_code=501, detail="this backend cannot score prompt logprobs")
    if body.stream:
        raise HTTPException(status_code=400, detail="prompt_logprobs cannot be streamed")
    k = max(1, min(int(body.prompt_logprobs), 100))
    try:
        scored = await loop.run_in_executor(None, backend.score_logprobs, token_ids, k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    entries = []
    for d in scored:
        if d is None:
            entries.append(None)
            continue
        ranked = sorted(d.items(), key=lambda kv: -kv[1])
        entries.append({str(t): {"logprob": lp, "rank": r + 1} for r, (t, lp) in enumerate(ranked)})
    return {"id": f"cmpl-{next(_ids)}", "object": "text_completion", "created": int(time.time()),
            "model": body.model,
            "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "length",
                         "prompt_logprobs": entries}],
            "usage": {"prompt_tokens": len(token_ids), "completion_tokens": 0, "total_tokens": len(token_ids)}}


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request, response: Response):
    """Text completion over the continuous-batch scheduler — the open-loop bench surface."""
    arrival = time.perf_counter()
    scheduler = request.app.state.scheduler
    tokenizer = request.app.state.tokenizer
    loop = asyncio.get_running_loop()

    session_id, turn_index, trace_id = _correlation(request, "bench")
    if isinstance(body.prompt, list) and body.prompt and isinstance(body.prompt[0], int):
        token_ids = _token_ids(body.prompt, tokenizer)          # pre-tokenized, as vLLM accepts
    else:
        prompt = _prompt_str(body.prompt)
        try:
            token_ids = await loop.run_in_executor(None, tokenizer.encode, prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if body.prompt_logprobs is not None:
        if trace_id:
            response.headers["X-Trace-Id"] = trace_id
        return await _score(request, body, token_ids, loop)
    check_context(len(token_ids), body.max_tokens, tokenizer)   # scoring generates nothing, so it skips this

    sampling = SamplingParams(temperature=body.temperature, top_p=body.top_p, top_k=body.top_k)
    cid = f"cmpl-{next(_ids)}"
    created = int(time.time())

    if body.stream:
        req = ScheduledRequest(
            token_ids=token_ids, max_tokens=body.max_tokens, session_id=session_id,
            future=loop.create_future(), token_queue=asyncio.Queue(), sampling=sampling,
            trace_id=trace_id, turn_index=turn_index, arrival_ts=arrival,
        )
        try:
            scheduler.enqueue(req)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        except RequestTooLargeError as e:
            raise HTTPException(status_code=400, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        try:
            first = await _first_token(req)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        except RequestTooLargeError as e:             # with ENGINE_PROCESS the engine's verdict arrives here
            raise HTTPException(status_code=400, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        return StreamingResponse(
            _stream(req, first, tokenizer, cid, created, body.model, len(token_ids), body.max_tokens),
            media_type="text/event-stream", headers={"X-Trace-Id": req.trace_id},
        )

    req = ScheduledRequest(
        token_ids=token_ids, max_tokens=body.max_tokens, session_id=session_id,
        future=loop.create_future(), sampling=sampling,
        trace_id=trace_id, turn_index=turn_index, arrival_ts=arrival,
    )
    try:
        generated_ids = await scheduler.submit(req)
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
    except RequestTooLargeError as e:
        raise HTTPException(status_code=400, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
    text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)
    n = len(generated_ids)
    response.headers["X-Trace-Id"] = req.trace_id
    return {
        "id": cid, "object": "text_completion", "created": created, "model": body.model,
        "choices": [{"text": text, "index": 0, "logprobs": None,
                     "finish_reason": "length" if n >= body.max_tokens else "stop"}],
        "usage": {"prompt_tokens": len(token_ids), "completion_tokens": n,
                  "total_tokens": len(token_ids) + n},
    }


def _chat_chunk(cid: str, created: int, model: str, delta: dict, finish: str | None) -> str:
    """One streaming SSE chunk in OpenAI chat.completion.chunk format."""
    payload = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _chat_stream(req: ScheduledRequest, first: int | None, tokenizer, cid: str, created: int,
                       model: str, prompt_tokens: int, max_tokens: int) -> AsyncGenerator[str, None]:
    """Role delta first (clients key off it), then one content delta per token, then usage."""
    n = 0
    ended = False
    detok = tokenizer.incremental()
    try:
        yield _chat_chunk(cid, created, model, {"role": "assistant", "content": ""}, None)
        async for ids in _token_batches(req, first):
            text = detok.add(ids)
            n += len(ids)
            if text:
                yield _chat_chunk(cid, created, model, {"content": text}, None)
        if text := detok.flush():
            yield _chat_chunk(cid, created, model, {"content": text}, None)
        ended = True
    finally:
        req.cancelled = not ended      # client went away mid-stream: the scheduler stops the row
    if _failed_midway(req, n):
        return
    yield _chat_chunk(cid, created, model, {}, "length" if n >= max_tokens else "stop")
    usage_payload = {
        "id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n,
                                 "total_tokens": prompt_tokens + n},
    }
    yield f"data: {json.dumps(usage_payload)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request, response: Response):
    """Chat completion over the continuous-batch scheduler — the frontend-facing surface."""
    arrival = time.perf_counter()
    scheduler = request.app.state.scheduler
    tokenizer = request.app.state.tokenizer
    loop = asyncio.get_running_loop()

    messages = [m.model_dump() for m in body.messages]
    try:
        # thinking=False, chosen rather than inherited. A compatibility shim must not switch the
        # model into a mode no caller asked for: encode_messages defaults to True, which once made
        # every answer open with a thinking preamble and run every corpus request to max_tokens
        # (kb-20260919-9ea56f98); for Qwen3 it renders the empty <think> block instead.
        token_ids = await loop.run_in_executor(
            None, lambda: tokenizer.encode_messages(messages, thinking=False))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    max_tokens = body.output_budget
    check_context(len(token_ids), max_tokens, tokenizer)
    sampling = SamplingParams(temperature=body.temperature, top_p=body.top_p, top_k=body.top_k)
    cid = f"chatcmpl-{next(_ids)}"
    session_id, turn_index, trace_id = _correlation(request, "chat")
    created = int(time.time())

    if body.stream:
        req = ScheduledRequest(
            token_ids=token_ids, max_tokens=max_tokens, session_id=session_id,
            future=loop.create_future(), token_queue=asyncio.Queue(), sampling=sampling,
            trace_id=trace_id, turn_index=turn_index, arrival_ts=arrival,
        )
        try:
            scheduler.enqueue(req)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        except RequestTooLargeError as e:
            raise HTTPException(status_code=400, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        try:
            first = await _first_token(req)
        except QueueFullError as e:
            raise HTTPException(status_code=429, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        except RequestTooLargeError as e:             # with ENGINE_PROCESS the engine's verdict arrives here
            raise HTTPException(status_code=400, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
        return StreamingResponse(
            _chat_stream(req, first, tokenizer, cid, created, body.model, len(token_ids), max_tokens),
            media_type="text/event-stream", headers={"X-Trace-Id": req.trace_id},
        )

    req = ScheduledRequest(
        token_ids=token_ids, max_tokens=max_tokens, session_id=session_id,
        future=loop.create_future(), sampling=sampling,
        trace_id=trace_id, turn_index=turn_index, arrival_ts=arrival,
    )
    try:
        generated_ids = await scheduler.submit(req)
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
    except RequestTooLargeError as e:
        raise HTTPException(status_code=400, detail=str(e),
                                headers={"X-Trace-Id": req.trace_id})
    text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)
    n = len(generated_ids)
    response.headers["X-Trace-Id"] = req.trace_id
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": body.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "length" if n >= max_tokens else "stop"}],
        "usage": {"prompt_tokens": len(token_ids), "completion_tokens": n,
                  "total_tokens": len(token_ids) + n},
    }


@router.get("/v1/models")
async def models(request: Request):
    """Model list — some harnesses probe this before benchmarking."""
    model_id = getattr(request.app.state, "model_name", "inference-server")
    return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}
