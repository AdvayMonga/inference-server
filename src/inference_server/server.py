"""FastAPI server — accepts text, generates LLM responses via the backend."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from inference_server.backends import create_backend
from inference_server.config import settings
from inference_server.tokenizer import Tokenizer


class GenerateRequest(BaseModel):
    text: str
    max_tokens: int = settings.max_tokens


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


@asynccontextmanager
async def lifespan(app):
    # Startup — load model and tokenizer once
    backend = create_backend(settings.backend)
    tokenizer = Tokenizer(settings.model_name, settings.context_window)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, backend.load_model, settings.model_name)

    app.state.backend = backend
    app.state.tokenizer = tokenizer

    yield

    # Shutdown — cleanup if needed


app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    backend = app.state.backend
    tokenizer = app.state.tokenizer
    loop = asyncio.get_event_loop()

    # Tokenize in thread (CPU-bound, don't block event loop)
    token_ids = await loop.run_in_executor(None, tokenizer.encode_chat, request.text)

    # Generate in thread (model inference, definitely don't block event loop)
    generated_ids = await loop.run_in_executor(
        None, backend.generate, token_ids, request.max_tokens
    )

    # Detokenize in thread
    output_text = await loop.run_in_executor(None, tokenizer.decode, generated_ids)

    return GenerateResponse(text=output_text, tokens_generated=len(generated_ids))
