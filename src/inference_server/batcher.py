"""Batch processor — collects requests into batches and runs them together."""

import asyncio
import logging
from dataclasses import dataclass

from inference_server.backends.base import InferenceBackend
from inference_server.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request waiting in the queue."""
    token_ids: list[int]
    max_tokens: int
    future: asyncio.Future


class BatchProcessor:
    """Collects incoming requests and runs them as batches through the backend."""

    def __init__(self, backend: InferenceBackend):
        self.backend = backend
        self.queue: asyncio.Queue[BatchRequest] = asyncio.Queue(
            maxsize=settings.max_queue_size
        )
        self._worker_task: asyncio.Task | None = None

    def start(self):
        """Start the background batch worker loop."""
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker_loop())

    async def stop(self):
        """Stop the worker and drain remaining requests."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, token_ids: list[int], max_tokens: int) -> list[int]:
        """Submit a request and wait for the result."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = BatchRequest(token_ids=token_ids, max_tokens=max_tokens, future=future)
        await self.queue.put(request)
        return await future

    async def _collect_batch(self) -> list[BatchRequest]:
        """Wait for at least one request, then drain up to max_batch_size within timeout."""
        batch = []

        first = await self.queue.get()
        batch.append(first)

        timeout = settings.batch_timeout_ms / 1000.0
        try:
            deadline = asyncio.get_event_loop().time() + timeout
            while len(batch) < settings.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                next_req = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                batch.append(next_req)
        except asyncio.TimeoutError:
            pass

        return batch

    async def _worker_loop(self):
        """Continuously collect batches and run them through the backend."""
        loop = asyncio.get_running_loop()

        while True:
            batch = []
            try:
                batch = await self._collect_batch()
                batch_size = len(batch)

                batch_token_ids = [r.token_ids for r in batch]
                batch_max_tokens = [r.max_tokens for r in batch]

                logger.info(f"Processing batch of {batch_size} requests")

                results = await loop.run_in_executor(
                    None, self.backend.generate_batch, batch_token_ids, batch_max_tokens
                )

                for request, result in zip(batch, results):
                    if not request.future.cancelled():
                        request.future.set_result(result)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(e)
