"""Scheduler interface and ContinuousBatchScheduler implementation."""

import asyncio
import logging
import queue as stdqueue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from inference_server.backends.base import InferenceBackend

logger = logging.getLogger(__name__)


@dataclass
class ScheduledRequest:
    """A request handed to a scheduler. The scheduler resolves `future` on completion."""
    token_ids: list[int]
    max_tokens: int
    session_id: str
    future: asyncio.Future
    token_queue: asyncio.Queue | None = None
    generated: list[int] = field(default_factory=list)
    cache_hit_tokens: int = 0   # set by scheduler after prefill


class SchedulerInterface(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    async def stop(self) -> None: ...
    @abstractmethod
    async def submit(self, request: ScheduledRequest) -> list[int]: ...


@dataclass
class _ActiveRow:
    """One slot in the running batch."""
    request: ScheduledRequest
    current_token: int       # token to feed next forward pass
    real_kv_len: int         # actual KV cache length for this row


class ContinuousBatchScheduler(SchedulerInterface):
    """Iteration-level scheduler: per-step admit, decode, evict. FIFO admission."""

    def __init__(self, backend: InferenceBackend, max_batch_size: int = 16):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self._pending: stdqueue.Queue[ScheduledRequest] = stdqueue.Queue()
        self._active: list[_ActiveRow] = []
        self._batched_kv: object | None = None
        self._attention_mask: torch.Tensor | None = None
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- Public interface ---

    def start(self) -> None:
        if self._worker is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="cb-scheduler", daemon=True)
        self._worker.start()

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._stop_event.set()
        # nudge the queue so the worker unblocks
        self._pending.put(None)  # type: ignore[arg-type]
        await asyncio.get_running_loop().run_in_executor(None, self._worker.join)
        self._worker = None

    async def submit(self, request: ScheduledRequest) -> list[int]:
        self._pending.put(request)
        return await request.future

    # --- Worker loop ---

    def _run(self) -> None:
        """Single-threaded scheduler loop."""
        device = self.backend.device_str
        try:
            while not self._stop_event.is_set():
                # If batch empty and nothing pending, block briefly on the queue.
                if not self._active and self._pending.empty():
                    try:
                        first = self._pending.get(timeout=0.1)
                    except stdqueue.Empty:
                        continue
                    if first is None:
                        break
                    self._pending.put(first)  # put back so admit phase picks it up

                with self.backend._lock:  # serialize against legacy generate/stream
                    # Admit first so freshly admitted first_tokens get processed below
                    self._admit_pending(device)
                    self._evict_finished()
                    if self._active:
                        self._decode_step(device)
        except Exception as e:
            logger.exception("Scheduler crashed: %s", e)
            self._fail_all(e)

    # --- Phase 1: process tokens, evict finished rows ---

    def _evict_finished(self) -> None:
        """Process each active row's current_token, evict finished rows."""
        # Walk indices high→low so popping doesn't shift earlier indices
        to_evict: list[int] = []
        for i, row in enumerate(self._active):
            tok = row.current_token

            if self.backend.is_eos(tok):
                to_evict.append(i)
                continue

            row.request.generated.append(tok)
            self._stream_token(row.request, tok)

            if len(row.request.generated) >= row.request.max_tokens:
                to_evict.append(i)

        for i in reversed(to_evict):
            self._evict_row(i)

    def _evict_row(self, idx: int) -> None:
        row = self._active.pop(idx)
        if self._batched_kv is not None:
            if len(self._active) == 0:
                self._batched_kv = None
                self._attention_mask = None
            else:
                self._batched_kv = self.backend.remove_row_from_cache(self._batched_kv, idx)
                self._attention_mask = torch.cat([
                    self._attention_mask[:idx],
                    self._attention_mask[idx + 1:],
                ], dim=0)
        self._resolve(row.request)

    # --- Phase 2: admit new requests ---

    def _admit_pending(self, device: str) -> None:
        while len(self._active) < self.max_batch_size:
            try:
                req = self._pending.get_nowait()
            except stdqueue.Empty:
                return
            if req is None:
                return
            try:
                kv, first_token, kv_len = self.backend.prefill(req.token_ids, req.session_id)
                req.cache_hit_tokens = self.backend.last_cache_hit_tokens
            except Exception as e:
                logger.exception("Prefill failed for session %s", req.session_id)
                self._reject(req, e)
                continue
            self._splice_in(kv, kv_len, device)
            self._active.append(_ActiveRow(request=req, current_token=first_token, real_kv_len=kv_len))

    def _splice_in(self, new_kv: object, new_kv_len: int, device: str) -> None:
        """Add a new row's KV to the batched cache; backend handles cache surgery."""
        if self._batched_kv is None:
            self._batched_kv = self.backend.splice_into_batched(None, new_kv, new_kv_len)
            self._attention_mask = torch.ones(1, new_kv_len, device=device, dtype=torch.long)
            return

        existing_len = self.backend.kv_length(self._batched_kv)
        max_len = max(existing_len, new_kv_len)
        existing_pad = max_len - existing_len
        new_pad = max_len - new_kv_len

        self._batched_kv = self.backend.splice_into_batched(self._batched_kv, new_kv, new_kv_len)

        # Attention mask is torch-typed scheduler state; pad existing rows then append new row.
        if existing_pad > 0:
            zeros = torch.zeros(self._attention_mask.shape[0], existing_pad, device=device, dtype=torch.long)
            self._attention_mask = torch.cat([zeros, self._attention_mask], dim=1)
        new_row_mask = torch.zeros(1, max_len, device=device, dtype=torch.long)
        new_row_mask[0, max_len - new_kv_len:] = 1
        self._attention_mask = torch.cat([self._attention_mask, new_row_mask], dim=0)

    # --- Phase 3: batched decode step ---

    def _decode_step(self, device: str) -> None:
        batch_size = len(self._active)
        current_tokens = torch.tensor(
            [[r.current_token] for r in self._active],
            device=device,
        )
        # Extend attention mask by one column for the new input token
        self._attention_mask = torch.cat([
            self._attention_mask,
            torch.ones(batch_size, 1, device=device, dtype=torch.long),
        ], dim=1)
        position_ids = torch.tensor(
            [[r.real_kv_len] for r in self._active],
            device=device, dtype=torch.long,
        )

        next_tokens, self._batched_kv = self.backend.decode_step_batched(
            current_tokens, self._batched_kv, self._attention_mask, position_ids,
        )

        for i, row in enumerate(self._active):
            row.current_token = int(next_tokens[i].item())
            row.real_kv_len += 1

    # --- Cross-thread helpers ---

    def _stream_token(self, request: ScheduledRequest, token: int) -> None:
        if request.token_queue is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(request.token_queue.put_nowait, token)

    def _resolve(self, request: ScheduledRequest) -> None:
        if self._loop is None:
            return
        result = list(request.generated)
        if request.token_queue is not None:
            self._loop.call_soon_threadsafe(request.token_queue.put_nowait, None)
        # Release cached blocks tied to this prompt
        if self.backend.cache_adapter is not None:
            self.backend.cache_adapter.release(request.token_ids, session_id=request.session_id)

        def _set():
            if not request.future.done():
                request.future.set_result(result)
        self._loop.call_soon_threadsafe(_set)

    def _reject(self, request: ScheduledRequest, exc: BaseException) -> None:
        if self._loop is None:
            return
        if request.token_queue is not None:
            self._loop.call_soon_threadsafe(request.token_queue.put_nowait, None)

        def _set():
            if not request.future.done():
                request.future.set_exception(exc)
        self._loop.call_soon_threadsafe(_set)

    def _fail_all(self, exc: BaseException) -> None:
        for row in self._active:
            self._reject(row.request, exc)
        self._active.clear()
        self._batched_kv = None
        self._attention_mask = None
