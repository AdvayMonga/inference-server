"""Scheduler interface and ContinuousBatchScheduler implementation."""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from inference_server.backends.base import InferenceBackend
from inference_server.metrics import MetricsTracker
from inference_server.sampling import SamplingParams, sample
from inference_server.scheduling_policy import FCFSPolicy, SchedulingPolicy

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Raised by submit() when the pending queue has no capacity."""


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
    arrival_seq: int = 0        # set by scheduler at enqueue (monotonic)
    priority: int = 0           # higher = more important; tiebreak field for policies
    enqueue_ts: float = 0.0     # set by scheduler at enqueue (perf_counter)
    first_token_ts: float = 0.0 # set when first token is produced
    sampling: SamplingParams = field(default_factory=SamplingParams)


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


@dataclass
class _PrefillingRow:
    """Request whose prefill is in progress (chunked-prefill mode)."""
    request: ScheduledRequest
    matched: int                 # cached prefix length from lookup
    tokens_fed: int              # tokens currently in partial_kv (starts == matched)
    partial_kv: object | None    # KV so far; None if no cache hit and no chunk fed yet


def plan_wave(window: list["ScheduledRequest"], k: int) -> list["ScheduledRequest"]:
    """Pick up to `k` requests from `window` (policy order) that pad together well.

    A prefill wave is right-padded to Smax, so a ragged wave wastes most of its compute: on a
    ShareGPT-ish length distribution a K=8 wave is ~59% padding. Grouping similar lengths
    recovers a large part of that.

    The window HEAD is always included. That is what makes this starvation-free: no request can
    be passed over once it reaches the front, so reordering never defers anyone past their turn
    as head. It costs about half the achievable saving versus an unbounded sort — deliberately,
    because TTFT p95 is the SLO metric and the p95 requests are precisely the long prompts an
    unbounded sort would keep deferring.
    """
    if k <= 0 or not window:
        return []
    if k >= len(window):
        return list(window)
    order = sorted(range(len(window)), key=lambda i: len(window[i].token_ids))
    head_at = order.index(0)
    best = None
    for start in range(max(0, head_at - k + 1), min(head_at + 1, len(order) - k + 1)):
        sel = order[start:start + k]
        lens = [len(window[i].token_ids) for i in sel]
        cost = max(lens) * len(lens) - sum(lens)          # padded tokens this wave would add
        if best is None or cost < best[0]:
            best = (cost, sel)
    return [window[i] for i in sorted(best[1])]           # admit in policy order within the wave


class ContinuousBatchScheduler(SchedulerInterface):
    """Iteration-level scheduler: per-step admit, decode, evict. FIFO admission."""

    # Prefill strategies. `monolithic` = one forward on admit; `chunked` = V-A (one prefill
    # chunk + one decode step per iter, kills HOL blocking). The enum is the forward-compat
    # extension point: `mixed_batch` (V-B) and `disaggregated` (P/D) are future strategies that
    # reuse the same `_prefilling` phase + `prefill_chunk` primitive + `_promote_to_decode` seam.
    PREFILL_MODES = ("monolithic", "chunked", "batched")

    def __init__(self, backend: InferenceBackend, max_batch_size: int = 16,
                 max_queue_size: int = 1000,
                 policy: SchedulingPolicy | None = None,
                 max_active_kv_tokens: int = 0,
                 prefill_chunk_size: int = 0,
                 prefill_mode: str | None = None,
                 wave_window_mult: int = 0):
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        # 0 means "no explicit cap"; we use a huge sentinel so checks are uniform.
        self.max_active_kv_tokens = max_active_kv_tokens if max_active_kv_tokens > 0 else 2**31
        # Tokens fed per prefill chunk (chunked mode only).
        self.prefill_chunk_size = prefill_chunk_size
        # Default mode from chunk_size (back-compat: chunk_size>0 → chunked), explicit override wins.
        self.prefill_mode = prefill_mode or ("chunked" if prefill_chunk_size > 0 else "monolithic")
        if self.prefill_mode not in self.PREFILL_MODES:
            raise ValueError(f"unknown prefill_mode {self.prefill_mode!r}; expected one of {self.PREFILL_MODES}")
        if self.prefill_mode == "chunked" and prefill_chunk_size <= 0:
            raise ValueError("prefill_mode='chunked' requires prefill_chunk_size > 0")
        if self.prefill_mode == "batched" and not hasattr(backend, "prefill_batch"):
            raise ValueError("prefill_mode='batched' requires a backend with prefill_batch (custom-*)")
        # Window (as a multiple of free slots) that batched-mode wave planning may reorder
        # within, to group similar prompt lengths. DEFAULT 0 (off) — measured inert on our
        # current workload: at rates 1-6 the queue never backs up, so 83-91% of prefill waves
        # are K=1 and there is no padding to save (see wave_sizes in stats()). Turn it on when
        # the engine actually runs queued at the 100-256 concurrency target, where waves are
        # wide and padding is 59%+ of prefill. Only affects prefill_mode='batched'.
        self.wave_window_mult = wave_window_mult
        self.policy: SchedulingPolicy = policy if policy is not None else FCFSPolicy()
        self._pending_lock = threading.Lock()
        self._pending_cv = threading.Condition(self._pending_lock)
        self._pending_count = 0
        self._arrival_counter = 0
        # Backends with paged per-row KV ignore the scheduler-built mask; skip building it.
        self._needs_mask = getattr(backend, "needs_attention_mask", True)
        self._active: list[_ActiveRow] = []
        self._prefilling: list[_PrefillingRow] = []
        self._batched_kv: object | None = None
        self._attention_mask: torch.Tensor | None = None
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Counters
        self._total_admitted = 0
        self._total_completed = 0
        self._total_rejected = 0
        self._pending_high_water = 0
        self._kv_admit_blocked = 0
        self._active_kv_reserved = 0  # sum of (prompt_len + max_tokens) over in-flight rows
        self._prefill_chunks_processed = 0
        # Histogram of batched-prefill wave sizes. Padding waste only exists when waves are
        # WIDE; if the queue never backs up every arrival is its own K=1 wave and there is
        # nothing to group. This is how we tell those regimes apart.
        self._wave_sizes: dict[int, int] = {}
        self._metrics = MetricsTracker()

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
        with self._pending_cv:
            self._pending_cv.notify_all()
        await asyncio.get_running_loop().run_in_executor(None, self._worker.join)
        self._worker = None

    def enqueue(self, request: ScheduledRequest) -> None:
        """Synchronously enqueue. Raises QueueFullError if no capacity."""
        request.enqueue_ts = time.perf_counter()
        with self._pending_cv:
            if self._pending_count >= self.max_queue_size:
                self._total_rejected += 1
                raise QueueFullError(
                    f"pending queue at capacity ({self.max_queue_size})"
                )
            self._arrival_counter += 1
            request.arrival_seq = self._arrival_counter
            self.policy.on_request_arrived(request)
            self._pending_count += 1
            if self._pending_count > self._pending_high_water:
                self._pending_high_water = self._pending_count
            self._pending_cv.notify()

    async def submit(self, request: ScheduledRequest) -> list[int]:
        self.enqueue(request)
        return await request.future

    def stats(self) -> dict:
        with self._pending_lock:
            pending_depth = self._pending_count
        cache = self.backend.cache_adapter
        kv_pressure = cache.pressure if cache is not None else 0.0
        kv_free_blocks = cache.free_blocks if cache is not None else 0
        return {
            "active_size": len(self._active),
            "prefilling_depth": len(self._prefilling),
            "pending_depth": pending_depth,
            "max_batch_size": self.max_batch_size,
            "max_queue_size": self.max_queue_size,
            "policy": type(self.policy).__name__,
            "total_admitted": self._total_admitted,
            "total_completed": self._total_completed,
            "total_rejected": self._total_rejected,
            "pending_high_water": self._pending_high_water,
            "kv_pressure": kv_pressure,
            "kv_free_blocks": kv_free_blocks,
            "kv_admit_blocked": self._kv_admit_blocked,
            "active_kv_reserved": self._active_kv_reserved,
            "active_kv_budget": self.max_active_kv_tokens,
            "prefill_mode": self.prefill_mode,
            "prefill_chunk_size": self.prefill_chunk_size,
            "prefill_chunks_processed": self._prefill_chunks_processed,
            "wave_sizes": dict(sorted(self._wave_sizes.items())),
            **self._metrics.snapshot(),
        }

    # --- Worker loop ---

    def _run(self) -> None:
        """Single-threaded scheduler loop."""
        device = self.backend.device_str
        try:
            while not self._stop_event.is_set():
                # If nothing in-flight and nothing pending, wait on the cv.
                if not self._active and not self._prefilling:
                    with self._pending_cv:
                        if self._pending_count == 0 and not self._stop_event.is_set():
                            self._pending_cv.wait(timeout=0.1)
                    if self._stop_event.is_set():
                        break

                with self.backend._lock:  # serialize against legacy generate/stream
                    # Order matters: admit (lookup-only when chunked) → advance one prefill
                    # chunk (may promote a row to _active with a fresh first_token) → evict
                    # processes those first_tokens this same iter → decode advances active.
                    self._admit_pending(device)
                    self._advance_prefill_chunk(device)
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
            row = self._active[i]
            self.policy.on_request_finished(row.request)
            self._evict_row(i)

    def _evict_row(self, idx: int) -> None:
        row = self._active.pop(idx)
        self._active_kv_reserved -= len(row.request.token_ids) + row.request.max_tokens
        self.backend.kv_release(len(row.request.token_ids), row.request.max_tokens)
        if self._batched_kv is not None:
            # Always route through the backend so paged caches can free blocks (no-op GC
            # for DynamicCache). Null afterward when the batch empties.
            self._batched_kv = self.backend.remove_row_from_cache(self._batched_kv, idx)
            if len(self._active) == 0:
                self._batched_kv = None
                self._attention_mask = None
            elif self._needs_mask:
                self._attention_mask = torch.cat([
                    self._attention_mask[:idx],
                    self._attention_mask[idx + 1:],
                ], dim=0)
        self._resolve(row.request)

    def _consume(self, request: "ScheduledRequest", plan: list) -> None:
        """Remove `request` from the policy and from this wave's plan (caller holds the lock)."""
        self.policy.pick(request)
        if plan and plan[0] is request:
            plan.pop(0)

    # --- Phase 2: admit new requests ---

    def _admit_pending(self, device: str) -> None:
        cache = self.backend.cache_adapter
        to_admit: list[ScheduledRequest] = []  # 'batched' mode: reserved reqs awaiting one prefill_batch
        plan: list[ScheduledRequest] = []
        if self.prefill_mode == "batched" and self.wave_window_mult > 0:
            # One prefill_batch pads every row to Smax, so choose a wave that pads well.
            # plan_wave always includes the window head → nobody is deferred past their turn.
            free = self.max_batch_size - len(self._active) - len(self._prefilling)
            if free > 0:
                with self._pending_cv:
                    window = self.policy.peek_window(free * self.wave_window_mult)
                plan = plan_wave(window, free)
        while len(self._active) + len(self._prefilling) + len(to_admit) < self.max_batch_size:
            with self._pending_cv:
                # HOL-wait: peek, KV-fit check, then consume only if it fits.
                peeked = plan[0] if plan else self.policy.peek_next()
                if peeked is None:
                    break  # break, not return: fall through to the batched-prefill flush below
                reservation = len(peeked.token_ids) + peeked.max_tokens

                # Active-KV gate: protects against decode-time OOM.
                if reservation > self.max_active_kv_tokens:
                    self._consume(peeked, plan)
                    self._pending_count -= 1
                    self._total_rejected += 1
                    err = QueueFullError(
                        f"request reserves {reservation} KV tokens, exceeds active budget "
                        f"({self.max_active_kv_tokens})"
                    )
                    self.policy.on_request_finished(peeked)
                    self._reject(peeked, err)
                    continue
                if self._active_kv_reserved + reservation > self.max_active_kv_tokens:
                    self._kv_admit_blocked += 1
                    break

                # Cache-pool gate: avoids forced eviction churn.
                if cache is not None:
                    needed = cache.blocks_needed(reservation)
                    if needed > cache.free_blocks:
                        if needed > cache.total_blocks:
                            self._consume(peeked, plan)
                            self._pending_count -= 1
                            self._total_rejected += 1
                            err = QueueFullError(
                                f"request needs {needed} KV blocks, exceeds cache capacity "
                                f"({cache.total_blocks})"
                            )
                            self.policy.on_request_finished(peeked)
                            self._reject(peeked, err)
                            continue
                        self._kv_admit_blocked += 1
                        break

                # Per-pool window-aware KV gate (custom backend; no-op default). Soft-hold if
                # this request's per-pool footprint doesn't fit every pool's free blocks.
                if not self.backend.kv_reserve(len(peeked.token_ids), peeked.max_tokens):
                    self._kv_admit_blocked += 1
                    break

                req = peeked
                self._consume(req, plan)
                self._pending_count -= 1
                self._active_kv_reserved += reservation
            if self.prefill_mode == "batched":
                to_admit.append(req)  # defer the forward; one prefill_batch after the loop
                continue
            try:
                if self.prefill_mode == "chunked":
                    # Chunked-prefill mode: lookup only, defer forward pass to _advance_prefill_chunk
                    partial_kv, matched = self.backend.prefill_lookup(req.token_ids, req.session_id)
                    req.cache_hit_tokens = self.backend.last_cache_hit_tokens
                    self._prefilling.append(_PrefillingRow(
                        request=req, matched=matched,
                        tokens_fed=matched, partial_kv=partial_kv,
                    ))
                    self._total_admitted += 1
                    continue

                kv, first_token, kv_len = self.backend.prefill(req.token_ids, req.session_id)
                req.cache_hit_tokens = self.backend.last_cache_hit_tokens
            except Exception as e:
                logger.exception("Prefill failed for session %s", req.session_id)
                self._active_kv_reserved -= reservation
                self.backend.kv_release(len(req.token_ids), req.max_tokens)
                self.policy.on_request_finished(req)
                self._reject(req, e)
                continue
            self._promote_to_decode(req, kv, kv_len, first_token, device)
            self._total_admitted += 1

        if to_admit:
            self._admit_batched(to_admit, device)

    def _admit_batched(self, reqs: list[ScheduledRequest], device: str) -> None:
        """Prefill all reserved reqs in ONE batched forward (prefill_mode='batched'). This is the
        TTFT lever: prefill is dispatch-bound (~4× a decode step), so a serial per-request admission
        wave costs N × that — batching pays the dispatch once. On batch failure, reject the whole
        wave (release each reservation) rather than risk partial/leaked state."""
        try:
            self._wave_sizes[len(reqs)] = self._wave_sizes.get(len(reqs), 0) + 1
            results = self.backend.prefill_batch([r.token_ids for r in reqs])
        except Exception as e:
            logger.exception("Batched prefill failed (%d reqs)", len(reqs))
            for req in reqs:
                self._active_kv_reserved -= len(req.token_ids) + req.max_tokens
                self.backend.kv_release(len(req.token_ids), req.max_tokens)
                self.policy.on_request_finished(req)
                self._reject(req, e)
            return
        for req, (kv, first_token, kv_len) in zip(reqs, results):
            req.cache_hit_tokens = 0  # batched path does a full prefill (no prefix-cache lookup)
            self._promote_to_decode(req, kv, kv_len, first_token, device)
            self._total_admitted += 1

    def _promote_to_decode(self, request: ScheduledRequest, kv: object, kv_len: int,
                           first_token: int, device: str) -> None:
        """Join a completed prefill's KV to the running decode batch (its first decode token).

        DISAGGREGATION SEAM: today this is a local splice on the same worker. Under P/D
        disaggregation the KV is produced on a prefill worker and TRANSFERRED here before the row
        decodes — this is the single point that handoff plugs into. The rest of the loop is already
        prefill/decode-decoupled (`_prefilling` phase + `prefill_chunk` primitive), so disagg adds a
        transfer here + a remote prefill loop, not a rewrite. Used by both monolithic admit and the
        chunked final-chunk path."""
        request.first_token_ts = time.perf_counter()
        self._splice_in(kv, kv_len, device)
        self._active.append(_ActiveRow(request=request, current_token=first_token, real_kv_len=kv_len))

    # --- Chunked prefill: advance one in-flight prefill by one chunk per iter ---

    def _advance_prefill_chunk(self, device: str) -> None:
        """Advance the head _prefilling row by one chunk. Promote to _active if final."""
        if not self._prefilling:
            return
        prow = self._prefilling[0]
        ids = prow.request.token_ids

        if prow.matched == len(ids):
            # Full cache hit — feed last token as a one-token primer (mirrors monolithic).
            chunk = [ids[-1]]
            is_final = True
        else:
            end = min(prow.tokens_fed + self.prefill_chunk_size, len(ids))
            chunk = ids[prow.tokens_fed:end]
            is_final = (end == len(ids))

        try:
            kv, last_token, kv_len = self.backend.prefill_chunk(chunk, prow.partial_kv, sampling=prow.request.sampling)
        except Exception as e:
            logger.exception("prefill_chunk failed for session %s", prow.request.session_id)
            self._prefilling.pop(0)
            self._active_kv_reserved -= len(prow.request.token_ids) + prow.request.max_tokens
            self.backend.kv_release(len(prow.request.token_ids), prow.request.max_tokens)
            self.policy.on_request_finished(prow.request)
            self._reject(prow.request, e)
            return

        prow.partial_kv = kv
        prow.tokens_fed += len(chunk)
        self._prefill_chunks_processed += 1
        self._metrics.record_chunk()

        if is_final:
            # Store the new KV portion in the cache (skip already-cached prefix).
            try:
                self.backend.prefill_store(ids, kv, prow.matched, prow.request.session_id)
            except Exception:
                logger.exception("prefill_store failed for session %s", prow.request.session_id)
            self._prefilling.pop(0)
            self._promote_to_decode(prow.request, kv, kv_len, last_token, device)

    def _splice_in(self, new_kv: object, new_kv_len: int, device: str) -> None:
        """Add a new row's KV to the batched cache; backend handles cache surgery."""
        first = self._batched_kv is None
        self._batched_kv = self.backend.splice_into_batched(
            None if first else self._batched_kv, new_kv, new_kv_len
        )
        if not self._needs_mask:
            return
        if first:
            self._attention_mask = torch.ones(1, new_kv_len, device=device, dtype=torch.long)
            return

        # Attention mask is torch-typed scheduler state; pad existing rows then append new row.
        # Drive sizing off the mask's OWN width, not kv_length: for paged backends
        # kv_length = max(per-row len) drops when the longest row is evicted, while the mask
        # width doesn't — so using kv_length here would cat mismatched widths and crash.
        existing_len = self._attention_mask.shape[1]
        max_len = max(existing_len, new_kv_len)
        existing_pad = max_len - existing_len
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
        if self._needs_mask:
            # Extend attention mask by one column for the new input token
            self._attention_mask = torch.cat([
                self._attention_mask,
                torch.ones(batch_size, 1, device=device, dtype=torch.long),
            ], dim=1)
        position_ids = torch.tensor(
            [[r.real_kv_len] for r in self._active],
            device=device, dtype=torch.long,
        )

        sampling_per_row = [r.request.sampling for r in self._active]
        next_tokens, self._batched_kv = self.backend.decode_step_batched(
            current_tokens, self._batched_kv, self._attention_mask, position_ids,
            sampling_per_row=sampling_per_row,
        )

        # One D2H copy for the whole batch; a per-row .item() is a separate device sync each.
        for row, tok in zip(self._active, next_tokens.tolist()):
            row.current_token = tok
            row.real_kv_len += 1
            self.policy.on_tokens_processed(row.request, 1)

    # --- Cross-thread helpers ---

    def _stream_token(self, request: ScheduledRequest, token: int) -> None:
        if request.token_queue is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(request.token_queue.put_nowait, token)

    def _resolve(self, request: ScheduledRequest) -> None:
        if self._loop is None:
            return
        result = list(request.generated)
        self._total_completed += 1
        if request.first_token_ts > 0 and request.enqueue_ts > 0:
            now = time.perf_counter()
            ttft = request.first_token_ts - request.enqueue_ts
            total = now - request.enqueue_ts
            n = len(result)
            tpot = (total - ttft) / (n - 1) if n > 1 else 0.0
            self._metrics.record(ttft, tpot, total, n)
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
        # QueueFullError = admission/queue reject; anything else = a failure (prefill error, crash).
        self._metrics.record_rejection(failed=not isinstance(exc, QueueFullError))
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
            self.backend.kv_release(len(row.request.token_ids), row.request.max_tokens)
            self.policy.on_request_finished(row.request)
            self._reject(row.request, exc)
        for prow in self._prefilling:
            self.backend.kv_release(len(prow.request.token_ids), prow.request.max_tokens)
            self.policy.on_request_finished(prow.request)
            self._reject(prow.request, exc)
        self._active.clear()
        self._prefilling.clear()
        self._active_kv_reserved = 0
        self._batched_kv = None
        self._attention_mask = None
