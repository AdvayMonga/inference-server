"""The engine in its own process (ENGINE_PROCESS=1): the backend and the scheduler run in a child process, the HTTP
server, tokenizer and streams in this one, joined by a pipe. In one process the asyncio thread (per-token
detokenizing, SSE encoding for every stream) and the scheduler thread share the GIL; at 256 streams the scheduler
thread was 58% active and the GPU 85% busy against vLLM's 98% (H200, 2026-10-19), and vLLM keeps them apart the
same way.

Engine side: the real ContinuousBatchScheduler, with stand-ins for its event loop, token queues and futures that
write into an outbox; each evict pass (the scheduler hands a step's tokens over at once) becomes one message.
API side: `EngineClient`, with the scheduler calls the HTTP routes use (enqueue, submit, stats, stop) and a
`backend` / `cache_adapter` that forward logprob scoring and cache stats.

Messages API -> engine: ("submit", rid, fields), ("cancel", rid), ("score", qid, token_ids, k), ("stop",).
Engine -> API: ("ready",), ("tokens", [(rid, item)]) with item a token id, ("meta", cache_hit_tokens),
("error", kind, text) or None (end of stream), ("stats", stats, cache_info), ("score", qid, result)."""

from __future__ import annotations

import asyncio
import importlib
import itertools
import logging
import multiprocessing as mp
import threading
import time

logger = logging.getLogger(__name__)

STATS_EVERY_S = 0.5


# ---------------------------------------------------------------------------------------- engine side
class _Outbox:
    def __init__(self, conn, lock):
        self.conn, self.lock, self.items = conn, lock, []

    def flush(self) -> None:
        with self.lock:
            if self.items:
                items, self.items = self.items, []
                self.conn.send(("tokens", items))


class _Loop:
    """The scheduler's event loop: callbacks run at once (on the scheduler thread), then the outbox is sent."""

    def __init__(self, outbox: _Outbox):
        self.outbox = outbox

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)
        self.outbox.flush()


class _Queue:
    """A request's token queue: tokens and the end marker into the outbox, the cache hit with the first token."""

    def __init__(self, rid: int, outbox: _Outbox, live: dict):
        self.rid, self.outbox, self.live, self.req, self.sent = rid, outbox, live, None, False

    def put_nowait(self, item) -> None:
        if not self.sent and item is not None and self.req is not None:
            self.outbox.items.append((self.rid, ("meta", self.req.cache_hit_tokens)))
        self.sent = True
        self.outbox.items.append((self.rid, item))
        if item is None:
            self.live.pop(self.rid, None)


class _Future:
    def __init__(self, rid: int, outbox: _Outbox):
        self.rid, self.outbox, self._done = rid, outbox, False

    def done(self) -> bool:
        return self._done

    def set_result(self, result) -> None:
        self._done = True

    def set_exception(self, exc: BaseException) -> None:
        self._done = True
        self.outbox.items.append((self.rid, ("error", type(exc).__name__, str(exc))))


def _default_factory():
    from inference_server.config import settings
    from inference_server.server import build_backend, build_scheduler
    backend, cache = build_backend(settings)
    return backend, build_scheduler(backend, settings), cache


def _engine_main(conn, factory: str | None) -> None:
    """The child: build the engine, run the scheduler on its own thread, serve the pipe on this one."""
    from inference_server.sampling import SamplingParams
    from inference_server.scheduler import QueueFullError, RequestTooLargeError, ScheduledRequest
    if factory:
        mod, fn = factory.split(":")
        backend, sched, cache = getattr(importlib.import_module(mod), fn)()
    else:
        backend, sched, cache = _default_factory()
    lock = threading.Lock()
    outbox = _Outbox(conn, lock)
    live: dict[int, ScheduledRequest] = {}
    sched._loop = _Loop(outbox)
    sched._stop_event.clear()
    sched._worker = threading.Thread(target=sched._run, name="cb-scheduler", daemon=True)
    sched._worker.start()
    stop = threading.Event()

    def push_stats():
        while not stop.wait(STATS_EVERY_S):
            info = getattr(cache, "hit_rate_info", None) if cache is not None else None
            with lock:
                conn.send(("stats", sched.stats(), info))

    threading.Thread(target=push_stats, name="engine-stats", daemon=True).start()

    def score(qid, token_ids, k):
        try:
            result = backend.score_logprobs(token_ids, k)
        except Exception as e:  # noqa: BLE001 - the error goes back to the caller
            result = e
        with lock:
            conn.send(("score", qid, result))

    with lock:
        conn.send(("ready",))
    while True:
        msg = conn.recv()
        kind = msg[0]
        if kind == "submit":
            rid, f = msg[1], msg[2]
            q = _Queue(rid, outbox, live)
            req = ScheduledRequest(token_ids=f["token_ids"], max_tokens=f["max_tokens"], session_id=f["session_id"],
                                   future=_Future(rid, outbox), token_queue=q,
                                   sampling=SamplingParams(**f["sampling"]), trace_id=f["trace_id"],
                                   turn_index=f["turn_index"], arrival_ts=f["arrival_ts"], priority=f["priority"])
            q.req = req
            live[rid] = req
            try:
                sched.enqueue(req)
            except (QueueFullError, RequestTooLargeError) as e:
                live.pop(rid, None)
                with lock:
                    conn.send(("tokens", [(rid, ("error", type(e).__name__, str(e))), (rid, None)]))
        elif kind == "cancel":
            req = live.get(msg[1])
            if req is not None:
                req.cancelled = True
        elif kind == "score":
            threading.Thread(target=score, args=msg[1:], daemon=True).start()
        elif kind == "stop":
            stop.set()
            sched._stop_event.set()
            with sched._pending_cv:
                sched._pending_cv.notify_all()
            sched._worker.join()
            break


# ---------------------------------------------------------------------------------------- API side
class _BackendProxy:
    """What the HTTP routes need from the backend: logprob scoring (called from an executor thread)."""

    def __init__(self, client: "EngineClient"):
        self.client = client

    def score_logprobs(self, token_ids, k):
        qid = next(self.client._ids)
        ev, box = threading.Event(), {}
        self.client._scores[qid] = (ev, box)
        self.client._send(("score", qid, token_ids, k))
        ev.wait()
        result = box["result"]
        if isinstance(result, BaseException):
            raise result
        return result


class _CacheProxy:
    def __init__(self, client: "EngineClient"):
        self.client = client

    @property
    def hit_rate_info(self):
        return self.client._cache_info


class EngineClient:
    """Stands in for the scheduler on the HTTP side; the engine runs in a child process (`factory`: "module:fn"
    returning (backend, scheduler, cache adapter) there, default the served configuration)."""

    def __init__(self, factory: str | None = None):
        self.factory = factory
        self._ids = itertools.count(1)
        self._reqs: dict = {}
        self._scores: dict = {}
        self._stats: dict = {}
        self._cache_info = None
        self._send_lock = threading.Lock()
        self.backend = _BackendProxy(self)
        self.cache_adapter = _CacheProxy(self)
        self._proc = None

    # lifecycle
    def launch(self, timeout_s: float = 3600.0) -> None:
        """Start the child and wait until its engine is built (blocking: run it in an executor)."""
        ctx = mp.get_context("spawn")                          # CUDA in the child only
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_engine_main, args=(child, self.factory), name="engine", daemon=True)
        self._proc.start()
        if not self._conn.poll(timeout_s) or self._conn.recv()[0] != "ready":
            raise RuntimeError("engine process did not come up")

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._reader = threading.Thread(target=self._read, name="engine-reader", daemon=True)
        self._reader.start()
        self._cancel_task = self._loop.create_task(self._watch_cancels())

    async def stop(self) -> None:
        if self._proc is None:
            return
        self._cancel_task.cancel()
        self._send(("stop",))
        await self._loop.run_in_executor(None, self._proc.join, 60)
        self._proc = None

    # the scheduler calls the routes make
    def enqueue(self, req) -> None:
        rid = next(self._ids)
        req.enqueue_ts = time.perf_counter()
        self._reqs[rid] = req
        s = req.sampling
        self._send(("submit", rid, {
            "token_ids": list(req.token_ids), "max_tokens": req.max_tokens, "session_id": req.session_id,
            "sampling": {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k},
            "trace_id": req.trace_id, "turn_index": req.turn_index, "arrival_ts": req.arrival_ts,
            "priority": req.priority}))

    async def submit(self, req) -> list[int]:
        self.enqueue(req)
        return await req.future

    def stats(self) -> dict:
        return dict(self._stats)

    # plumbing
    def _send(self, msg) -> None:
        with self._send_lock:
            self._conn.send(msg)

    def _read(self) -> None:
        while True:
            try:
                msg = self._conn.recv()
            except (EOFError, OSError):
                self._loop.call_soon_threadsafe(self._engine_gone)
                return
            if msg[0] == "score":
                ev, box = self._scores.pop(msg[1])
                box["result"] = msg[2]
                ev.set()
            elif msg[0] == "stats":
                self._stats, self._cache_info = msg[1], msg[2]
            else:
                self._loop.call_soon_threadsafe(self._dispatch, msg)

    def _dispatch(self, msg) -> None:
        from inference_server.scheduler import QueueFullError, RequestTooLargeError
        if msg[0] != "tokens":
            return
        for rid, item in msg[1]:
            req = self._reqs.get(rid)
            if req is None:
                continue
            if item is None:                                   # end of stream
                del self._reqs[rid]
                if not req.future.done():
                    req.future.set_result(list(req.generated))
                if req.token_queue is not None:
                    req.token_queue.put_nowait(None)
            elif isinstance(item, tuple) and item[0] == "meta":
                req.cache_hit_tokens = item[1]
            elif isinstance(item, tuple):                      # ("error", kind, text): ahead of the end marker
                cls = {"QueueFullError": QueueFullError, "RequestTooLargeError": RequestTooLargeError}.get(item[1],
                                                                                                         RuntimeError)
                if not req.future.done():
                    req.future.set_exception(cls(item[2]))
            else:
                if not req.first_token_ts:
                    req.first_token_ts = time.perf_counter()
                req.generated.append(item)
                if req.token_queue is not None:
                    req.token_queue.put_nowait(item)

    def _engine_gone(self) -> None:
        err = RuntimeError("engine process exited")
        for rid, req in list(self._reqs.items()):
            if not req.future.done():
                req.future.set_exception(err)
            if req.token_queue is not None:
                req.token_queue.put_nowait(None)
        self._reqs.clear()

    async def _watch_cancels(self) -> None:
        """Requests whose client went away (the routes set `cancelled`): the engine stops their rows."""
        told: set[int] = set()
        while True:
            await asyncio.sleep(0.05)
            for rid, req in list(self._reqs.items()):
                if req.cancelled and rid not in told:
                    told.add(rid)
                    self._send(("cancel", rid))
            told &= set(self._reqs)
