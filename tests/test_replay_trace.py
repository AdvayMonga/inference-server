"""Open-loop trace replay against an in-process mock server: the schedule is the input.

Same ASGITransport pattern as test_bench_serving.py — no model, no port."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from inference_server.research import chat_template as CT
from inference_server.research import harness as H
from inference_server.research.compare import comparable
from inference_server.research.corpus import TraceRequest, WorkloadClass
from inference_server.research.schemas import Vitals

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "bench"))
import replay_trace as rt  # noqa: E402

CLS = WorkloadClass("steady_interactive", "", 200.0, 50.0, 4.0, "a", "b")
MODEL = "fake/instruct-model"
TEMPLATE_OVERHEAD = 7        # tokens the mock's "chat template" adds around the prompt


class FakeTokenizer:
    """1 char = 1 token, plus a fixed template wrapper — the mock server's arithmetic."""
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, *, enable_thinking=True, **kw):
        n = len(messages[0]["content"]) + TEMPLATE_OVERHEAD + (0 if enable_thinking else 2)
        return {"input_ids": list(range(n))}


def _fake_tokenizer(monkeypatch, tokenizer=None):
    monkeypatch.setitem(CT._CACHE, MODEL, FakeTokenizer() if tokenizer is None else tokenizer)


def _trace(n: int, gap: float) -> list[TraceRequest]:
    return [TraceRequest(round(i * gap, 3), f"s-{i}", 0, f"Case {i}. hi", 8) for i in range(n)]


def _app(ttft_s: float = 0.0, itl_s: float = 0.0, n_tokens: int = 5,
         release: asyncio.Event | None = None):
    app = FastAPI()
    app.state.arrived = 0
    app.state.requests = []     # (headers, body) of every arrival, like the real shim sees them

    async def _serve(request: Request, chat: bool):
        app.state.arrived += 1
        body = await request.json()
        app.state.requests.append((dict(request.headers), body))
        echo = {"X-Trace-Id": request.headers.get("x-trace-id", "minted")}
        # The chat route encodes a TEMPLATED prompt, so it reports more prompt tokens than the
        # raw one. That is what a replay checks its chat_template fingerprint against.
        text = body["messages"][0]["content"] if chat else body["prompt"]
        prompt_tokens = len(text) + (TEMPLATE_OVERHEAD if chat else 0)

        def token_chunk():
            payload = ({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]} if chat
                       else {"choices": [{"text": "x", "finish_reason": None}]})
            return f"data: {json.dumps(payload)}\n\n"

        async def gen():
            if release is not None:
                await release.wait()
            await asyncio.sleep(ttft_s)
            for _ in range(n_tokens):
                yield token_chunk()
                await asyncio.sleep(itl_s)
            usage = {"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                              "completion_tokens": n_tokens}}
            yield f"data: {json.dumps(usage)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers=echo)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _serve(request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _serve(request, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}

    @app.get("/scheduler/stats")
    async def sched():
        return {"active_high_water": 3, "wave_sizes": {1: 4}}

    @app.get("/cache/stats")
    async def cache():
        return {"hit_rate": 0.0, "lookups": 6}

    return app


async def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_arrivals_follow_the_trace_clock_not_the_server():
    """Six arrivals 50ms apart must all have fired by 300ms even though the server has answered
    none of them. A closed loop would be stuck on the first."""
    release = asyncio.Event()
    app = _app(release=release)

    async def go():
        async with await _client(app) as client:
            task = asyncio.create_task(rt.run_replay(client, _trace(6, 0.05)))
            await asyncio.sleep(0.4)
            assert app.state.arrived == 6, "every arrival fired while nothing completed"
            release.set()
            return await task

    res = asyncio.run(go())
    assert [r.index for r in res.rows] == list(range(6))
    for r in res.rows:
        assert abs(r.fired_s - r.arrival_s) < 0.04, (r.arrival_s, r.fired_s)
    assert min(r.ttft_ms for r in res.rows) > 100          # the last arrival waited ~150ms
    assert res.wall_s > 0.4                                 # drain is in the window


def test_request_still_open_at_drain_timeout_counts_as_an_error():
    """Dropping it would inflate n_ok exactly under the overload the replay exists to measure."""
    app = _app(ttft_s=0.01)
    stuck = asyncio.Event()

    @app.post("/v1/stuck")
    async def never():
        await stuck.wait()

    calls = {"n": 0}
    real = rt.one_request

    async def one_stuck(client, prompt, max_tokens, **kw):
        calls["n"] += 1
        if calls["n"] == 2:                     # the second arrival never gets a token
            await client.post("/v1/stuck")
        return await real(client, prompt, max_tokens, **kw)

    async def go():
        rt.one_request = one_stuck
        try:
            async with await _client(app) as client:
                return await rt.run_replay(client, _trace(3, 0.01), drain_timeout_s=0.3)
        finally:
            rt.one_request = real
            stuck.set()

    res = asyncio.run(go())
    assert [r.index for r in res.rows] == [0, 1, 2]
    late = res.rows[1]
    assert late.error == "drain_timeout" and late.ttft_ms is None and late.out_tokens == 0
    assert late.trace_id == "replay-1"                      # a failed row still has its key
    assert abs(late.fired_s - 0.01) < 0.04
    s = res.summary(CLS)
    assert s["n_ok"] == 2 and s["n_err"] == 1
    assert 0.3 < res.wall_s < 1.0


def test_rate_scale_compresses_the_schedule():
    async def go():
        async with await _client(_app()) as client:
            return await rt.run_replay(client, _trace(5, 0.1), rate_scale=2.0)

    res = asyncio.run(go())
    for r in res.rows:
        assert abs(r.fired_s - r.arrival_s / 2.0) < 0.04
    assert res.wall_s < 0.4                                 # 0.4s of trace replayed in ~0.2s


def test_client_measures_ttft_tpot_and_class_slo_judges():
    async def go():
        async with await _client(_app(ttft_s=0.02, itl_s=0.005)) as client:
            return await rt.run_replay(client, _trace(4, 0.02))

    res = asyncio.run(go())
    s = res.summary(CLS)
    assert s["n_ok"] == 4 and s["n_err"] == 0
    # ASGITransport buffers the streamed body, so ITL is not observable in-process (same limit
    # as test_bench_serving); TTFT is, and it absorbs the 5 x 5ms token gaps.
    assert 15 < s["ttft_p95"] < 150 and s["tpot_p95"] is not None
    assert s["within_slo"] is True
    assert res.summary(WorkloadClass("tight", "", 10.0, 50.0, 1.0, "a", "b"))["within_slo"] is False
    slow_decode = rt.ReplayResult(rows=[rt.Row(i, "s", 0, f"t-{i}", 0.0, 0.0, 20.0, 60.0, 5, None)
                                        for i in range(4)], wall_s=1.0)
    assert slow_decode.summary(CLS)["within_slo"] is False           # TPOT arm: 60 >= 50
    assert slow_decode.summary(WorkloadClass("no_tpot", "", 200.0, None, 1.0, "a", "b"))["within_slo"]


def test_panel_is_stamped_with_corpus_version_and_written_with_its_rows(tmp_path):
    app = _app(ttft_s=0.01)

    async def go():
        async with await _client(app) as client:
            res = await rt.run_replay(client, _trace(3, 0.01))
            return res, await rt.fetch_stats(client, "/scheduler/stats"), \
                await rt.fetch_stats(client, "/cache/stats"), \
                await rt.fetch_stats(client, "/nope")

    res, sched, cache, missing = asyncio.run(go())
    assert missing == {}
    panel = rt.build_panel(res, CLS, corpus_version="v1" * 8, split="seen", rate_scale=1.0,
                           scheduler_stats=sched, cache_stats=cache, trace_prefix="x-seen-ab12cd")
    assert panel.validity.harness == "replay_trace"
    assert panel.validity.harness_config["trace_prefix"] == "x-seen-ab12cd"   # the join key
    assert panel.validity.corpus_version == "v1" * 8
    assert panel.validity.workload_class == "steady_interactive"
    assert panel.validity.harness_config["split"] == "seen"
    assert panel.validity.workload_regime == "cache_miss_heavy"   # from the engine's hit_rate
    assert panel.validity.concurrency_observed == 3
    assert panel.slo_ttft_ms == 200.0 and panel.tok_s_within_slo is not None

    other = rt.build_panel(res, CLS, corpus_version="v2" * 8, split="seen", rate_scale=1.0,
                           scheduler_stats=sched, cache_stats=cache)
    assert not comparable(panel, other), "a different corpus version is not comparable"

    path = H.emit(panel, runs_dir=tmp_path)
    rt.write_rows(path.with_suffix(".csv"), res.rows)
    assert Vitals.load(path).validity.corpus_version == "v1" * 8
    with open(path.with_suffix(".csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and rows[0]["session_id"] == "s-0" and rows[0]["error"] == ""
    assert [r["trace_id"] for r in rows] == ["replay-0", "replay-1", "replay-2"]


def test_replay_sends_the_trace_session_turn_and_trace_id_and_its_sampling():
    """turn 1 of session s-3 must reach the engine AS session s-3 turn 1, not as a fresh
    bench-N session; and the request key must be the one the CSV row carries."""
    app = _app()
    trace = [TraceRequest(0.0, "s-3", 0, "Case 3. hi", 8),
             TraceRequest(0.01, "s-3", 1, "Case 3. hi. and then?", 8,
                          sampling={"temperature": 0.7, "top_p": 0.9, "top_k": 40})]

    async def go():
        async with await _client(app) as client:
            return await rt.run_replay(client, trace, trace_prefix="steady_interactive-seen")

    res = asyncio.run(go())
    (h0, b0), (h1, b1) = app.state.requests
    assert (h0["x-session-id"], h0["x-turn-index"], h0["x-trace-id"]) == \
        ("s-3", "0", "steady_interactive-seen-0")
    assert (h1["x-session-id"], h1["x-turn-index"], h1["x-trace-id"]) == \
        ("s-3", "1", "steady_interactive-seen-1")
    assert (b0["temperature"], b0["top_p"], b0["top_k"]) == (0.0, 1.0, 0)      # trace default
    assert (b1["temperature"], b1["top_p"], b1["top_k"]) == (0.7, 0.9, 40)     # trace sampling
    assert [r.trace_id for r in res.rows] == ["steady_interactive-seen-0", "steady_interactive-seen-1"]


def test_slo_broken_means_no_throughput_within_slo():
    async def go():
        async with await _client(_app(ttft_s=0.05)) as client:
            return await rt.run_replay(client, _trace(3, 0.01))

    res = asyncio.run(go())
    tight = WorkloadClass("tight", "", 10.0, None, 1.0, "a", "b")
    panel = rt.build_panel(res, tight, corpus_version="v", split="heldout", rate_scale=1.0,
                           scheduler_stats={}, cache_stats={})
    assert panel.tok_s_within_slo is None and panel.slo_tpot_ms is None
    assert panel.validity.workload_regime == "synthetic"        # no cache stats: say so


def test_trace_prefix_is_unique_per_invocation():
    """Two replays against one server process share one telemetry file; ids must not collide."""
    a, b = rt.new_trace_prefix("steady_interactive", "seen"), rt.new_trace_prefix("steady_interactive", "seen")
    assert a != b and a.startswith("steady_interactive-seen-") and len(a.split("-")[-1]) == 6


# ----------------------------------------------------- chat templating (kb-20260919-6ef4e6bf)

def test_replay_posts_to_the_chat_route_by_default_and_says_so_in_the_panel(monkeypatch):
    """A corpus prompt sent raw to an instruct model is answered with nothing; the whole point
    of the chat route is that the model sees a templated turn instead."""
    _fake_tokenizer(monkeypatch)
    app = _app()
    trace = _trace(3, 0.01)

    async def go():
        async with await _client(app) as client:
            return await rt.run_replay(client, trace)

    res = asyncio.run(go())
    assert all(b.get("messages") == [{"role": "user", "content": t.prompt}]
               for (_, b), t in zip(app.state.requests, trace))
    assert all(r.out_tokens == 5 for r in res.rows), "delta.content chunks are counted"
    # The templated length the server reported, not the trace's own byte count.
    assert [r.prompt_tokens for r in res.rows] == \
        [len(t.prompt) + TEMPLATE_OVERHEAD for t in trace]

    panel = rt.build_panel(res, CLS, corpus_version="v1" * 8, split="seen", rate_scale=1.0,
                           scheduler_stats={}, cache_stats={}, model_name=MODEL, trace=trace)
    assert panel.validity.harness_config["prompt_format"] == "chat"
    fp = panel.validity.chat_template
    assert fp["tokenizer"] == MODEL and fp["enable_thinking"] is True
    assert fp["probe_tokens"] == len("probe") + TEMPLATE_OVERHEAD
    assert fp["verified"] is True, "the stamp was checked against the server that served the run"


def test_raw_route_is_still_reachable_and_never_compares_against_a_templated_run(monkeypatch):
    """The trace bytes are identical on both routes, so corpus_version cannot tell them apart.
    prompt_format is what refuses the comparison."""
    _fake_tokenizer(monkeypatch)
    app = _app()
    trace = _trace(2, 0.01)

    async def go(fmt):
        async with await _client(app) as client:
            return await rt.run_replay(client, trace, prompt_format=fmt)

    def panel(fmt):
        return rt.build_panel(asyncio.run(go(fmt)), CLS, corpus_version="v1" * 8, split="seen",
                              rate_scale=1.0, scheduler_stats={}, cache_stats={},
                              prompt_format=fmt, model_name=MODEL, trace=trace)

    raw, chat = panel("raw"), panel("chat")
    assert raw.validity.chat_template is None, "nothing is templated on the raw route"
    assert raw.validity.corpus_version == chat.validity.corpus_version
    cmp_ = comparable(raw, chat, same_group_required=False)
    assert not cmp_ and any("prompt_format" in r for r in cmp_.reasons)
    assert app.state.requests[0][1].get("prompt") == trace[0].prompt   # raw sent bytes as-is


def test_a_tokenizer_that_disagrees_with_the_server_marks_the_panel_unverified(monkeypatch):
    """The fingerprint is computed client-side; the server is what applies the template. A panel
    whose stamp describes a different tokenizer is worse than one with no stamp."""

    class Drifted(FakeTokenizer):
        def apply_chat_template(self, messages, *, enable_thinking=True, **kw):
            return {"input_ids": list(range(len(messages[0]["content"]) + 99))}

    _fake_tokenizer(monkeypatch, Drifted())
    app = _app()
    trace = _trace(2, 0.01)

    async def go():
        async with await _client(app) as client:
            return await rt.run_replay(client, trace)

    panel = rt.build_panel(asyncio.run(go()), CLS, corpus_version="v1" * 8, split="seen",
                           rate_scale=1.0, scheduler_stats={}, cache_stats={},
                           model_name=MODEL, trace=trace)
    assert panel.validity.chat_template["verified"] is False
    cmp_ = comparable(panel, panel, same_group_required=False)
    assert not cmp_ and any("verification" in r for r in cmp_.reasons)
