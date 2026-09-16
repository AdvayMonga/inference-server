"""Open-loop trace replay against an in-process mock server: the schedule is the input.

Same ASGITransport pattern as test_bench_serving.py — no model, no port."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from inference_server.research import harness as H
from inference_server.research.compare import comparable
from inference_server.research.corpus import TraceRequest, WorkloadClass
from inference_server.research.schemas import Vitals

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "bench"))
import replay_trace as rt  # noqa: E402

CLS = WorkloadClass("steady_interactive", "", 200.0, 50.0, 4.0, "a", "b")


def _trace(n: int, gap: float) -> list[TraceRequest]:
    return [TraceRequest(round(i * gap, 3), f"s-{i}", 0, f"Case {i}. hi", 8) for i in range(n)]


def _app(ttft_s: float = 0.0, itl_s: float = 0.0, n_tokens: int = 5,
         release: asyncio.Event | None = None):
    app = FastAPI()
    app.state.arrived = 0

    @app.post("/v1/completions")
    async def completions():
        app.state.arrived += 1

        async def gen():
            if release is not None:
                await release.wait()
            await asyncio.sleep(ttft_s)
            for _ in range(n_tokens):
                yield f'data: {json.dumps({"choices": [{"text": "x", "finish_reason": None}]})}\n\n'
                await asyncio.sleep(itl_s)
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

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
    slow_decode = rt.ReplayResult(rows=[rt.Row(i, "s", 0, 0.0, 0.0, 20.0, 60.0, 5, None)
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
                           scheduler_stats=sched, cache_stats=cache)
    assert panel.validity.harness == "replay_trace"
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
