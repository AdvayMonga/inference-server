"""The closed-loop load generator: samplers, aggregation, and the driver against a mock server.

Closed-loop means N users each fire one request at a time and think between them, so the
offered load is a RESPONSE to server speed. That is the opposite of bench_serving.py's
open-loop Poisson driver, and the property these tests pin is exactly that: never more than N
in flight, one session per user, and throughput measured over the window that was actually
observed."""

import asyncio
import json
import random
import sys
from functools import partial
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "bench"))
import load_test as lt  # noqa: E402
import prompt_bank  # noqa: E402


# ---- samplers -------------------------------------------------------------------------

def test_mixed_sampler_matches_bin_weights_and_shapes():
    rng = random.Random(7)
    draws = [lt.sample_mixed(rng) for _ in range(4000)]
    by_max = {mx: 0 for _, _, mx in lt.MIXED_BINS}
    for prompt, mx in draws:
        by_max[mx] += 1
        p_toks = next(p for _, p, m in lt.MIXED_BINS if m == mx)
        assert len(prompt) == p_toks * 4          # ~4 chars per token
    for w, _, mx in lt.MIXED_BINS:
        assert abs(by_max[mx] / len(draws) - w) < 0.04, (mx, by_max[mx] / len(draws), w)


def test_realistic_sampler_draws_a_bank_prompt_with_its_buckets_range():
    bucket_of = {}
    for name, prompts, _ in prompt_bank.PROMPT_MIX:
        for p in prompts:
            bucket_of[p] = name
    rng = random.Random(3)
    seen = {name: 0 for name, _, _ in prompt_bank.PROMPT_MIX}
    n = 3000
    for _ in range(n):
        prompt, mx = lt.sample_realistic(rng)
        name = bucket_of[prompt]                  # KeyError = a prompt not in the bank
        lo, hi = prompt_bank.MAX_TOKENS_RANGE[name]
        assert lo <= mx <= hi
        seen[name] += 1
    for name, _, w in prompt_bank.PROMPT_MIX:
        assert abs(seen[name] / n - w) < 0.04, (name, seen[name] / n, w)


def test_build_prompt_hits_its_target_and_shares_a_prefix():
    short, long = lt.build_prompt(60), lt.build_prompt(500)
    assert len(short) == 240 and len(long) == 2000
    assert long.startswith(short)                 # why short/long/mixed are cache-HIT workloads


# ---- aggregation ----------------------------------------------------------------------

def _ok(ttft, tpot, total, tokens):
    return lt.Sample(ttft_s=ttft, tpot_s=tpot, total_s=total, tokens=tokens)


def test_summary_percentiles_and_error_counting():
    res = lt.LevelResult(workload="short", concurrency=4, duration_s=10.0)
    res.samples = [_ok(0.010 * i, 0.002 * i, 0.5, 10) for i in range(1, 11)]
    res.samples += [lt.Sample(0, 0, 0, 0, error="http 429")] * 3
    s = res.summary()
    assert s["n_ok"] == 10 and s["n_err"] == 3
    assert s["ttft_p50"] == pytest.approx(55.0)   # interpolated median of 10..100 ms
    assert s["ttft_p95"] == pytest.approx(95.5)
    assert s["tpot_p50"] == pytest.approx(11.0)
    assert s["wall_s"] == 10.0                    # falls back to the nominal window when unset


def test_summary_of_all_errors_is_zeros_not_a_crash():
    res = lt.LevelResult(workload="short", concurrency=1, duration_s=5.0)
    res.samples = [lt.Sample(0, 0, 0, 0, error="boom")] * 2
    s = res.summary()
    assert s["n_ok"] == 0 and s["n_err"] == 2 and s["tok_per_s"] == 0


def test_throughput_is_measured_over_the_wall_clock_not_the_nominal_window():
    """Workers finish the request in flight at the deadline, so completions land after it.
    Counting them over the nominal duration overstated throughput."""
    res = lt.LevelResult(workload="long", concurrency=2, duration_s=1.0, wall_s=2.0)
    res.samples = [_ok(0.1, 0.01, 0.9, 10) for _ in range(10)]
    s = res.summary()
    assert s["req_per_s"] == pytest.approx(5.0)   # 10 / 2.0s, not 10 / 1.0s
    assert s["tok_per_s"] == pytest.approx(50.0)
    assert s["wall_s"] == 2.0


# ---- the driver, against an in-process server ------------------------------------------

def _mock_server(n_tokens=4, ttft_s=0.01, itl_s=0.002, status=200):
    """A /generate that speaks the server's SSE dialect and records what it sees."""
    app = FastAPI()
    seen = {"sessions": set(), "in_flight": 0, "max_in_flight": 0, "requests": 0}

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json()
        seen["requests"] += 1
        seen["sessions"].add(body.get("session_id", "default"))
        if status != 200:
            return JSONResponse({"detail": "queue full"}, status_code=status)

        async def gen():
            seen["in_flight"] += 1
            seen["max_in_flight"] = max(seen["max_in_flight"], seen["in_flight"])
            try:
                await asyncio.sleep(ttft_s)
                yield f"data: {json.dumps({'ttft_ms': ttft_s * 1000, 'prompt_tokens': 5, 'cache_hit_tokens': 0})}\n\n"
                for i in range(n_tokens):
                    yield f"data: tok{i}\n\n"
                    await asyncio.sleep(itl_s)
                yield "data: [DONE]\n\n"
            finally:
                seen["in_flight"] -= 1
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app, seen


def _run_level_against(app, monkeypatch, **kw):
    transport = httpx.ASGITransport(app=app)
    monkeypatch.setattr(lt.httpx, "AsyncClient", partial(httpx.AsyncClient, transport=transport))
    return asyncio.run(lt.run_level("http://test", **kw))


def test_closed_loop_never_exceeds_n_in_flight_and_gives_each_user_a_session(monkeypatch):
    app, seen = _mock_server(n_tokens=4)
    res = _run_level_against(app, monkeypatch, workload="short", concurrency=3, duration_s=0.3)
    ok = [s for s in res.samples if s.error is None]
    assert len(ok) >= 6                                      # 3 users, several rounds each
    assert seen["max_in_flight"] == 3                        # never more than N: closed loop
    assert seen["sessions"] == {"load-0", "load-1", "load-2"}  # N users, not one "default"
    assert all(s.tokens == 4 for s in ok)                    # counted tokens, not the meta line
    assert all(0.005 < s.ttft_s < 0.2 for s in ok)
    assert all(0.0 < s.tpot_s < 0.05 for s in ok)
    assert res.wall_s >= res.duration_s                      # the in-flight tail is inside the window


def test_realistic_workload_drives_the_prompt_bank_per_user(monkeypatch):
    app, seen = _mock_server(n_tokens=2, ttft_s=0.001, itl_s=0.0)
    res = _run_level_against(app, monkeypatch, workload="realistic", concurrency=2, duration_s=0.2)
    assert seen["requests"] >= 4 and all(s.error is None for s in res.samples)


def test_rejected_users_back_off_instead_of_hot_looping(monkeypatch):
    app, seen = _mock_server(status=429)
    monkeypatch.setattr(lt, "ERROR_BACKOFF_S", 0.1)
    res = _run_level_against(app, monkeypatch, workload="short", concurrency=1, duration_s=0.35)
    s = res.summary()
    assert s["n_ok"] == 0
    assert 2 <= s["n_err"] <= 5                              # ~0.35s / 0.1s backoff, not thousands
