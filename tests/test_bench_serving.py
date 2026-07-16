"""Open-loop serving harness — SLO gating, percentiles, and the Poisson driver.

Drives the real harness against an in-process mock ASGI server (known delays) via
ASGITransport — no model, no port. Verifies TTFT/TPOT measurement, SLO gate, and knee logic."""

import asyncio
import json
import random
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import bench_serving as bs  # noqa: E402


def _row(rate, tok_per_s, within, ttft95=100.0, tpot95=20.0):
    return {"rate": rate, "n_ok": 10, "n_err": 0, "achieved_rps": rate, "tok_per_s": tok_per_s,
            "ttft_p50": 50.0, "ttft_p95": ttft95, "tpot_p50": 10.0, "tpot_p95": tpot95,
            "within_slo": within}


def test_slo_gate_from_samples():
    fast = bs.RateResult(rate=4.0, duration_s=10.0)
    fast.samples = [bs.Sample(ttft_s=0.05, tpot_s=0.02, out_tokens=10) for _ in range(20)]
    s = fast.summary()
    assert s["within_slo"] is True and s["n_ok"] == 20
    assert abs(s["ttft_p95"] - 50.0) < 1.0        # all 50ms

    slow = bs.RateResult(rate=32.0, duration_s=10.0)
    slow.samples = [bs.Sample(ttft_s=0.5, tpot_s=0.02, out_tokens=10) for _ in range(20)]
    assert slow.summary()["within_slo"] is False  # 500ms TTFT > 200 SLO

    slow_tpot = bs.RateResult(rate=32.0, duration_s=10.0)
    slow_tpot.samples = [bs.Sample(ttft_s=0.05, tpot_s=0.08, out_tokens=10) for _ in range(20)]
    assert slow_tpot.summary()["within_slo"] is False  # 80ms/tok TPOT > 50 SLO


def test_report_picks_best_and_knee(capsys):
    rows = [_row(2, 100, True), _row(4, 200, True),
            _row(8, 250, False, ttft95=420.0), _row(16, 260, False, ttft95=900.0)]
    out = bs.report(rows)
    assert out["max_slo_tok_s"] == 200    # best within-SLO throughput (not the higher 250/260)
    assert out["knee_rate"] == 8          # first SLO violation


def test_open_loop_driver_measures_against_mock():
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions():
        async def gen():
            await asyncio.sleep(0.02)                       # TTFT ~20ms
            for _ in range(5):
                yield f'data: {json.dumps({"choices": [{"text": "x", "finish_reason": None}]})}\n\n'
                await asyncio.sleep(0.005)                  # ITL ~5ms
            yield f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": 5}})}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await bs.run_rate(client, rate=20.0, duration_s=0.5, rng=random.Random(0))

    res = asyncio.run(go())
    ok = [s for s in res.samples if s.error is None]
    assert len(ok) >= 3                                     # open-loop fired several requests
    assert all(s.out_tokens == 5 for s in ok)              # counted token chunks, not usage chunk
    assert all(0.01 < s.ttft_s < 0.2 for s in ok)          # measured the ~20ms TTFT
    assert all(0.0 < s.tpot_s < 0.05 for s in ok)          # measured the ~5ms ITL
