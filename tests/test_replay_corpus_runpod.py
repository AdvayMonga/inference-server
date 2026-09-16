"""The RunPod corpus-replay instrument, driven against an in-process mock server.

Same ASGITransport pattern as test_replay_trace.py. The server launch is monkeypatched to a
fake process; the mock writes REAL telemetry rows through `telemetry.RowStore`, so the
trace-id join between replay rows and telemetry rows is exercised against the real schema."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from inference_server.research import harness as H
from inference_server.research.venues import extract_payload
from inference_server.telemetry import RequestRecord, RowStore

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "bench"))
import replay_corpus_runpod as rcr  # noqa: E402


def _mock_server(store: RowStore) -> FastAPI:
    """Streams 4 tokens per request and files one telemetry row per request, like the engine."""
    app = FastAPI()
    app.state.seen = []

    @app.post("/v1/completions")
    async def completions(request: Request):
        h, body = request.headers, await request.json()
        trace_id = h.get("x-trace-id", "")
        app.state.seen.append(trace_id)
        rec = RequestRecord(
            trace_id=trace_id, session_id=h.get("x-session-id", ""),
            turn_index=int(h.get("x-turn-index", "0")), arrival_ts=time.time(),
            pending_depth=0, active_size=len(app.state.seen) % 3, max_batch_size=8,
            kv_free_blocks=100, kv_free_frac=0.5, concurrent_sessions=0,
            prompt_tokens=len(body["prompt"]) // 4, replica_age_s=1.0)
        rec.finish("ok", enqueue_ts=0.0, admit_ts=0.01, first_token_ts=0.05, end_ts=0.5,
                   tokens_out=4, cache_hit_tokens=0)
        store.put(rec)

        async def gen():
            for _ in range(4):
                yield f'data: {json.dumps({"choices": [{"text": "x", "finish_reason": None}]})}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"X-Trace-Id": trace_id})

    @app.get("/scheduler/stats")
    async def sched():
        return {"active_high_water": 2, "wave_sizes": {1: 3}}

    @app.get("/cache/stats")
    async def cache():
        return {"hit_rate": 0.0, "lookups": 8}

    return app


class FakeProc:
    """Enough of Popen for start/stop. Terminating closes the store: that is the flush the
    real lifespan shutdown performs."""

    def __init__(self, store: RowStore | None = None):
        self.store, self.terminated = store, False

    def poll(self):
        return 0 if self.terminated else None

    def send_signal(self, sig):
        self.terminated = True
        if self.store is not None:
            self.store.close()

    def wait(self, timeout=None):
        return 0


def _wire(monkeypatch, tmp_path: Path, app: FastAPI | None, proc: FakeProc, ready: bool):
    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "telemetry"))
    # The harness pins the run group once per process, from the env the launcher exports before
    # python starts; patch that value rather than the env var it was read from.
    monkeypatch.setattr(H, "_RUN_GROUP", "grp-test")
    monkeypatch.setenv("REPLAY_CLASSES", "cold_start")
    monkeypatch.setenv("REPLAY_SPLITS", "seen")
    monkeypatch.setenv("REPLAY_RATE_SCALES", "50,100")
    monkeypatch.setenv("REPLAY_WARMUP_N", "2")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_never_in_payload")
    monkeypatch.setattr(rcr, "start_server", lambda env, port, log: proc)
    monkeypatch.setattr(rcr, "wait_ready", lambda *a, **k: ready)
    monkeypatch.setattr(rcr, "hardware_name", lambda: "test-gpu")
    if app is not None:
        monkeypatch.setattr(rcr, "make_client", lambda base_url: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"))
    orig = rcr.run_plan
    monkeypatch.setattr(rcr, "run_plan", lambda *a, **k: orig(*a, **{**k, "settle_s": 0.0}))


# ---------------------------------------------------------------- configuration

def test_engine_env_mirrors_modal_app_and_the_environment_wins(tmp_path):
    env = rcr.engine_env({})
    assert env["BACKEND"] == "custom-cuda" and env["MODEL_NAME"] == "google/gemma-4-E4B-it"
    assert env["MAX_BATCH_SIZE"] == "256" and env["PREFILL_MODE"] == "batched"
    assert Path(env["TELEMETRY_DIR"]).is_dir(), "a fresh telemetry dir is minted when unset"

    env = rcr.engine_env({"MAX_BATCH_SIZE": "32", "TELEMETRY_DIR": str(tmp_path),
                          "HF_TOKEN": "tok", "SCHEDULING_POLICY": "fair"})
    assert env["MAX_BATCH_SIZE"] == "32" and env["TELEMETRY_DIR"] == str(tmp_path)
    assert env["HF_TOKEN"] == "tok" and env["SCHEDULING_POLICY"] == "fair"
    assert "HF_TOKEN" not in rcr.public_env(env), "the secret reaches the server, never stdout"


def test_plan_is_the_cross_product_with_documented_defaults():
    assert rcr.parse_plan({}) == [("steady_interactive", "seen", 1.0),
                                  ("steady_interactive", "seen", 2.0),
                                  ("steady_interactive", "seen", 4.0)]
    plan = rcr.parse_plan({"REPLAY_CLASSES": "cold_start,long_context",
                           "REPLAY_SPLITS": "seen,heldout", "REPLAY_RATE_SCALES": "0.5"})
    assert plan == [("cold_start", "seen", 0.5), ("cold_start", "heldout", 0.5),
                    ("long_context", "seen", 0.5), ("long_context", "heldout", 0.5)]


# ---------------------------------------------------------------- the payload

def test_payload_carries_panels_rows_and_joined_telemetry(monkeypatch, tmp_path, capsys):
    store = RowStore(tmp_path / "telemetry")
    app = _mock_server(store)
    proc = FakeProc(store)
    _wire(monkeypatch, tmp_path, app, proc, ready=True)

    assert rcr.main() == 0
    payload = extract_payload(capsys.readouterr().out)

    assert proc.terminated, "the server is stopped before the telemetry is read"
    assert payload["hardware"] == "test-gpu"
    assert "HF_TOKEN" not in payload["engine_env"]
    assert payload["engine_env"]["BACKEND"] == "custom-cuda"
    assert payload["plan"] == [["cold_start", "seen", 50.0], ["cold_start", "seen", 100.0]]

    assert len(payload["panels"]) == 2
    for d in payload["panels"]:
        v = d["validity"]
        assert v["harness"] == "replay_trace" and v["run_group"] == "grp-test"
        assert v["workload_class"] == "cold_start" and v["harness_config"]["split"] == "seen"
    assert [d["validity"]["harness_config"]["rate_scale"] for d in payload["panels"]] == [50.0, 100.0]

    assert [r["rate_scale"] for r in payload["replays"]] == [50.0, 100.0]
    for rep in payload["replays"]:
        assert rep["class"] == "cold_start" and rep["summary"]["n_ok"] == 8
        assert len(rep["rows"]) == 8 and rep["rows"][0]["trace_id"] == f"{rep['trace_prefix']}-0"
        # the join: one telemetry row per replay row, keyed by the same trace id
        assert sorted(t["trace_id"] for t in rep["telemetry_rows"]) == \
            sorted(r["trace_id"] for r in rep["rows"])
        assert rep["telemetry_rows"][0]["terminal_state"] == "ok"
        assert rep["telemetry_rows"][0]["prefill_s"] == pytest.approx(0.04)
    prefixes = {r["trace_prefix"] for r in payload["replays"]}
    assert len(prefixes) == 2, "two replays of one trace get two nonces"

    # warm-up hit the server (2 requests) but its rows were discarded by the prefix join
    assert len(app.state.seen) == 2 + 8 + 8
    warm = [t for t in app.state.seen if t.startswith("warmup-")]
    assert len(warm) == 2
    joined = {t["trace_id"] for r in payload["replays"] for t in r["telemetry_rows"]}
    assert not any(t in joined for t in warm)


def test_a_server_that_never_becomes_ready_is_an_error_payload_not_a_silent_result(
        monkeypatch, tmp_path, capsys):
    """LOOP.md: no evidence must fail loudly. The venue still gets a parseable block, with the
    log tail that says why, and the exit code says it is not a result."""
    proc = FakeProc()
    _wire(monkeypatch, tmp_path, None, proc, ready=False)
    (tmp_path / "telemetry").mkdir()

    def start(env, port, log):
        Path(log).write_text("\n".join(f"line {i}" for i in range(80)) + "\n")
        return proc
    monkeypatch.setattr(rcr, "start_server", start)

    assert rcr.main() == 1
    out, err = capsys.readouterr()
    payload = extract_payload(out)
    assert payload["panels"] == [] and payload["replays"] == []
    assert "never became ready" in payload["error"]
    assert payload["server_log_tail"] == [f"line {i}" for i in range(30, 80)]
    assert "line 79" in err, "the tail is on stderr too, where the venue's error message reads"
    assert proc.terminated, "an unready server is still a running process; it must be stopped"


def test_telemetry_join_is_by_prefix_and_drops_foreign_rows():
    replays = [{"trace_prefix": "cold_start-seen-aaaaaa", "telemetry_rows": []},
               {"trace_prefix": "cold_start-seen-bbbbbb", "telemetry_rows": []}]
    rows = [{"trace_id": "cold_start-seen-aaaaaa-0"}, {"trace_id": "cold_start-seen-aaaaaa-10"},
            {"trace_id": "cold_start-seen-bbbbbb-0"}, {"trace_id": "warmup-seen-cccccc-0"},
            {"trace_id": "cold_start-seen-aaaaaab-0"}]     # a longer nonce is not a prefix hit
    rcr.attach_telemetry(replays, rows)
    assert [t["trace_id"] for t in replays[0]["telemetry_rows"]] == \
        ["cold_start-seen-aaaaaa-0", "cold_start-seen-aaaaaa-10"]
    assert [t["trace_id"] for t in replays[1]["telemetry_rows"]] == ["cold_start-seen-bbbbbb-0"]


def test_read_telemetry_reads_every_sqlite_file_in_the_dir(tmp_path):
    for run in ("a", "b"):
        s = RowStore(tmp_path, run_id=run)
        rec = RequestRecord(trace_id=f"t-{run}", session_id="s", turn_index=0, arrival_ts=1.0,
                            pending_depth=0, active_size=0, max_batch_size=1, kv_free_blocks=None,
                            kv_free_frac=None, concurrent_sessions=0, prompt_tokens=3,
                            replica_age_s=0.0)
        s.put(rec)
        s.close()
    rows = rcr.read_telemetry(tmp_path)
    assert sorted(r["trace_id"] for r in rows) == ["t-a", "t-b"]
    assert rcr.read_telemetry(tmp_path / "empty-nonexistent") == []
