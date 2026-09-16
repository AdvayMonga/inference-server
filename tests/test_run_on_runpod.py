"""The RunPod launcher's local half: what it writes home, where the HF token comes from, and
that a dry run spends nothing and needs no key."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "tools"))
import run_on_runpod as rr  # noqa: E402

PAYLOAD = {
    "panels": [],
    "engine_env": {"BACKEND": "custom-cuda", "MODEL_NAME": "google/gemma-4-E4B-it"},
    "hardware": "NVIDIA A100 80GB PCIe",
    "plan": [["steady_interactive", "seen", 1.0], ["steady_interactive", "seen", 2.5]],
    "replays": [
        {"class": "steady_interactive", "split": "seen", "rate_scale": 1.0, "trace_prefix": "p1",
         "rows": [{"index": 0, "trace_id": "p1-0", "ttft_ms": 12.5, "error": None}],
         "telemetry_rows": [{"trace_id": "p1-0", "prefill_s": 0.01, "terminal_state": "ok"}]},
        {"class": "steady_interactive", "split": "seen", "rate_scale": 2.5, "trace_prefix": "p2",
         "rows": [{"index": 0, "trace_id": "p2-0", "ttft_ms": 30.0, "error": "drain_timeout"}],
         "telemetry_rows": []},
    ],
}


def test_replays_are_written_as_csv_under_the_run_group(tmp_path):
    written = rr.write_replays(PAYLOAD, "grp-7", tmp_path)
    names = sorted(p.name for p in written)
    assert names == ["engine_env.json",
                     "steady_interactive-seen-x1.rows.csv",
                     "steady_interactive-seen-x1.telemetry.csv",
                     "steady_interactive-seen-x2.5.rows.csv",
                     "steady_interactive-seen-x2.5.telemetry.csv"]
    assert all(p.parent == tmp_path / "grp-7" for p in written)

    with open(tmp_path / "grp-7" / "steady_interactive-seen-x1.rows.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"index": "0", "trace_id": "p1-0", "ttft_ms": "12.5", "error": ""}]
    with open(tmp_path / "grp-7" / "steady_interactive-seen-x1.telemetry.csv", newline="") as f:
        assert list(csv.DictReader(f))[0]["prefill_s"] == "0.01"
    # an empty replay still gets its file, so a missing file means a missing run, not no rows
    assert (tmp_path / "grp-7" / "steady_interactive-seen-x2.5.telemetry.csv").read_text() == ""

    meta = json.loads((tmp_path / "grp-7" / "engine_env.json").read_text())
    assert meta["engine_env"]["MODEL_NAME"] == "google/gemma-4-E4B-it"
    assert meta["hardware"] == "NVIDIA A100 80GB PCIe" and len(meta["plan"]) == 2


def test_hf_token_prefers_env_then_the_cache_file_then_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert rr.hf_token() == (None, "none")

    f = tmp_path / ".cache" / "huggingface" / "token"
    f.parent.mkdir(parents=True)
    f.write_text("hf_fromfile\n")
    assert rr.hf_token() == ("hf_fromfile", str(f))

    monkeypatch.setenv("HF_TOKEN", "hf_fromenv")
    assert rr.hf_token() == ("hf_fromenv", "env")


def test_every_engine_knob_the_instrument_advertises_is_forwarded():
    """Two hand-maintained lists drifted: the launcher forwarded neither SCHEDULING_POLICY nor
    MAX_QUEUE_SIZE, so a knob set in the local shell was silently ignored on the pod."""
    import replay_corpus_runpod as rcr
    src = (REPO / "scripts" / "tools" / "run_on_runpod.py").read_text()
    assert "*ENGINE_DEFAULTS, *ENGINE_PASSTHROUGH" in src, "forward the instrument's own tables"
    for knob in ("SCHEDULING_POLICY", "MAX_QUEUE_SIZE", "KV_CACHE_BLOCK_SIZE", "MAX_QUEUE_WAIT_S",
                 "PREFILL_CHUNK_SIZE", "WAVE_WINDOW_MULT", "CONTEXT_WINDOW",
                 "CUSTOM_BACKEND_PREFILL_GRAPH"):
        assert knob in (*rcr.ENGINE_DEFAULTS, *rcr.ENGINE_PASSTHROUGH), knob


def test_dry_run_needs_no_api_key_and_never_prints_the_token(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "RUNPOD_API_KEY"}
    f = tmp_path / ".cache" / "huggingface" / "token"
    f.parent.mkdir(parents=True)
    f.write_text("hf_supersecret")
    env.update({"HOME": str(tmp_path), "PYTHONPATH": str(REPO / "src"),
                "REPLAY_CLASSES": "cold_start", "TELEMETRY_DIR": "/tmp/t"})
    # Empty, not absent: config.py runs load_dotenv() at import and a developer's .env
    # may carry HF_TOKEN. load_dotenv never overrides a variable that is already set, so
    # an empty one keeps the repo's .env out of this test and hf_token() falls to the file.
    env["HF_TOKEN"] = ""
    env["SCHEDULING_POLICY"] = "fair"
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "tools" / "run_on_runpod.py"),
                        "scripts/bench/replay_corpus_runpod.py", "--dry-run",
                        "--timeout", "5400"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "[dry-run] would rent NVIDIA A100 80GB PCIe and run scripts/bench/replay_corpus_runpod.py" in r.stdout
    assert f"[hf] token from {f}" in r.stdout
    assert "hf_supersecret" not in r.stdout + r.stderr
    assert "[dry-run] timeouts: run 5400s, ssh-ready 300s" in r.stdout
    env_line = next(ln for ln in r.stdout.splitlines() if ln.startswith("[dry-run] env:"))
    for k in ("HF_TOKEN", "REPLAY_CLASSES", "TELEMETRY_DIR", "RESEARCH_RUN_GROUP",
              "RESEARCH_ENGINE_SHA", "SCHEDULING_POLICY"):
        assert k in env_line, k
