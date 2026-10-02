"""lab.profile writes a complete bundle whose trace carries the engine's phase ranges."""

from __future__ import annotations

import json

import pytest

from lab import bundle, gpu, profile
from tests.stub_backend import StubBackend


@pytest.mark.asyncio
async def test_bundle_is_complete(tmp_path):
    prompts = profile.synthetic_prompts(4, 8)
    out = await profile.run(StubBackend(), prompts, max_tokens=3, out=tmp_path, warmup=1,
                            max_batch_size=4)
    names = {p.name for p in out.iterdir()}
    assert set(bundle.FILES) <= names
    assert ("gpu.csv" in names) == gpu.available()

    meta = json.loads((out / "meta.json").read_text())
    assert meta["requests"] == 4 and meta["max_tokens"] == 3 and meta["failed"] == 0
    assert meta["workload_hash"] == bundle.workload_hash(prompts, 3)
    assert meta["device"] == "cpu" and meta["window"]["wall_s"] > 0 and meta["torch"]

    trace = json.loads((out / "trace.json").read_text())
    names_in_trace = {e.get("name", "") for e in trace["traceEvents"]}
    assert any(n.startswith("decode step=") for n in names_in_trace)
    assert any(n.startswith("prefill step=") for n in names_in_trace)

    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    windows = [e["state"] for e in events if e["kind"] == "profile_window"]
    assert windows == ["begin", "end"]
    finished = [e for e in events if e["kind"] == "finish"]
    assert len(finished) == 5          # 1 warmup + 4 profiled
    stats = json.loads((out / "stats.json").read_text())
    assert stats["total_completed"] == 5 and stats["timeline"]["events_dropped"] == 0


def test_synthetic_prompts_are_deterministic():
    assert profile.synthetic_prompts(2, 5) == profile.synthetic_prompts(2, 5)
    assert profile.synthetic_prompts(2, 5) != profile.synthetic_prompts(2, 5, seed=1)
