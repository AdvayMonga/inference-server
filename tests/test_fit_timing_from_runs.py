"""fit_timing_from_runs.py on synthetic runs: recover known coefficients from telemetry-shaped
rows, and report rank correlation 1.0 when the hardware panels ARE the simulator's numbers."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from inference_server.research import harness as H
from inference_server.research.corpus import load_trace
from inference_server.research.simulator import SimConfig, TimingModel, simulate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "tools"))
import fit_timing_from_runs as ft  # noqa: E402

TRUTH = TimingModel(prefill=(0.012, 0.0011, 0.0), decode=(0.008, 0.00025, 0.0))
ENV = {"MODEL_NAME": "google/gemma-4-E4B-it", "MAX_BATCH_SIZE": "32", "PREFILL_MODE": "batched",
       "KV_CACHE_NUM_BLOCKS": "4096", "KV_CACHE_BLOCK_SIZE": "16"}


def _telemetry_row(i: int, ok: bool = True) -> dict:
    """A RequestRecord-shaped row whose spans obey TRUTH, with a cache hit on every third."""
    prompt, hit, active, steps = 40 + 17 * i, (24 if i % 3 == 0 else 0), i % 6, 5 + i % 9
    prefill = TRUTH.prefill_s(prompt - hit, 0)
    decode = steps * TRUTH.decode_step_s(active + 1, 0)
    return {"trace_id": f"p-{i}", "session_id": "s", "turn_index": 0, "arrival_ts": float(i),
            "pending_depth": 0, "active_size": active, "max_batch_size": 32,
            "kv_free_blocks": "", "kv_free_frac": "", "concurrent_sessions": 0,
            "prompt_tokens": prompt, "replica_age_s": 1.0, "config_id": "",
            "queue_wait_s": 0.001, "prefill_s": prefill, "decode_s": decode,
            "decode_steps": steps, "cache_hit_tokens": hit, "preempted": 0,
            "ttft_s": prefill + 0.001, "tpot_s": decode / steps, "total_s": 1.0,
            "tokens_out": steps + 1, "terminal_state": "ok" if ok else "expired"}


def _group(runs: Path, grp: str, rows: list[dict], meta: dict) -> None:
    d = runs / grp
    d.mkdir(parents=True)
    with open(d / "steady_interactive-seen-x1.telemetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (d / "engine_env.json").write_text(json.dumps(meta))


def test_fit_rows_apply_the_documented_mapping():
    rows = ft.fit_rows([_telemetry_row(3), _telemetry_row(4, ok=False)])
    assert len(rows) == 1, "only terminal_state == ok rows are evidence"
    r = rows[0]
    assert r["prompt_tokens"] == (40 + 51) - 24                 # uncached tokens
    assert r["batch_size"] == 3 % 6 + 1                         # active_size + self
    assert r["prefill_s"] == pytest.approx(TRUTH.prefill_s(67, 0))
    assert r["decode_step_s"] == pytest.approx(TRUTH.decode_step_s(4, 0))


def test_fit_recovers_the_coefficients_and_writes_the_model(tmp_path, capsys):
    runs = tmp_path / "runs"
    _group(runs, "grp-fit", [_telemetry_row(i) for i in range(40)],
           {"engine_env": ENV, "hardware": "NVIDIA A100 80GB PCIe"})
    out = tmp_path / "knowledge" / "timing" / "t.json"

    assert ft.main(["grp-fit", "--runs-dir", str(runs), "--out", str(out), "--sha", "abc123"]) == 0
    fit = TimingModel.from_json(out)
    assert fit.prefill == pytest.approx(TRUTH.prefill, abs=1e-9)
    assert fit.decode == pytest.approx(TRUTH.decode, abs=1e-9)
    assert (fit.model, fit.hardware, fit.engine_sha, fit.fitted_from) == \
        ("google/gemma-4-E4B-it", "NVIDIA A100 80GB PCIe", "abc123", "grp-fit")
    out_text = capsys.readouterr().out
    assert "prefill_s     = 0.012000" in out_text and "--validate" in out_text


def test_default_output_path_is_keyed_on_model_hardware_and_sha():
    assert ft.slug("google/gemma-4-E4B-it") == "google-gemma-4-e4b-it"
    assert ft.slug("NVIDIA A100 80GB PCIe") == "nvidia-a100-80gb-pcie"
    assert ft.TIMING_DIR == REPO / "knowledge" / "timing"


def test_fit_with_no_ok_rows_is_an_error(tmp_path):
    runs = tmp_path / "runs"
    _group(runs, "grp-empty", [_telemetry_row(i, ok=False) for i in range(5)], {"engine_env": ENV})
    assert ft.main(["grp-empty", "--runs-dir", str(runs), "--out", str(tmp_path / "t.json")]) == 1
    assert not (tmp_path / "t.json").exists()


def _hardware_panel(runs: Path, grp: str, cls_name: str, split: str, scale: float,
                    summary: dict) -> None:
    """A replay_trace-shaped panel whose numbers are exactly the simulator's."""
    v = H.build_validity("replay_trace",
                         {"workload_class": cls_name, "split": split, "rate_scale": scale},
                         n_samples=summary["n_ok"], workload_regime="mixed")
    v.run_group = grp
    panel = H.panel_from_stats(v, ttft_p50=summary["ttft_p50"], ttft_p95=summary["ttft_p95"],
                               tpot_p50=summary["tpot_p50"], tpot_p95=summary["tpot_p95"],
                               wall_s=summary["wall_s"])
    panel.to_json(runs / f"{v.run_id}.json")


def test_validate_reports_rho_one_when_hardware_equals_the_simulator(tmp_path, capsys):
    runs = tmp_path / "runs"
    (runs / "grp-val").mkdir(parents=True)
    (runs / "grp-val" / "engine_env.json").write_text(json.dumps({"engine_env": ENV}))
    timing_path = tmp_path / "t.json"
    TRUTH.to_json(timing_path)

    manifest, trace = load_trace("steady_interactive", "seen")
    cls = manifest.classes["steady_interactive"]
    # Below saturation, so TPOT differs too: at x1+ the batch pins at 32 in every config, TPOT
    # ties everywhere, and rank_correlation reports 0.0 for "no ordering to agree with".
    sims = {}
    for scale in (0.25, 0.5, 1.0):
        sims[scale] = simulate(trace, ft.sim_config(ENV, scale), TRUTH).summary(cls)
        _hardware_panel(runs, "grp-val", "steady_interactive", "seen", scale, sims[scale])
    for metric in ("ttft_p95", "tpot_p50"):
        assert len({s[metric] for s in sims.values()}) == 3, f"{metric} must differ per config"

    assert ft.main(["grp-val", "--runs-dir", str(runs), "--validate",
                    "--timing", str(timing_path)]) == 0
    out = capsys.readouterr().out
    assert "spearman ttft_p95: 1.0" in out and "spearman tpot_p50: 1.0" in out
    assert "steady_interactive/seen x0.25" in out

    panels, meta, _ = ft.load_group(runs, "grp-val")
    v = ft.validate(panels, TRUTH, meta["engine_env"])
    assert [d["rate_scale"] for d in v["pairs"]] == [0.25, 0.5, 1.0]
    for d in v["pairs"]:
        assert d["hw_ttft_p95"] == d["sim_ttft_p95"] and d["hw_tpot_p50"] == d["sim_tpot_p50"]


def test_sim_config_is_read_from_the_engine_env_the_server_ran_with():
    cfg = ft.sim_config({**ENV, "SCHEDULING_POLICY": "fair", "MAX_QUEUE_WAIT_S": "5"}, 2.0)
    assert cfg == SimConfig(max_batch_size=32, kv_blocks=4096, block_size=16, policy="fair",
                            prefill_mode="batched", rate_scale=2.0, max_queue_wait_s=5.0)
    assert ft.sim_config({}, 1.0).prefill_mode == "monolithic"


def test_validate_refuses_fewer_than_two_panels(tmp_path, capsys):
    runs = tmp_path / "runs"
    (runs / "grp-one").mkdir(parents=True)
    TRUTH.to_json(tmp_path / "t.json")
    assert ft.main(["grp-one", "--runs-dir", str(runs), "--validate",
                    "--timing", str(tmp_path / "t.json")]) == 1
    assert "needs >= 2" in capsys.readouterr().err
