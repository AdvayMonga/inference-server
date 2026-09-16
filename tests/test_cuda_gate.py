"""The CUDA gate has to be dry-runnable where there is no CUDA and no modal.

The gate's checks only mean something on a GPU, so what a laptop CAN prove is the plumbing:
the instrument imports without modal, reports rather than crashes when torch sees no device,
emits a payload the venue parses, and the modal scripts still call the same shared checks
instead of carrying a copy that drifts.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from inference_server.research.venues import extract_payload

REPO = Path(__file__).resolve().parents[1]
GPU_TESTS = REPO / "scripts" / "gpu_tests"
MODAL_GATE_SCRIPTS = ("test_paged_kernel_modal.py", "test_paged_prefill_kernel_modal.py")


def _imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _no_cuda() -> bool:
    torch = pytest.importorskip("torch")
    return not torch.cuda.is_available()


def test_gate_and_checks_never_import_modal():
    for f in ("cuda_gate.py", "checks.py"):
        assert "modal" not in _imported_modules(ast.parse((GPU_TESTS / f).read_text())), f


def test_gate_reports_every_check_skipped_without_cuda(monkeypatch):
    if not _no_cuda():
        pytest.skip("this test is about the no-CUDA path")
    monkeypatch.setitem(sys.modules, "modal", None)      # `import modal` now raises ImportError
    spec = importlib.util.spec_from_file_location("cuda_gate", GPU_TESTS / "cuda_gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    results = gate.run_checks(cuda=False)
    assert [r["name"] for r in results] == list(gate.checks.CHECKS)
    assert all(r["passed"] is False and "no CUDA" in r["detail"] for r in results)


def test_gate_emits_a_failing_verdict_the_venue_can_parse():
    """Run it as the pod would. No device here, so the verdict is a fail with an exit of 1 —
    a gate that cannot run has not passed — but the payload must still come home intact."""
    if not _no_cuda():
        pytest.skip("this test is about the no-CUDA path")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, str(GPU_TESTS / "cuda_gate.py")],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 1, r.stderr[-500:]
    assert "no CUDA" in r.stdout

    payload = extract_payload(r.stdout)
    assert payload["panels"] == [], "a gate measures nothing, so it emits no panel"
    gate = payload["gate"]
    assert gate["passed"] is False and gate["gpu"] is None
    assert gate["torch"] and "triton" in gate
    assert {c["name"] for c in gate["checks"]} >= {"paged_decode_parity", "paged_prefill_parity"}


def test_modal_gate_scripts_run_the_shared_checks():
    """AST-level, like test_kernel_source.py: the engine lane has no modal to import."""
    for f in MODAL_GATE_SCRIPTS:
        tree = ast.parse((GPU_TESTS / f).read_text())
        assert "checks" in _imported_modules(tree), f"{f} does not import checks"
        mounted = [
            a.value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add_local_python_source"
            for a in n.args if isinstance(a, ast.Constant)
        ]
        assert "checks" in mounted, f"{f} must ship checks.py to the Modal container"
        defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "_reference" not in defs, f"{f} carries its own reference — it will drift"



def _launcher(monkeypatch, payload):
    """Load run_on_runpod.py with the venue replaced, so nothing is rented."""
    spec = importlib.util.spec_from_file_location(
        "run_on_runpod", REPO / "scripts" / "tools" / "run_on_runpod.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen = {}

    def fake_run_instrument(script, env, *, repo, spec, **kw):
        seen["spec"], seen["kw"] = spec, kw
        return payload

    monkeypatch.setattr(mod, "run_instrument", fake_run_instrument)
    return mod, seen


def test_launcher_exits_with_the_gate_verdict_and_names_the_pod(monkeypatch, capsys):
    mod, seen = _launcher(monkeypatch, {"gate": {"passed": False, "checks": []}})
    monkeypatch.setattr(sys, "argv", ["x", "scripts/gpu_tests/cuda_gate.py", "--name", "ci-cuda-gate"])
    assert mod.main() == 1
    assert seen["spec"].name == "ci-cuda-gate"
    # the venue's wall budget and ssh wait are the launcher's to set, not defaults buried below
    assert seen["kw"]["run_timeout_s"] == 3600 and seen["kw"]["ready_timeout_s"] == 300.0
    assert '"passed": false' in capsys.readouterr().out


def test_launcher_surfaces_an_instrument_error_and_writes_no_panels(monkeypatch, capsys, tmp_path):
    mod, _ = _launcher(monkeypatch, {"error": "server not ready", "log_tail": ["OOM at layer 3"],
                                     "panels": [{"would": "be invalid"}]})
    monkeypatch.setattr(mod, "REPO", tmp_path)          # runs/ would land here; it must not
    monkeypatch.setattr(sys, "argv", ["x", "scripts/bench/replay.py"])
    assert mod.main() == 1
    err = capsys.readouterr().err
    assert "server not ready" in err and "OOM at layer 3" in err
    # The panel this payload carries does not parse, so nothing is written. A partial run whose
    # panels DO parse keeps them (test_venues: a partial run keeps the panels it did measure).
    assert not (tmp_path / "runs").exists()
