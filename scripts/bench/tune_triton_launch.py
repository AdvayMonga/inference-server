#!/usr/bin/env python
"""Tier 3: sweep the paged-attention kernels' Triton launch configs per bucket, write the table.

The engine reads the table at model load via CUSTOM_BACKEND_LAUNCH_TABLE (models/launch_table.py)
— never at first call, so tuning adds nothing to cold start. A key with no entry launches
exactly as before. Why a sweep at all: knowledge/kb-20260905-a474d802.json.

    RUNPOD_API_KEY=... MODEL_NAME=google/gemma-4-E4B-it MAX_BATCH_SIZE=256 \\
        scripts/tools/run_on_runpod.py scripts/bench/tune_triton_launch.py --gpu 'NVIDIA A100 80GB PCIe'
    PYTHONPATH=src python scripts/bench/tune_triton_launch.py          # directly, on a CUDA box

Env: MODEL_NAME (attention shapes from its config.json, no weights), MAX_BATCH_SIZE +
CUSTOM_BACKEND_COMPILE (the engine's own decode-bucket ladder), CUSTOM_BACKEND_BLOCK_SIZE,
TUNE_DECODE_KV_LENS (comma list), TUNE_OUT (table path).

Every candidate's output is checked against the default launch's; the kernels are not
batch-invariant (kb-20260901-011), so the bound is the parity checks' tolerance, and bit-exactness
is recorded per row rather than required.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "gpu_tests"))

import checks  # noqa: E402
from inference_server.models import launch_table  # noqa: E402

WARPS = (1, 2, 4, 8)
STAGES = (1, 2, 3, 4)
SPLITS = (1, 2, 4, 8)
BLOCK_MS = (16, 32, 64)
DECODE_TOL = checks.TOL          # paged_decode_parity's bound
PREFILL_TOL = 1e-2               # paged_prefill_parity's bound
MIN_GAIN = 1.05                  # a config must beat the default launch by 5% to earn an entry
FULL = 1 << 30                   # the kernels' "no window" sentinel
TIMING_DIR = REPO / "knowledge" / "timing"
DEFAULT_MODEL = "google/gemma-4-E4B-it"


@dataclass(frozen=True)
class Shape:
    """One attention layer type, as the kernels see it."""
    head_dim: int
    q_heads: int
    kv_heads: int
    window: int


@dataclass
class Case:
    """One table key and the configs swept for it."""
    kernel: str                  # paged_decode | paged_prefill | paged_prefill_tiled
    shape: int                   # decode: rows N (the graph bucket); prefill: suffix tokens S
    attn: Shape
    block_size: int
    candidates: list[dict[str, int]]

    @property
    def tol(self) -> float:
        return DECODE_TOL if self.kernel == "paged_decode" else PREFILL_TOL


# ------------------------------------------------------------------ pure (CPU-testable)

def attention_shapes(text_cfg: Any) -> list[Shape]:
    """Distinct (head_dim, heads, window) the model's layers launch with."""
    out: list[Shape] = []
    for t in dict.fromkeys(text_cfg.layer_types):
        sliding = t == "sliding_attention"
        s = Shape(text_cfg.head_dim if sliding else text_cfg.global_head_dim,
                  text_cfg.num_attention_heads, text_cfg.num_key_value_heads,
                  text_cfg.sliding_window if sliding else FULL)
        if s not in out:
            out.append(s)
    return out


def decode_space() -> list[dict[str, int]]:
    return [{"splits": s, "num_warps": w, "num_stages": st}
            for s, w, st in itertools.product(SPLITS, WARPS, STAGES)]


def prefill_space() -> list[dict[str, int]]:
    return [{"num_warps": w, "num_stages": st} for w, st in itertools.product(WARPS, STAGES)]


def tiled_space() -> list[dict[str, int]]:
    return [{"block_m": m, "num_warps": w, "num_stages": st}
            for m, w, st in itertools.product(BLOCK_MS, WARPS, STAGES)]


def plan(shapes: list[Shape], decode_buckets: tuple[int, ...], prefill_buckets: tuple[int, ...],
         block_size: int, tiled_min_tokens: int, tiled_head_dims: set[int]) -> list[Case]:
    """Every key to sweep. Tiled prefill only where the engine could route to it."""
    cases = [Case("paged_decode", n, a, block_size, decode_space())
             for n in decode_buckets for a in shapes]
    cases += [Case("paged_prefill", s, a, block_size, prefill_space())
              for s in prefill_buckets for a in shapes]
    cases += [Case("paged_prefill_tiled", s, a, block_size, tiled_space())
              for s in prefill_buckets for a in shapes
              if s >= tiled_min_tokens and a.head_dim in tiled_head_dims]
    return cases


def pick(case: Case, default_ms: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The table entry for this case: fastest valid config, if it beats the default by MIN_GAIN."""
    valid = [r for r in rows if r.get("error") is None and r["max_abs_diff"] <= case.tol]
    if not valid:
        return None
    best = min(valid, key=lambda r: r["ms"])
    if default_ms / best["ms"] < MIN_GAIN:
        return None
    return {"kernel": case.kernel, "shape": case.shape, "head_dim": case.attn.head_dim,
            "block_size": case.block_size, "config": dict(best["config"]),
            "ms": best["ms"], "default_ms": default_ms, "max_abs_diff": best["max_abs_diff"],
            "exact": best["exact"]}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "unknown"


def table_filename(gpu: str, sha: str) -> str:
    return f"triton-launch-{slug(gpu)}-{sha}.json"


def build_table(entries: list[dict[str, Any]], *, model: str, gpu: str, sha: str,
                run_group: str) -> dict[str, Any]:
    """The document launch_table.load reads. Every entry carries the GPU it was tuned on."""
    return {"kind": launch_table.KIND, "model": model, "hardware": gpu, "engine_sha": sha,
            "fitted_from": run_group,
            "entries": [{**e, "gpu": gpu} for e in entries]}


def write_table(doc: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def env_ints(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(x) for x in os.environ.get(name, default).split(",") if x.strip())


# ------------------------------------------------------------------ GPU

def _use(case: Case, cfg: dict[str, int] | None) -> None:
    """Make `cfg` the only table entry (None = the default launch)."""
    launch_table.clear()
    if cfg is not None:
        key = launch_table.entry_key(case.kernel, case.shape, case.attn.head_dim, case.block_size)
        launch_table._TABLE[key] = dict(cfg)


def _inputs(case: Case, kv_len: int) -> Callable[[], Any]:
    """A zero-arg call of the real kernel wrapper at this case's shape."""
    import torch

    from inference_server.models import paged_attention_kernel as K

    a, bs, dev = case.attn, case.block_size, "cuda"
    if case.kernel == "paged_decode":
        n, nb = case.shape, (kv_len + bs - 1) // bs
        kp = torch.randn(n * nb, a.kv_heads, bs, a.head_dim, device=dev, dtype=torch.bfloat16)
        vp = torch.randn_like(kp)
        bt = torch.arange(n * nb, device=dev, dtype=torch.int32).reshape(n, nb)
        sl = torch.full((n,), kv_len, device=dev, dtype=torch.int32)
        q = torch.randn(n, a.q_heads, a.head_dim, device=dev, dtype=torch.bfloat16)
        return lambda: K.paged_decode_attention(q, kp, vp, bt, sl, 1.0, a.window)
    s = case.shape
    nb = (s + bs - 1) // bs
    kp = torch.randn(nb, a.kv_heads, bs, a.head_dim, device=dev, dtype=torch.bfloat16)
    vp = torch.randn_like(kp)
    bt = torch.arange(nb, device=dev, dtype=torch.int32).reshape(1, nb)
    pl = torch.zeros(1, device=dev, dtype=torch.int32)
    sl = torch.full((1,), s, device=dev, dtype=torch.int32)
    q = torch.randn(1, s, a.q_heads, a.head_dim, device=dev, dtype=torch.bfloat16)
    return lambda: K.paged_prefill_attention(q, kp, vp, bt, pl, sl, 1.0, a.window)


def _time(case: Case, fn: Callable[[], Any]) -> float:
    """Median ms. Decode is timed inside a CUDA graph, as the engine replays it; eager timing
    would charge split-K's extra combine launch with CPU launch overhead a graph does not pay."""
    from triton.testing import do_bench, do_bench_cudagraph

    if case.kernel == "paged_decode":
        return float(do_bench_cudagraph(fn, rep=50, return_mode="median"))
    return float(do_bench(fn, warmup=10, rep=50, return_mode="median"))


def run_case(case: Case, kv_lens: tuple[int, ...]) -> tuple[float, list[dict[str, Any]]]:
    """(default ms, one row per candidate). Decode sums ms over kv_lens; prefill has one input."""
    import torch

    from inference_server.models import paged_attention_kernel as K

    K._TILED_PREFILL = case.kernel == "paged_prefill_tiled"
    fns = [_inputs(case, L) for L in (kv_lens if case.kernel == "paged_decode" else (0,))]
    _use(case, None)
    refs = [fn().float() for fn in fns]
    default_ms = sum(_time(case, fn) for fn in fns)
    rows = []
    for cfg in case.candidates:
        _use(case, cfg)
        row: dict[str, Any] = {"kernel": case.kernel, "shape": case.shape,
                               "head_dim": case.attn.head_dim, "config": cfg, "ms": None,
                               "max_abs_diff": None, "exact": None, "error": None}
        try:
            outs = [fn().float() for fn in fns]
            row["max_abs_diff"] = max(float((o - r).abs().max()) for o, r in zip(outs, refs))
            row["exact"] = all(torch.equal(o, r) for o, r in zip(outs, refs))
            row["ms"] = sum(_time(case, fn) for fn in fns)
        except Exception as e:                   # noqa: BLE001 — e.g. out of shared memory
            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rows.append(row)
    _use(case, None)
    K._TILED_PREFILL = False
    return default_ms, rows


def _stamp_device_state() -> None:
    """RESEARCH_DEVICE_STATE from the venue, or queried here when run directly."""
    if os.environ.get("RESEARCH_DEVICE_STATE"):
        return
    import subprocess

    import torch

    from inference_server.research.determinism import query_device

    state = query_device(lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
    state.cuda_version = torch.version.cuda
    os.environ["RESEARCH_DEVICE_STATE"] = json.dumps(state.to_dict())


def main() -> int:
    from inference_server.research import harness as H
    from inference_server.research.schemas import git_sha
    from inference_server.research.venues import emit_payload

    import torch

    if not torch.cuda.is_available():
        print("[tune] no CUDA device; nothing to tune")
        print(emit_payload({"panels": [], "error": "no CUDA device"}))
        return 1

    import triton
    from transformers import AutoConfig

    from inference_server.backends.custom_torch_backend import _PREFILL_BUCKETS, _decode_buckets
    from inference_server.config import settings
    from inference_server.models import paged_attention_kernel as K

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    _stamp_device_state()
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL)
    gpu = torch.cuda.get_device_name(0)
    block_size = int(os.environ.get("CUSTOM_BACKEND_BLOCK_SIZE", "16"))
    compile_on = os.environ.get("CUSTOM_BACKEND_COMPILE", "0") == "1"
    decode_buckets = _decode_buckets(settings.max_batch_size, coarse=compile_on)
    kv_lens = env_ints("TUNE_DECODE_KV_LENS", "256,2048")
    shapes = attention_shapes(AutoConfig.from_pretrained(model).text_config)
    cases = plan(shapes, decode_buckets, _PREFILL_BUCKETS, block_size,
                 K.TILED_MIN_TOKENS, set(K.LAUNCH_BY_HEAD_DIM))
    print(f"[tune] {model} on {gpu}: {len(cases)} keys, "
          f"{sum(len(c.candidates) for c in cases)} configs")

    sweep, entries = [], []
    for c in cases:
        default_ms, rows = run_case(c, kv_lens)
        sweep += [{**r, "default_ms": default_ms} for r in rows]
        e = pick(c, default_ms, rows)
        if e:
            entries.append(e)
        best = f"{e['config']} {default_ms / e['ms']:.2f}x" if e else "default kept"
        print(f"[tune] {c.kernel:<20} shape={c.shape:<5} D={c.attn.head_dim:<4} "
              f"default {default_ms:.4f} ms -> {best}", flush=True)

    sha = git_sha()
    doc = build_table(entries, model=model, gpu=gpu, sha=sha, run_group=H.run_group())
    out = Path(os.environ.get("TUNE_OUT") or TIMING_DIR / table_filename(gpu, sha))
    print(f"[tune] {len(entries)} of {len(cases)} keys beat the default launch; "
          f"table -> {write_table(doc, out)}")

    cfg = {"model": model, "gpu": gpu, "block_size": block_size, "compile": compile_on,
           "max_batch_size": settings.max_batch_size, "decode_buckets": list(decode_buckets),
           "prefill_buckets": list(_PREFILL_BUCKETS), "decode_kv_lens": list(kv_lens),
           "shapes": [asdict(s) for s in shapes], "warps": list(WARPS), "stages": list(STAGES),
           "splits": list(SPLITS), "block_ms": list(BLOCK_MS), "min_gain": MIN_GAIN,
           "decode_tol": DECODE_TOL, "prefill_tol": PREFILL_TOL,
           "timer": "do_bench_cudagraph (decode) / do_bench (prefill), median",
           "torch": torch.__version__, "triton": triton.__version__}
    panel = H.panel_from_stats(H.build_validity(
        "tune_triton_launch", cfg, n_samples=sum(r["ms"] is not None for r in sweep),
        workload_regime="synthetic",
        notes=f"{len(entries)}/{len(cases)} keys tuned; kernel-isolated, not end to end"))
    H.emit(panel, label="tune_triton_launch")
    print(emit_payload({"panels": [panel.to_dict()], "launch_table": doc, "sweep": sweep}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
