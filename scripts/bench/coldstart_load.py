#!/usr/bin/env python
"""One cold model load, measured as one panel (LOOP.md step 0, resources slice).

A cold start is a PROCESS, not an iteration: the second load in a process is warm in every way
that matters (imports resolved, allocator warm, weights in the OS page cache), so one run = one
process = one panel. Replicates come from running this file again, never from a loop inside it.

    RESEARCH_RUN_GROUP=grp-x RESEARCH_ARM=baseline PYTHONPATH=src python scripts/bench/coldstart_load.py

`workload_regime` is `synthetic`: nothing is served, so the cache regimes (`cache_hit_heavy` /
`cache_miss_heavy` / `mixed`) name a property this run does not have. Claiming one of them would
be worse than saying nothing — it would invite comparison with a serving panel.

The panel's `wall_s` is `GemmaForCausalLM.from_hf` end to end. The stage split underneath it
lives in `harness_config["stage_ms"]`, not in new panel fields: extending the panel is a
PANEL_VERSION bump that invalidates comparison to every existing panel, and a stage breakdown is
provenance for this harness rather than a vital sign of the engine.

The stages are timed by wrapping the three calls `from_hf` makes, from outside the engine. An
engine edit would move the sha this experiment vouches for, which is the one thing an instrument
must not do.
"""

from __future__ import annotations

import contextlib
import os
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch  # noqa: E402

from inference_server.research import harness as H  # noqa: E402

MODEL = os.environ.get("COLDSTART_MODEL", "google/gemma-4-E2B-it")
DTYPE = torch.bfloat16

_MISSING = object()


@contextlib.contextmanager
def _stage_timers(out: dict[str, float]):
    """Time the three calls from_hf makes, by patching their owners for the length of the load."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from inference_server.models.gemma4 import GemmaModel

    targets = [
        (AutoConfig, "from_pretrained", "config_read_ms"),
        (AutoModelForCausalLM, "from_pretrained", "hf_read_ms"),
        (GemmaModel, "load_hf_weights", "copy_ms"),
    ]
    saved = []
    for owner, name, key in targets:
        orig = getattr(owner, name)
        saved.append((owner, name, owner.__dict__.get(name, _MISSING)))

        def timed(*a, _orig=orig, _key=key, **kw):
            t0 = time.perf_counter()
            try:
                return _orig(*a, **kw)
            finally:
                out[_key] = round((time.perf_counter() - t0) * 1000, 1)

        setattr(owner, name, timed)
    try:
        yield
    finally:
        for owner, name, prev in saved:
            if prev is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, prev)


def _peak_rss_gb() -> float:
    """ru_maxrss is BYTES on Darwin and kilobytes on Linux; getting this wrong is a 1024x error."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)


def main() -> int:
    arm = os.environ.get("RESEARCH_ARM", "")
    if not arm:
        print("RESEARCH_ARM must name the arm this process measures (session.arm_label parses "
              "it out of the panel's notes)", file=sys.stderr)
        return 2

    from inference_server.models.gemma4 import GemmaForCausalLM

    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    with _stage_timers(stages):
        model = GemmaForCausalLM.from_hf(MODEL, dtype=DTYPE)
    wall_s = time.perf_counter() - t0

    measured = sum(stages.get(k, 0.0) for k in ("config_read_ms", "hf_read_ms", "copy_ms"))
    stages["construct_ms"] = round(wall_s * 1000 - measured, 1)

    cfg = {
        "model": MODEL,
        "gpu": None,                 # from_hf builds on the host; no accelerator is touched
        "device": "cpu",
        "dtype": "bfloat16",
        "platform": f"{platform.system().lower()}-{platform.machine()}",
        "torch": torch.__version__,
        "hf_cache_warm": True,       # weights already in ~/.cache/huggingface; no download timed
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "stage_ms": stages,
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    panel = H.panel_from_stats(
        H.build_validity(
            "coldstart_load", cfg, n_samples=1, workload_regime="synthetic",
            notes=f"arm={arm}; one cold from_hf per process; stage split in "
                  f"harness_config.stage_ms"),
        wall_s=round(wall_s, 3),
        peak_host_rss_gb=_peak_rss_gb(),
    )

    print(f"\ncold load  {MODEL}  arm={arm}")
    print(f"  total from_hf      {wall_s * 1000:9.0f} ms")
    for key in ("config_read_ms", "construct_ms", "hf_read_ms", "copy_ms"):
        print(f"  {key:<18} {stages.get(key, float('nan')):9.0f} ms")
    print(f"  peak host RSS      {panel.peak_host_rss_gb:9.2f} GB")
    print(f"  parameters         {cfg['model_params']:9d}")

    H.emit(panel, label="coldstart_load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
