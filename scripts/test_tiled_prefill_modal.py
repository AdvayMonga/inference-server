"""Tiled prefill attention: correctness against the untiled kernel, then speed.

Correctness first and hard — this kernel computes the same quantity by a different reduction
order, so the bar is tight-but-not-bitwise, and it must hold across the cases that break tiling:
tails that do not fill a tile, sliding windows, prefix-cache hits, and mixed head dims.

    scripts/run_instrument.sh scripts/test_tiled_prefill_modal.py
"""

from __future__ import annotations

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"RESEARCH_ENGINE_SHA": os.environ.get("RESEARCH_ENGINE_SHA", ""),
          "RESEARCH_RUN_GROUP": os.environ.get("RESEARCH_RUN_GROUP", "")})
    .add_local_python_source("inference_server")
)
app = modal.App("tiled-prefill-test", image=image)

BS = 16
NUM_KV = 2
GROUP = 4          # Hq = 8


@app.function(gpu=os.environ.get("BENCH_GPU", "A100-80GB"), timeout=1800)
def run():
    import time

    import torch

    from inference_server.models import paged_attention_kernel as K

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    Hq = NUM_KV * GROUP
    ok = True

    def make(N, S, D, nblocks=2048):
        kp = torch.randn(nblocks, NUM_KV, BS, D, device=dev, dtype=torch.bfloat16)
        vp = torch.randn(nblocks, NUM_KV, BS, D, device=dev, dtype=torch.bfloat16)
        q = torch.randn(N, S, Hq, D, device=dev, dtype=torch.bfloat16)
        maxb = 512
        bt = torch.randint(0, nblocks, (N, maxb), device=dev, dtype=torch.int32)
        return q, kp, vp, bt

    def call(q, kp, vp, bt, pl, sl, window, tiled):
        K._TILED_PREFILL = tiled
        return K.paged_prefill_attention(q, kp, vp, bt, pl, sl, scale=1.0, window=window)

    print("=== correctness: tiled vs untiled ===", flush=True)
    cases = [
        ("aligned tile",      1, 64,  256, [0],       [64],      1 << 30),
        ("ragged tail",       1, 70,  256, [0],       [70],      1 << 30),
        ("single query",      1, 1,   256, [0],       [1],       1 << 30),
        ("prefix hit",        1, 48,  256, [128],     [48],      1 << 30),
        ("sliding window",    1, 512, 256, [0],       [512],     512),
        ("window + prefix",   1, 96,  256, [512],     [96],      512),
        ("head_dim 512",      1, 128, 512, [0],       [128],     1 << 30),
        ("batch of 3 ragged", 3, 96,  256, [0, 32, 0], [96, 64, 5], 1 << 30),
        ("padding only tail", 1, 80,  256, [0],       [33],      1 << 30),
    ]
    for name, N, S, D, prefix, suffix, window in cases:
        q, kp, vp, bt = make(N, S, D)
        pl = torch.tensor(prefix, device=dev, dtype=torch.int32)
        sl = torch.tensor(suffix, device=dev, dtype=torch.int32)
        ref = call(q, kp, vp, bt, pl, sl, window, tiled=False).float()
        got = call(q, kp, vp, bt, pl, sl, window, tiled=True).float()
        # compare only real query rows; padding rows are undefined by contract
        diffs = []
        for r in range(N):
            n = suffix[r]
            a, b = ref[r, :n], got[r, :n]
            mag = max(float(a.abs().max()), 1e-6)
            diffs.append(float((a - b).abs().max()) / mag)
        worst = max(diffs)
        good = bool(worst < 2e-2 and torch.isfinite(got).all())
        ok &= good
        print(f"  {name:<20} D={D} rel={worst:.2e}  {'OK' if good else 'FAIL'}", flush=True)

    print(f"\ncorrectness: {'PASS' if ok else 'FAIL'}\n", flush=True)
    if not ok:
        return {"ok": False}

    print("=== speed (prefill attention only) ===", flush=True)
    print(f"  {'tokens':>7} {'D':>4} {'untiled ms':>11} {'tiled ms':>9} {'speedup':>8}", flush=True)
    rows = []
    for S in (240, 512, 824):
        for D in (256, 512):
            q, kp, vp, bt = make(1, S, D)
            pl = torch.tensor([0], device=dev, dtype=torch.int32)
            sl = torch.tensor([S], device=dev, dtype=torch.int32)

            def timeit(tiled):
                for _ in range(5):
                    call(q, kp, vp, bt, pl, sl, 1 << 30, tiled)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    call(q, kp, vp, bt, pl, sl, 1 << 30, tiled)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / 20 * 1000

            u, t = timeit(False), timeit(True)
            rows.append((S, D, u, t))
            print(f"  {S:>7} {D:>4} {u:>11.3f} {t:>9.3f} {u / t:>7.2f}x", flush=True)

    return {"ok": bool(ok), "rows": rows}
