"""Tier 3: can ANY tile/warp config beat the untiled kernel at head_dim 512?

Full-attention layers (D=512, no sliding window) do most of the attention work on a long prompt,
and tiling currently loses 0.53x there to register spill. Before writing a new kernel, check
whether it is merely a launch-configuration problem — the accumulator is [BLOCK_M, D] spread
across the block's threads, so BLOCK_M and num_warps trade off against the register file
together, and Triton's default heuristic was never told that.

Falsification: if no config beats untiled at D=512, the lever is not launch config and needs a
real kernel change (chunked head dim).

    scripts/run_instrument.sh scripts/sweep_tiled_launch_modal.py
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
app = modal.App("tiled-launch-sweep", image=image)

BS, NUM_KV, GROUP = 16, 2, 4
CONFIGS = [(8, 4), (8, 8), (16, 4), (16, 8), (32, 4), (32, 8), (64, 8)]


@app.function(gpu=os.environ.get("BENCH_GPU", "A100-80GB"), timeout=2700)
def run():
    import time

    import torch

    from inference_server.models import paged_attention_kernel as K

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    Hq = NUM_KV * GROUP
    out_rows = []

    for D in (512, 256):
        nb = 2048
        kp = torch.randn(nb, NUM_KV, BS, D, device=dev, dtype=torch.bfloat16)
        vp = torch.randn(nb, NUM_KV, BS, D, device=dev, dtype=torch.bfloat16)
        for S in (512, 824):
            q = torch.randn(1, S, Hq, D, device=dev, dtype=torch.bfloat16)
            bt = torch.randint(0, nb, (1, 512), device=dev, dtype=torch.int32)
            pl = torch.tensor([0], device=dev, dtype=torch.int32)
            sl = torch.tensor([S], device=dev, dtype=torch.int32)

            K._TILED_PREFILL = False
            ref = K.paged_prefill_attention(q, kp, vp, bt, pl, sl, 1.0, 1 << 30).float()

            def timeit(fn, n=20):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n * 1000

            base_ms = timeit(lambda: K.paged_prefill_attention(q, kp, vp, bt, pl, sl, 1.0, 1 << 30))
            print(f"\n=== D={D} S={S}: untiled {base_ms:.3f} ms ===", flush=True)

            for bm, warps in CONFIGS:
                K._TILED_PREFILL = True
                K.LAUNCH_BY_HEAD_DIM[D] = (bm, warps)
                try:
                    got = K.paged_prefill_attention(q, kp, vp, bt, pl, sl, 1.0, 1 << 30).float()
                    rel = float((ref - got).abs().max()) / max(float(ref.abs().max()), 1e-6)
                    ms = timeit(lambda: K.paged_prefill_attention(q, kp, vp, bt, pl, sl,
                                                                 1.0, 1 << 30))
                    ok = rel < 2e-2
                    print(f"  BLOCK_M={bm:<3} warps={warps}  {ms:7.3f} ms  "
                          f"{base_ms / ms:5.2f}x  rel={rel:.1e} {'ok' if ok else 'WRONG'}",
                          flush=True)
                    out_rows.append((D, S, bm, warps, base_ms, ms, ok))
                except Exception as e:
                    print(f"  BLOCK_M={bm:<3} warps={warps}  FAILED: {str(e)[:80]}", flush=True)
                K._TILED_PREFILL = False

    best512 = [r for r in out_rows if r[0] == 512 and r[6] and r[4] / r[5] > 1.0]
    print(f"\nD=512 configs that beat untiled: {len(best512)}")
    for r in sorted(best512, key=lambda x: -x[4] / x[5])[:3]:
        print(f"  S={r[1]} BLOCK_M={r[2]} warps={r[3]} -> {r[4] / r[5]:.2f}x")
    if not best512:
        print("  none => launch config is NOT the problem; needs a chunked-head-dim kernel")
    return out_rows
