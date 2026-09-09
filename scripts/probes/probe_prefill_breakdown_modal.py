"""Tier 3: where does the 4.4x prefill gap actually go?

Tier 1 established that at the p95 prompt (824 tok) attention is 1.5% of the FLOPs and the dense
GEMM dominates, yet measured prefill is 4.4x the arithmetic floor. This buckets one graphed
prefill's self-CUDA time by op category to say which part of that gap is reclaimable.

Falsification: if GEMM is already >70% of kernel time we are near the practical ceiling and the
gap is not overhead — the hypothesis dies here.

    scripts/run_instrument.sh scripts/probes/probe_prefill_breakdown_modal.py
"""

from __future__ import annotations

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
LENGTHS = [int(x) for x in os.environ.get("PROBE_LENGTHS", "240,512,824").split(",")]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "MODEL_NAME": MODEL, "MAX_BATCH_SIZE": "32", "CONTEXT_WINDOW": "8192",
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "KV_CACHE_NUM_BLOCKS": "16384", "CUSTOM_BACKEND_COMPILE": "0",
        "CUSTOM_BACKEND_PREFILL_GRAPH": os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "1"),
        "RESEARCH_ENGINE_SHA": os.environ.get("RESEARCH_ENGINE_SHA", ""),
        "RESEARCH_ENGINE_DIRTY": os.environ.get("RESEARCH_ENGINE_DIRTY", ""),
        "RESEARCH_RUN_GROUP": os.environ.get("RESEARCH_RUN_GROUP", ""),
    })
    .add_local_python_source("inference_server")
)
app = modal.App("prefill-breakdown", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# Leaf-kernel name -> category. Order matters: first match wins.
CATEGORIES = [
    ("gemm",       ("gemm", "cutlass", "ampere", "splitk", "s16816", "cublas")),
    ("attention",  ("paged_prefill", "paged_decode", "attention")),
    ("kv_scatter", ("index_put", "scatter", "copy_kernel")),
    ("memcpy",     ("memcpy", "memset")),
    ("norm_rope",  ("norm", "rope", "rotary", "pow", "rsqrt", "mul", "add", "gelu", "silu",
                    "tanh", "elementwise", "vectorized")),
    ("gather",     ("index_select", "gather", "embedding")),
    ("softmax",    ("softmax", "exp")),
]


def categorise(name: str) -> str:
    low = name.lower()
    for cat, needles in CATEGORIES:
        if any(n in low for n in needles):
            return cat
    return "other"


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def run():
    import time

    import torch
    from torch.profiler import ProfilerActivity, profile

    from inference_server.backends import create_backend
    from inference_server.config import settings

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    print(f"[probe] {settings.model_name} prefill_graph={backend._prefill_graph_on}\n",
          flush=True)

    lead = [0]

    def one(n: int):
        lead[0] += 1
        ids = [40000 + lead[0]] + [1000 + (i % 50000) for i in range(n - 1)]
        cache, first, _ = backend.prefill_batch([ids])[0]
        cache.free_all()
        return first

    results = {}
    for n in LENGTHS:
        for _ in range(4):                      # warm: capture the bucket graph
            one(n)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            one(n)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000

        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(3):
                one(n)
            torch.cuda.synchronize()

        buckets: dict[str, float] = {}
        total_us = 0.0
        for ev in prof.key_averages():
            self_us = getattr(ev, "self_device_time_total", 0) or 0
            if self_us <= 0 or ev.key.startswith("aten::"):
                continue                        # skip parents; count leaf kernels only
            buckets[categorise(ev.key)] = buckets.get(categorise(ev.key), 0.0) + self_us
            total_us += self_us

        results[n] = {"wall_ms": ms, "kernel_ms": total_us / 3 / 1000,
                      "buckets": {k: v / 3 / 1000 for k, v in
                                  sorted(buckets.items(), key=lambda kv: -kv[1])}}

        print(f"=== {n} tokens: {ms:.1f} ms wall, {total_us / 3 / 1000:.1f} ms in kernels",
              flush=True)
        for cat, cms in results[n]["buckets"].items():
            print(f"      {cat:<12} {cms:7.2f} ms  {cms / (total_us / 3 / 1000):6.1%}",
                  flush=True)
        gap = ms - total_us / 3 / 1000
        print(f"      {'[not in kernels]':<12} {gap:7.2f} ms  {gap / ms:6.1%}  "
              f"<- launch/host gap\n", flush=True)

    # --- the falsification test ---
    n = LENGTHS[-1]
    r = results[n]
    gemm_share = r["buckets"].get("gemm", 0.0) / max(r["kernel_ms"], 1e-9)
    print(f"VERDICT at {n} tokens: GEMM is {gemm_share:.1%} of kernel time")
    if gemm_share > 0.70:
        print("  => hypothesis FALSIFIED: already GEMM-dominated, the gap is not overhead")
    else:
        print(f"  => hypothesis HOLDS: {1 - gemm_share:.1%} of kernel time is not GEMM, "
              f"plus a {(r['wall_ms'] - r['kernel_ms']) / r['wall_ms']:.1%} host/launch gap")
    return results
