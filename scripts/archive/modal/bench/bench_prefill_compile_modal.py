"""Focused A/B: K=1 prefill latency, compiled vs eager, A100/E4B.

Verifies the prefill-compile change (custom_torch_backend: K=1 bucketed dynamic=False) WITHOUT
the fragile full serving sweep. One model load; toggles backend._compile_prefill in-process so
both paths run the SAME inputs. Prefix-cache store is disabled so unique bench prompts don't leak
blocks (every call is a full uncached prefill — the 73ms case that sets unloaded TTFT).

    venv/bin/modal run --detach scripts/bench/bench_prefill_compile_modal.py
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")

_env = {
    "CUSTOM_BACKEND_COMPILE": "1",            # so _prefill_fwd is compiled (we toggle its use)
    "MODEL_NAME": MODEL, "BENCH_GPU": GPU, "BENCH_MODEL": MODEL, "MAX_BATCH_SIZE": "256",
    "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
    "KV_CACHE_NUM_BLOCKS": "16384",
}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env(_env)
    .add_local_python_source("inference_server")
)
app = modal.App("prefill-ab", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

PROMPT_LENS = [64, 128, 240, 480]   # suffix lengths → buckets 64,128,256,512


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1200)
def run():
    import statistics
    import time

    import torch

    from inference_server.backends import create_backend

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    backend.prefix_cache.store = lambda *a, **k: None   # no leak: every prefill is a full miss
    print(f"[prefill-ab] {MODEL} on {GPU}  compile_prefill={backend._compile_prefill}", flush=True)

    lead = [0]

    def one(n):
        lead[0] += 1
        ids = [40000 + lead[0]] + list(range(100, 100 + n - 1))   # unique lead → cache miss
        for c, _, _ in backend.prefill_batch([ids]):              # K=1
            c.free_all()

    def time_path(n, compiled, iters=20):
        backend._compile_prefill = compiled
        for _ in range(6):        # warmup (triggers this bucket's compile when compiled=True)
            one(n)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one(n)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return statistics.median(ts)

    print(f"\n  {'prompt':>7} {'eager ms':>9} {'compiled ms':>12} {'speedup':>8}", flush=True)
    for n in PROMPT_LENS:
        eager = time_path(n, False)
        comp = time_path(n, True)
        print(f"  {n:>7} {eager:>9.1f} {comp:>12.1f} {eager / comp:>7.2f}x", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
