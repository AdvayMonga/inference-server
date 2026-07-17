"""Verify + benchmark K=1 CUDA-graph prefill vs eager, A100/E4B.

One model load; toggles backend._prefill_graph_on in-process so both paths run the SAME (unique,
cache-miss) prompts. Checks PARITY (graphed first-token == eager first-token) and latency. Prefix
store disabled so unique bench prompts don't leak and both paths see matched=0.

    venv/bin/modal run --detach scripts/bench_prefill_graph_modal.py
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")

_env = {
    "CUSTOM_BACKEND_PREFILL_GRAPH": "1",      # allow graph capture (we toggle its use in-process)
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
app = modal.App("prefill-graph-ab", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

PROMPT_LENS = [64, 128, 240, 480]   # → buckets 64,128,256,512


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def run():
    import statistics
    import time

    import torch

    from inference_server.backends import create_backend

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    backend.prefix_cache.store = lambda *a, **k: None   # no leak; both paths see matched=0
    print(f"[prefill-graph] {MODEL} on {GPU}  graph_on flag={backend._prefill_graph_on}", flush=True)

    lead = [0]

    def prefill_once(n, use_graph):
        lead[0] += 1
        ids = [40000 + lead[0]] + list(range(100, 100 + n - 1))   # unique lead → cache miss
        backend._prefill_graph_on = use_graph
        cache, first, _ = backend.prefill_batch([ids])[0]
        cache.free_all()
        return first

    def time_path(n, use_graph, iters=20):
        for _ in range(6):        # warmup (first graphed call of a bucket captures)
            prefill_once(n, use_graph)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prefill_once(n, use_graph)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return statistics.median(ts)

    print(f"\n  {'prompt':>7} {'eager ms':>9} {'graph ms':>9} {'speedup':>8} {'parity':>7}", flush=True)
    for n in PROMPT_LENS:
        # parity: same prompt through both paths (store disabled → both matched=0)
        lead[0] += 1
        ids = [50000 + lead[0]] + list(range(100, 100 + n - 1))
        backend._prefill_graph_on = False
        fe = backend.prefill_batch([ids])[0][1]
        backend._prefill_graph_on = True
        fg = backend.prefill_batch([ids])[0][1]
        parity = "OK" if fe == fg else f"X {fe}!={fg}"

        eager = time_path(n, False)
        graph = time_path(n, True)
        print(f"  {n:>7} {eager:>9.1f} {graph:>9.1f} {eager / graph:>7.2f}x {parity:>7}", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
