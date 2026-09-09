"""How much of batched prefill is _PrefillCtx Python bookkeeping vs actual model work?

_PrefillCtx.block_table_tensor() rebuilds a [K, maxb] int32 tensor from Python lists on EVERY
layer call (42 layers per prefill, K host->device copies each), and append() loops rows in
Python. Decode had exactly this shape of problem and de-Pythoning it (BatchedDecodeState) was
recorded as ~95% of decode latency.

Decides the next move: if bookkeeping is a large share, de-Python _PrefillCtx (helps EVERY
prefill including prefix-cache hits). If it is small, spend the effort on a 2D (K, S) prefill
CUDA-graph ladder instead.

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/probes/probe_prefill_ctx_overhead_modal.py
"""

import os

import modal

MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "MODEL_NAME": MODEL, "MAX_BATCH_SIZE": "32", "CONTEXT_WINDOW": "8192",
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "KV_CACHE_NUM_BLOCKS": "16384", "CUSTOM_BACKEND_COMPILE": "0",
    })
    .add_local_python_source("inference_server")
)
app = modal.App("prefill-ctx-overhead", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

CASES = [(1, 240), (4, 240), (8, 240), (8, 480)]


@app.function(gpu="A100-80GB", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def run():
    import time

    import torch

    from inference_server.backends import create_backend
    from inference_server.backends.custom_torch_backend import _PrefillCtx
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PagedKVCache

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    model, dev = backend.model, backend.device
    n_layers = len(model.model.layers)
    live_pools = [L for L, p in enumerate(backend.pools) if p is not None]
    print(f"layers={n_layers}  pools with KV={len(live_pools)}\n", flush=True)

    def setup(K, S):
        caches = [PagedKVCache(pools=backend.pools) for _ in range(K)]
        ids = torch.randint(10, 200000, (K, S), device=dev)
        pos = torch.arange(S, device=dev).unsqueeze(0).expand(K, S).contiguous()
        ctx = _PrefillCtx(caches, backend.pools, [0] * K, [S] * K, dev)
        idx = torch.full((K,), S - 1, dtype=torch.long, device=dev)
        return caches, ids, pos, ctx, idx

    def timeit(fn, iters=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    print(f"{'K':>4} {'S':>5} {'forward ms':>11} {'bt_tensor ms':>13} {'append ms':>10} "
          f"{'bookkeeping':>12}", flush=True)
    for K, S in CASES:
        caches, ids, pos, ctx, idx = setup(K, S)
        fwd = timeit(lambda: model(ids, position_ids=pos, paged_ctx=ctx, logits_index=idx))
        for c in caches:
            c.free_all()

        # cost of ONE forward's worth of block_table_tensor calls (one per layer)
        caches, ids, pos, ctx, idx = setup(K, S)
        model(ids, position_ids=pos, paged_ctx=ctx, logits_index=idx)   # populate the caches

        def bt_all():
            for L in live_pools:
                ctx.block_table_tensor(L)

        bt = timeit(bt_all, iters=20) * (n_layers / len(live_pools))

        # head_dim differs per pool (sliding 256 vs full 512) — size each from the pool itself
        kv_by_pool = {L: torch.randn(K, backend.pools[L].k.shape[1], S,
                                     backend.pools[L].k.shape[-1],
                                     device=dev, dtype=torch.bfloat16) for L in live_pools}

        # append grows the caches, so rebuild between reps: measure a single sweep, few reps
        ap_times = []
        for _ in range(3):
            cs, i2, p2, c2, x2 = setup(K, S)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for L in live_pools:
                c2.append(L, kv_by_pool[L], kv_by_pool[L])
            torch.cuda.synchronize()
            ap_times.append((time.perf_counter() - t0) * 1000 * (n_layers / len(live_pools)))
            for c in cs:
                c.free_all()
        ap = sorted(ap_times)[1]
        for c in caches:
            c.free_all()

        print(f"{K:>4} {S:>5} {fwd:>11.2f} {bt:>13.2f} {ap:>10.2f} "
              f"{(bt + ap) / fwd * 100:>11.1f}%", flush=True)
