"""Isolate the cost of running lm_head over EVERY prefill position vs only the needed one.

Prefill produces [K, Smax, hidden] but needs exactly one logit row per sequence. Projecting
every position is a [K*Smax, 2560] @ [2560, 262144] GEMM plus a softcap over the whole thing —
for a wave of 8 prompts of 240 tokens, ~2.6 TFLOP and ~1GB materialized to read 8 rows.

Times the SAME forward with logits_index set vs None, in one process, so nothing else varies.

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/bench_lm_head_slice_modal.py
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
app = modal.App("lm-head-slice-ab", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

CASES = [(1, 240), (4, 240), (8, 240), (8, 480), (16, 240)]


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

    def build_ctx(K, S):
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

    print(f"{'K':>4} {'S':>5} {'sliced ms':>10} {'full ms':>9} {'saved ms':>9} {'speedup':>8} "
          f"{'full logits':>12}", flush=True)
    for K, S in CASES:
        caches, ids, pos, ctx, idx = build_ctx(K, S)
        sliced = timeit(lambda: model(ids, position_ids=pos, paged_ctx=ctx, logits_index=idx))
        full = timeit(lambda: model(ids, position_ids=pos, paged_ctx=ctx))
        for c in caches:
            c.free_all()
        mb = K * S * 262144 * 2 / 1e6
        print(f"{K:>4} {S:>5} {sliced:>10.2f} {full:>9.2f} {full - sliced:>9.2f} "
              f"{full / sliced:>7.2f}x {mb:>10.0f}MB", flush=True)

    # correctness: sliced row must equal the same row of the full logits
    caches, ids, pos, ctx, idx = build_ctx(4, 240)
    a = model(ids, position_ids=pos, paged_ctx=ctx, logits_index=idx).float()
    b = model(ids, position_ids=pos, paged_ctx=ctx).float()
    for c in caches:
        c.free_all()
    same = all(int(a[r, 0].argmax()) == int(b[r, 239].argmax()) for r in range(4))
    d = max(float((a[r, 0] - b[r, 239]).abs().max()) for r in range(4))
    print(f"\nargmax agrees on all rows: {same}   maxdiff={d:.3e}", flush=True)
    return same
