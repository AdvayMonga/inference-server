"""WHERE does batched prefill KV diverge from single-row prefill KV?

test_prefill_ctx_scatter_modal showed relerr 0.25-1.4 between prefill_batch([p]) and
prefill_batch([p0,p1,...]) for the same p — far beyond bf16 rounding, and identical on main,
so it predates the vectorized scatter. PREFILL_MODE=batched is what the serving config runs,
so if this is a real bug every serving number is suspect.

Localizes it: per (row, layer) report WHICH token positions differ. The shape of the answer
names the cause —
  * only positions >= a row's real length  -> harmless padding tail, test artifact
  * only the LAST real position            -> off-by-one in the suffix/append length
  * positions beyond the SHORTEST row       -> cross-row contamination via padding
  * all positions, growing with depth       -> genuine numerical divergence from batching

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/probes/diag_batched_prefill_kv_modal.py
"""

import os

import modal

MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E2B-it")
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
app = modal.App("batched-prefill-kv-diag", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=os.environ.get("BENCH_GPU", "A100-80GB"),
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def run():
    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PrefixCache

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    live = [L for L, p in enumerate(backend.pools) if p is not None]
    print(f"pools with KV: {live[:6]}... ({len(live)})  "
          f"window={backend.pools[live[0]].window}", flush=True)

    def read_kv(cache, layer, n):
        pool = backend.pools[layer]
        bs = pool.block_size
        bt = cache.block_tables[layer]
        ks, vs = [], []
        for pos in range(n):
            bid = bt[pos // bs]
            if bid < 0:
                return None
            ks.append(pool.k[bid, :, pos % bs, :].clone())
            vs.append(pool.v[bid, :, pos % bs, :].clone())
        return torch.stack(ks), torch.stack(vs)

    base = list(range(1000, 1000 + 300))
    prompts = [[9100 + i] + base[: 7 + i * 60] for i in range(4)]   # ragged: 8, 68, 128, 188
    lens = [len(p) for p in prompts]
    print(f"prompt lengths: {lens}  (Smax={max(lens)})\n", flush=True)

    backend.prefix_cache = PrefixCache(pools=backend.pools)
    singles = [backend.prefill_batch([p])[0] for p in prompts]
    backend.prefix_cache = PrefixCache(pools=backend.pools)
    batched = backend.prefill_batch(list(prompts))

    print("per (row, layer): first differing token position, and how many positions differ",
          flush=True)
    print(f"{'row':>4} {'len':>5} {'lyr':>6} {'first_diff_pos':>15} {'n_diff':>7} "
          f"{'relerr':>9}", flush=True)
    for r, p in enumerate(prompts):
        for L in live:
            A = read_kv(singles[r][0], L, len(p))
            B = read_kv(batched[r][0], L, len(p))
            if A is None or B is None:
                continue
            for t, tag in ((0, "K"), (1, "V")):
                a, b = A[t].float(), B[t].float()
                per_pos = (a - b).abs().amax(dim=(1, 2))                 # [n_tokens]
                mag = max(float(a.abs().max()), 1e-6)
                diff = (per_pos / mag) > 5e-2
                n = int(diff.sum())
                if n:
                    first = int(diff.nonzero()[0])
                    print(f"{r:>4} {len(p):>5} {L:>4}{tag:>2} {first:>15} {n:>7} "
                          f"{float(per_pos.max()) / mag:>9.4f}", flush=True)
        print("", flush=True)

    for c, _, _ in singles:
        c.free_all()
    for c, _, _ in batched:
        c.free_all()
