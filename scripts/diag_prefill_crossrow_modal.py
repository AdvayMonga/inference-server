"""Is batched prefill leaking between rows, or is it just rounding amplified through depth?

diag_batched_prefill_kv showed batched-vs-single KV differing only in deep layers (7-14) at
1-7 isolated positions. Two readings:
  (A) BENIGN: batching changes GEMM shapes -> sub-threshold rounding at layer 0 -> amplified
      through RMSNorm/GeGLU/softmax over 15 layers until isolated positions blow up.
  (B) BUG: padding or block-table entries from neighbouring rows contaminate a row's KV.

They are distinguished by ONE question: does row r's KV depend on what the OTHER rows contain?
Run the same wave twice, holding the probe row identical and changing only its neighbours,
keeping K and Smax fixed so the GEMM shapes are byte-identical between the two runs.
  identical KV  -> (A), no contamination
  different KV  -> (B), a real bug in the path the serving config uses

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/diag_prefill_crossrow_modal.py
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
app = modal.App("prefill-crossrow-diag", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

PROBE_ROW = 2


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

    def read_all(cache, n):
        out = {}
        for L in live:
            pool = backend.pools[L]
            bs = pool.block_size
            bt = cache.block_tables[L]
            ks, vs = [], []
            ok = True
            for pos in range(n):
                bid = bt[pos // bs]
                if bid < 0:
                    ok = False
                    break
                ks.append(pool.k[bid, :, pos % bs, :].clone())
                vs.append(pool.v[bid, :, pos % bs, :].clone())
            if ok:
                out[L] = (torch.stack(ks), torch.stack(vs))
        return out

    base = list(range(1000, 1000 + 300))
    probe = [9999] + base[:127]                       # 128 tokens, held fixed
    nb_a = [[9100 + i] + base[: 7 + i * 60] for i in range(4)]
    nb_b = [[4200 + i] + base[50: 57 + i * 60] for i in range(4)]   # same lengths, other content

    def wave(neigh):
        rows = list(neigh)
        rows[PROBE_ROW] = probe                       # same slot, same K, same Smax
        backend.prefix_cache = PrefixCache(pools=backend.pools)
        res = backend.prefill_batch(rows)
        kv = read_all(res[PROBE_ROW][0], len(probe))
        tok = res[PROBE_ROW][1]
        for c, _, _ in res:
            c.free_all()
        return kv, tok

    kv_a, tok_a = wave(nb_a)
    kv_b, tok_b = wave(nb_b)
    print(f"lengths A={[len(p) for p in nb_a]}  B={[len(p) for p in nb_b]}", flush=True)
    print(f"probe row={PROBE_ROW} len={len(probe)}  first_token A={tok_a} B={tok_b}\n", flush=True)

    worst, ndiff = 0.0, 0
    for L in live:
        if L not in kv_a or L not in kv_b:
            continue
        for t in (0, 1):
            a, b = kv_a[L][t].float(), kv_b[L][t].float()
            d = float((a - b).abs().max())
            worst = max(worst, d)
            ndiff += int(d > 0)
    print(f"probe-row KV across two different neighbour sets: max abs diff = {worst:.6f} "
          f"({ndiff} tensors differ at all)", flush=True)
    print(f"\nVERDICT: {'BENIGN — no cross-row dependence; the batched-vs-single gap is rounding amplified through depth' if worst == 0.0 else 'BUG — a row KV depends on its neighbours content'}",
          flush=True)
    return worst == 0.0
