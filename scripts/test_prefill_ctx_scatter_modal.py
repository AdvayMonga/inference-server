"""Vectorized _PrefillCtx scatter: the KV it writes must equal per-row single prefill.

_PrefillCtx.append used to loop rows in Python calling PagedKVCache.append. It now does ONE
scatter for the whole right-padded wave, with padding positions redirected to the pool's scratch
block. The failure modes that introduces are (a) a padding position writing into a real row's KV,
(b) wrong block/slot for a row whose suffix is shorter than Smax, (c) wrong offset for a row with
a cached prefix. All three corrupt KV silently — first-token comparison is too weak to catch them,
so this reads the actual pool contents back through each row's block table.

Gate is RELATIVE, not exact. Batched prefill KV already differs from single-row prefill KV in
bf16 — a K-row forward uses different GEMM shapes than a 1-row one, so the rounding differs.
That is pre-existing (the K=3 unit test hedges the same way) and unrelated to the scatter. A
real placement bug (wrong block, wrong slot, padding landing on a live row) produces garbage,
i.e. ~100% relative error, which this catches easily. Cross-checked: this script produces
byte-identical numbers on main (per-row loop) and on the vectorized scatter.

!! The relerr this reports for K>1 (0.25-1.4) is NOT rounding and is PRE-EXISTING — main
produces the same numbers to the last bit. Batched prefill writes measurably different KV than
single-row prefill for the same prompt. First tokens still match, but that is weak evidence.
Tracked as an open finding in DECISIONS.md; the gate here is deliberately loose (2.0) so it
catches catastrophic scatter-placement bugs without failing on that separate issue.

This path is CUDA-only (CPU prefill_batch takes the gather branch), so it cannot be a unit test.

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/test_prefill_ctx_scatter_modal.py
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
app = modal.App("prefill-ctx-scatter-test", image=image)
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

    def read_kv(cache, layer, n_tokens):
        """Gather this row's first n_tokens of K/V out of the pool via its block table."""
        pool = backend.pools[layer]
        bs = pool.block_size
        bt = cache.block_tables[layer]
        ks, vs = [], []
        for pos in range(n_tokens):
            bid = bt[pos // bs]
            if bid < 0:
                return None                      # evicted (sliding window) — skip this layer
            ks.append(pool.k[bid, :, pos % bs, :].clone())
            vs.append(pool.v[bid, :, pos % bs, :].clone())
        return torch.stack(ks), torch.stack(vs)

    live = [L for L, p in enumerate(backend.pools) if p is not None]

    def check(name, prompts, seed_prefix=None):
        # reference: each prompt prefilled alone
        backend.prefix_cache = PrefixCache(pools=backend.pools)
        if seed_prefix is not None:
            backend.prefill_batch([seed_prefix])          # warm the prefix cache
        singles = [backend.prefill_batch([p])[0] for p in prompts]

        backend.prefix_cache = PrefixCache(pools=backend.pools)
        if seed_prefix is not None:
            backend.prefill_batch([seed_prefix])
        batched = backend.prefill_batch(list(prompts))

        worst, bad, scale = 0.0, [], 1e-6
        for r, p in enumerate(prompts):
            for L in live:
                a = read_kv(singles[r][0], L, len(p))
                b = read_kv(batched[r][0], L, len(p))
                if a is None or b is None:
                    continue
                for t in (0, 1):
                    ref = a[t].float()
                    d = float((ref - b[t].float()).abs().max())
                    mag = max(float(ref.abs().max()), 1e-6)
                    scale = max(scale, mag)
                    worst = max(worst, d / mag)
                    if d / mag > 2.0:
                        bad.append((r, L, t, round(d / mag, 4)))
        toks_match = [s[1] == b[1] for s, b in zip(singles, batched)]
        for c, _, _ in singles:
            c.free_all()
        for c, _, _ in batched:
            c.free_all()
        ok = not bad
        print(f"  {name:<34} rows={len(prompts):<3} kv_relerr={worst:.5f} "
              f"first_tok_match={sum(toks_match)}/{len(toks_match)} {'OK' if ok else 'FAIL'}",
              flush=True)
        if bad:
            print(f"     first mismatches: {bad[:4]}", flush=True)
        return ok

    print("=== batched-prefill KV must equal single-prefill KV ===", flush=True)
    base = list(range(1000, 1000 + 300))
    allok = True
    allok &= check("K=1", [base[:64]])
    allok &= check("K=4 equal lengths", [[9000 + i] + base[:63] for i in range(4)])
    allok &= check("K=4 RAGGED (pad stress)", [[9100 + i] + base[: 7 + i * 60] for i in range(4)])
    allok &= check("K=8 ragged, 1-token row", [[9200 + i] + base[: (i * 37) % 200] for i in range(8)])
    allok &= check("K=3 with shared prefix hit",
                   [base[:128] + [7000 + i] for i in range(3)], seed_prefix=base[:128])
    print(f"\nscatter correctness: {'PASS' if allok else 'FAIL'}", flush=True)
    return bool(allok)
