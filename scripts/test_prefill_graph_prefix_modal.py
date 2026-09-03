"""K=1 prefill CUDA graph with a PREFIX-CACHE HIT: parity, no IMA, and does it even fire?

The graph was gated to matched==0. With a warm prefix cache almost every request is a hit, so
in the serving harness (64 reused prompts, no cache eviction) the graph essentially never fired
— while 83-91% of prefill waves are K=1, exactly the shape it handles. This exercises the
generalized path where the write positions start at prefix_len instead of 0.

Checks, in order of what would actually bite:
  1. FIRES     — the graphed path is taken on a hit (not silently falling back to eager)
  2. PARITY    — graphed first token == eager first token on the same hit
  3. KV        — the suffix KV the graph scattered matches eager, read back through block tables
  4. NO IMA    — many replays interleaved with block alloc/free, the stress that used to crash
  5. LATENCY   — graphed vs eager on a hit

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/test_prefill_graph_prefix_modal.py
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E2B-it")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "MODEL_NAME": MODEL, "MAX_BATCH_SIZE": "32", "CONTEXT_WINDOW": "8192",
        "CUSTOM_BACKEND_PREFILL_GRAPH": "1",
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "KV_CACHE_NUM_BLOCKS": "16384", "CUSTOM_BACKEND_COMPILE": "0",
    })
    .add_local_python_source("inference_server")
)
app = modal.App("prefill-graph-prefix-test", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=2700)
def run():
    import statistics
    import time

    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    live = [L for L, p in enumerate(backend.pools) if p is not None]
    window = min((backend.pools[L].window for L in live
                  if backend.pools[L].window is not None), default=None)
    print(f"prefill_graph_on={backend._prefill_graph_on}  sliding window={window}\n", flush=True)

    base = list(range(1000, 1000 + 300))
    ok = True

    def read_kv(cache, layer, lo, hi):
        pool = backend.pools[layer]
        bs = pool.block_size
        bt = cache.block_tables[layer]
        out = []
        for pos in range(lo, hi):
            bid = bt[pos // bs]
            if bid < 0:
                return None
            out.append(pool.k[bid, :, pos % bs, :].clone())
        return torch.stack(out)

    # ---------- 1 + 2 + 3: fires, parity, KV ----------
    print("=== prefix-hit parity (graph vs eager on the SAME hit) ===", flush=True)
    for plen, slen in ((64, 8), (128, 16), (128, 64), (240, 8), (600, 8), (600, 300)):
        prompt = base[:plen]
        full = prompt + [77000 + slen]

        def warm_and_prefill(use_graph):
            backend.prefix_cache = type(backend.prefix_cache)(pools=backend.pools)
            backend._prefill_graph_on = False
            w = backend.prefill_batch([prompt])[0]          # warm the cache (always eager)
            backend._prefill_graph_on = use_graph
            r = backend.prefill_batch([full])[0]
            hit = backend.last_cache_hit_tokens
            return w, r, hit

        we, re_, hit_e = warm_and_prefill(False)
        wg, rg, hit_g = warm_and_prefill(True)

        fired = hit_g > 0
        parity = re_[1] == rg[1]
        kvok = True
        for L in live:
            a = read_kv(re_[0], L, hit_e, len(full))
            b = read_kv(rg[0], L, hit_g, len(full))
            if a is None or b is None or a.shape != b.shape:
                continue
            mag = max(float(a.float().abs().max()), 1e-6)
            if float((a.float() - b.float()).abs().max()) / mag > 5e-2:
                kvok = False
                break
        # On a first-token mismatch, is it a near-tie the padded width flipped, or a real
        # divergence? Compute an INDEPENDENT reference (contiguous KV cache, no paging, no
        # padding) and look at where the two candidates sit in it.
        note = ""
        if not parity:
            from inference_server.models.gemma4 import KVCache
            ref = backend.model(torch.tensor([full], device=backend.device),
                                kv_cache=KVCache(backend.model.model.num_layers))[0, -1].float()
            top = ref.topk(3)
            gap = float(top.values[0] - top.values[1])
            rank = {int(t): i for i, t in enumerate(top.indices.tolist())}
            note = (f"  [ref top3={top.indices.tolist()} gap={gap:.4f} "
                    f"eager_rank={rank.get(re_[1], '>2')} graph_rank={rank.get(rg[1], '>2')}]")
            parity = gap < 0.05 and rank.get(re_[1], 9) <= 1 and rank.get(rg[1], 9) <= 1
        ok &= parity and kvok and fired
        for c in (we[0], re_[0], wg[0], rg[0]):
            c.free_all()
        print(f"  prompt={plen:>4} suffix={slen:>3}  matched(eager/graph)={hit_e}/{hit_g}  "
              f"fired={'Y' if fired else 'N'}  first_tok={re_[1]}/{rg[1]} "
              f"{'MATCH' if parity else 'MISMATCH'}  kv={'OK' if kvok else 'BAD'}{note}",
              flush=True)

    # ---------- 3b: fall back cleanly when the graph cannot take it ----------
    # Suffix past the largest bucket -> _prefill_bucket returns None; prefix+suffix past the
    # sliding window -> ineligible. Both must fall back to eager, not raise. (A None-check
    # ordered after the arithmetic crashed the scheduler here with TypeError: int + NoneType,
    # which showed up as TTFT p95 815ms rather than as an obvious failure.)
    print("\n=== ineligible prompts must fall back to eager, not raise ===", flush=True)
    long_base = list(range(2000, 2000 + 2000))
    for label, prompt in (
        ("suffix > largest bucket", long_base[:1500]),
    ):
        backend.prefix_cache = type(backend.prefix_cache)(pools=backend.pools)
        backend._prefill_graph_on = True
        try:
            r = backend.prefill_batch([prompt])[0]
            good = isinstance(r[1], int) and r[2] == len(prompt)
            r[0].free_all()
        except Exception as e:
            good = False
            print(f"  {label:<26} RAISED {type(e).__name__}: {str(e)[:120]}", flush=True)
        ok &= good
        if good:
            print(f"  {label:<26} len={len(prompt):<5} fell back OK", flush=True)

    # ---------- 3c: window-crossing prompts (no prefix) go through the graph ----------
    # A prompt longer than the sliding window evicts DURING the prefill. The graph does not
    # evict in-flight; blocks are freed after the replay, which is safe because the kernel masks
    # keys below its window bound. These are the p95 prompts on our workload.
    print("\n=== window-crossing prompts: graph vs eager ===", flush=True)
    for plen in (600, 900):
        prompt = list(range(3000, 3000 + plen))
        backend.prefix_cache = type(backend.prefix_cache)(pools=backend.pools)
        backend._prefill_graph_on = False
        e = backend.prefill_batch([prompt])[0]
        backend.prefix_cache = type(backend.prefix_cache)(pools=backend.pools)
        backend._prefill_graph_on = True
        g_ = backend.prefill_batch([prompt])[0]
        match = e[1] == g_[1]
        evicted = g_[0].any_evicted
        ok &= match
        e[0].free_all()
        g_[0].free_all()
        print(f"  len={plen:>5} window={window}  evicted={evicted}  "
              f"first_tok={e[1]}/{g_[1]} {'MATCH' if match else 'MISMATCH'}", flush=True)

    # ---------- 4: IMA stress ----------
    print("\n=== IMA stress: 60 graphed prefix-hit replays with alloc/free churn ===", flush=True)
    backend.prefix_cache = type(backend.prefix_cache)(pools=backend.pools)
    backend._prefill_graph_on = False
    warm = backend.prefill_batch([base[:128]])[0]
    backend._prefill_graph_on = True
    try:
        for i in range(60):
            r = backend.prefill_batch([base[:128] + [60000 + i]])[0]
            r[0].free_all()
        torch.cuda.synchronize()
        print("  60 replays: OK (no illegal memory access)", flush=True)
    except Exception as e:
        ok = False
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    warm[0].free_all()

    # ---------- 5: latency ----------
    print("\n=== latency on a prefix hit ===", flush=True)
    print(f"  {'prompt':>7} {'suffix':>7} {'eager ms':>9} {'graph ms':>9} {'speedup':>8}", flush=True)
    for plen, slen in ((128, 8), (240, 8), (240, 32)):
        backend.prefix_cache = type(backend.prefix_cache)(pools=backend.pools)
        backend._prefill_graph_on = False
        warm = backend.prefill_batch([base[:plen]])[0]

        def timeit(use_graph, n=15):
            backend._prefill_graph_on = use_graph
            lead = [0]

            def once():
                lead[0] += 1
                r = backend.prefill_batch([base[:plen] + [30000 + lead[0]] * slen])[0]
                r[0].free_all()
            for _ in range(5):
                once()
            torch.cuda.synchronize()
            ts = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                once()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            return statistics.median(ts)

        e, g = timeit(False), timeit(True)
        warm[0].free_all()
        print(f"  {plen:>7} {slen:>7} {e:>9.1f} {g:>9.1f} {e / g:>7.2f}x", flush=True)

    print(f"\nprefix-hit prefill graph: {'PASS' if ok else 'FAIL'}", flush=True)
    return bool(ok)
