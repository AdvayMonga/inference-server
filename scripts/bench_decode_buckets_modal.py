"""A/B the bucketed decode CUDA graphs against the old single-max-batch graph (A100/E4B).

Baseline (main) captured ONE graph at max_batch_size and replayed it at full size every step,
so 4 active rows with max_batch=256 ran a 256-row forward — 64x padding. This measures what
that cost and proves the bucketed replay is token-identical.

  PARITY: graphed decode must produce the SAME tokens as eager decode at every n.
  PERF:   ms/step at each n, bucketed vs forced-max-bucket (the old behavior, same process).

    venv/bin/modal run --detach scripts/bench_decode_buckets_modal.py
"""

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
    .env({
        "MODEL_NAME": os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it"),
        "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "256"),
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "CONTEXT_WINDOW": "8192",
        "CUSTOM_BACKEND_COMPILE": os.environ.get("CUSTOM_BACKEND_COMPILE", "0"),
    })
)
app = modal.App("decode-bucket-ab", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
STEPS = 30
WARMUP = 10


@app.function(gpu=os.environ.get("BENCH_GPU", "A100-80GB"),
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=3600)
def run():
    import time

    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PagedKVCache

    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    model, dev = backend.model, backend.device
    print(f"model={settings.model_name} max_batch={backend._graph_max_rows} "
          f"buckets={backend._graph_buckets} compile={backend._compile_on}\n", flush=True)

    def prompt(i):
        return [2, 651, 6037, 576, 100 + i, 603, 8, 235248 - i]

    def build(n):
        """Prefill n distinct rows into a BatchedDecodeState; return (state, first_tokens)."""
        state, cur = None, []
        for i in range(n):
            c = PagedKVCache(pools=backend.pools)
            tok = int(model(torch.tensor([prompt(i)], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            state = backend.splice_into_batched(state, c, c.seq_len)
            cur.append(tok)
        return state, cur

    def free(state):
        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)

    def steps(state, cur, k):
        """Run k decode steps, return the token sequence produced per row."""
        out = [[] for _ in range(len(cur))]
        for _ in range(k):
            tokens = torch.tensor([[t] for t in cur], device=dev)
            pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            nxt, _ = backend.decode_step_batched(tokens, state, None, pos)
            cur = [int(nxt[i]) for i in range(len(cur))]
            for i, t in enumerate(cur):
                out[i].append(t)
        return out

    # ---- PARITY: graphed (bucketed) vs eager, same prompts, same steps ----
    print("=== parity: bucketed graph vs eager ===", flush=True)
    ok = True
    for n in [1, 3, 5, 17, 32]:
        backend._graph_on = False
        s, c = build(n)
        want = steps(s, c, 8)
        free(s)

        backend._graph_on = True
        s, c = build(n)
        got = steps(s, c, 8)
        free(s)

        match = want == got
        ok &= match
        print(f"  n={n:<4} {'MATCH' if match else 'MISMATCH'}"
              f"{'' if match else f' eager={want} graph={got}'}", flush=True)
    print(f"parity: {'PASS' if ok else 'FAIL'}\n", flush=True)

    # ---- PERF: bucketed vs forced-max (the old single-graph behavior) ----
    backend._graph_on = True
    maxb = backend._graph_buckets[-1]
    real_pick = type(backend)._decode_bucket

    def measure(n, force_max):
        # Emulate main's behavior by pinning every replay to the largest bucket.
        type(backend)._decode_bucket = ((lambda self, k: maxb) if force_max else real_pick)
        s, c = build(n)
        steps(s, c, WARMUP)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps(s, c, STEPS)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / STEPS * 1000
        free(s)
        return ms

    print("=== perf: ms/step ===", flush=True)
    print(f"{'n':>5} {'bucket':>7} {'bucketed':>10} {'forced-max':>11} {'speedup':>8}", flush=True)
    rows = []
    for n in NS:
        if n > backend._graph_max_rows:
            continue
        b = real_pick(backend, n)
        fast = measure(n, force_max=False)
        slow = measure(n, force_max=True)
        rows.append((n, b, fast, slow))
        print(f"{n:>5} {b:>7} {fast:>10.2f} {slow:>11.2f} {slow / fast:>7.2f}x", flush=True)

    type(backend)._decode_bucket = real_pick
    print("\nn,bucket,ms_bucketed,ms_forced_max,speedup", flush=True)
    for n, b, fast, slow in rows:
        print(f"{n},{b},{fast:.3f},{slow:.3f},{slow / fast:.3f}", flush=True)
    return ok
