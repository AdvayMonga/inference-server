"""Does mixed-batch prefill (V-B) actually pay on THIS engine? Measure before building.

V-B packs decode tokens and a prefill chunk into ONE forward so prefill never stalls decode.
That is unambiguously right for vLLM. But it has a precondition we may not meet: a mixed step
has a variable (B, S) shape, so it CANNOT replay our decode CUDA graph — it must run eager.

Our eager forward is dispatch-bound and nearly flat in size (bench_prefill_graph_modal measured
108.8ms at 64 tokens and 113.2ms at 480). If an eager mixed step costs ~100ms while a graphed
decode step costs ~20ms, then chunking a prompt across k mixed steps costs k*100ms and is far
WORSE than the current stall of one graphed prefill (43ms after the prefill-graph fix).

So measure the two numbers the decision turns on:
  A. graphed decode step (what we have today, per row count)
  B. eager  decode step (the floor cost of any mixed step at that row count)
The ratio B/A is the per-step tax V-B would pay on every step carrying prefill work.

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/probes/probe_mixed_step_premise_modal.py
"""

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "MODEL_NAME": os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it"),
        "MAX_BATCH_SIZE": "32", "CONTEXT_WINDOW": "8192",
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "KV_CACHE_NUM_BLOCKS": "16384", "CUSTOM_BACKEND_COMPILE": "0",
    })
    .add_local_python_source("inference_server")
)
app = modal.App("mixed-step-premise", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="A100-80GB", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def run():
    import time

    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PagedKVCache

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    model, dev = backend.model, backend.device

    def build(n):
        state, cur = None, []
        for i in range(n):
            c = PagedKVCache(pools=backend.pools)
            ids = [2, 651, 6037, 576, 100 + i, 603, 8, 235248 - i]
            tok = int(model(torch.tensor([ids], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            state = backend.splice_into_batched(state, c, c.seq_len)
            cur.append(tok)
        return state, cur

    def free(state):
        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)

    def measure(n, graph_on, iters=25):
        backend._graph_on = graph_on
        state, cur = build(n)

        def step():
            nonlocal cur
            t = torch.tensor([[x] for x in cur], device=dev)
            p = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            nxt, _ = backend.decode_step_batched(t, state, None, p)
            cur = [int(x) for x in nxt]

        for _ in range(8):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
        free(state)
        return ms

    print("=== per-step cost: graphed vs eager (the V-B tax) ===", flush=True)
    print(f"{'rows':>5} {'graphed ms':>11} {'eager ms':>10} {'eager tax':>10}", flush=True)
    rows = []
    for n in (1, 4, 8, 16, 32):
        g = measure(n, True)
        e = measure(n, False)
        rows.append((n, g, e))
        print(f"{n:>5} {g:>11.2f} {e:>10.2f} {e / g:>9.2f}x", flush=True)

    print("\n=== what that means for a 240-token prompt ===", flush=True)
    n, g, e = rows[2]  # 8 rows
    for chunks in (1, 2, 4):
        print(f"  V-B, {chunks} chunk(s): {chunks} eager mixed steps = {chunks * e:6.1f} ms "
              f"of step time (vs {g:.1f} ms/step graphed)", flush=True)
    print(f"  today (prefill graph): one 43 ms prefill stall, decode steps stay graphed at "
          f"{g:.1f} ms", flush=True)
    return [(n, round(g, 2), round(e, 2)) for n, g, e in rows]
