"""Diagnose the n=17 eager-vs-graph token mismatch: real bug or argmax noise?

Two questions the token comparison can't answer:
  1. Is the eager-vs-graph LOGIT difference at the bf16/Triton noise floor, and is the
     diverging row a near-tie at the top of the distribution (so any epsilon flips it)?
  2. Is the graphed result BUCKET-INVARIANT? If the same 17 rows give different tokens
     when replayed at bucket 32 vs 64 vs 256, padding is corrupting real rows -> real bug.
     If they agree across buckets but differ from eager, it is numerics.

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/diag_bucket_parity_modal.py
"""

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "MODEL_NAME": os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it"),
        "MAX_BATCH_SIZE": "256",
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "CONTEXT_WINDOW": "8192", "CUSTOM_BACKEND_COMPILE": "0",
    })
    .add_local_python_source("inference_server")
)
app = modal.App("bucket-parity-diag", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

N = 17


@app.function(gpu="A100-80GB", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=3600)
def run():
    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PagedKVCache

    torch.set_grad_enabled(False)   # these call model() directly; without this every
                                    # raw forward retains an autograd graph -> OOM
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    model, dev = backend.model, backend.device

    def prompt(i):
        return [2, 651, 6037, 576, 100 + i, 603, 8, 235248 - i]

    def build(n):
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

    def one_step_logits(state, cur, forced_bucket=None):
        """Run ONE decode step; return the [n, V] last-position logits."""
        real = type(backend)._decode_bucket
        if forced_bucket is not None:
            type(backend)._decode_bucket = lambda self, k: forced_bucket
        try:
            tokens = torch.tensor([[t] for t in cur], device=dev)
            pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            n = state.n_rows
            state.prepare_step()
            if backend._graph_on:
                if not backend._graphs:
                    backend._capture_all_graphs()
                b = forced_bucket if forced_bucket is not None else backend._decode_bucket(n)
                logits = backend._replay_decode(tokens, pos, state, b)
            else:
                logits = backend._decode_fwd(tokens, position_ids=pos, paged_ctx=state)
            state.advance()
            return logits[:, -1, :].float().clone()
        finally:
            type(backend)._decode_bucket = real

    # ---- Q1: eager vs graph logits on identical state ----
    backend._graph_on = False
    s, c = build(N)
    eager = one_step_logits(s, c)
    free(s)

    backend._graph_on = True
    s, c = build(N)
    graphed = one_step_logits(s, c)
    free(s)

    print("\n=== Q1: eager vs graphed logits, one step, n=17 ===", flush=True)
    print(f"{'row':>4} {'maxdiff':>10} {'top1-top2 (eager)':>18} {'argmax e/g':>14} {'same':>5}",
          flush=True)
    for i in range(N):
        e, g = eager[i], graphed[i]
        top2 = e.topk(2).values
        gap = float(top2[0] - top2[1])
        ae, ag = int(e.argmax()), int(g.argmax())
        print(f"{i:>4} {float((e - g).abs().max()):>10.5f} {gap:>18.5f} "
              f"{ae:>6}/{ag:<7} {'YES' if ae == ag else 'NO':>5}", flush=True)

    # ---- Q2: bucket invariance of the graphed result ----
    print("\n=== Q2: graphed tokens vs replay bucket (same 17 rows, 8 steps) ===", flush=True)
    results = {}
    for bucket in (32, 64, 128, 256):
        backend._graph_on = True
        s, c = build(N)
        seq = [[] for _ in range(N)]
        for _ in range(8):
            real = type(backend)._decode_bucket
            type(backend)._decode_bucket = lambda self, k, b=bucket: b
            try:
                tokens = torch.tensor([[t] for t in c], device=dev)
                pos = torch.tensor([[int(s.seq_lens[i])] for i in range(s.n_rows)], device=dev)
                nxt, _ = backend.decode_step_batched(tokens, s, None, pos)
            finally:
                type(backend)._decode_bucket = real
            c = [int(nxt[i]) for i in range(N)]
            for i, t in enumerate(c):
                seq[i].append(t)
        free(s)
        results[bucket] = seq
        print(f"  bucket {bucket:>3}: row16 = {seq[16]}", flush=True)

    base = results[32]
    invariant = all(results[b] == base for b in results)
    print(f"\nbucket-invariant: {'YES (padding is clean; divergence is numerics)' if invariant else 'NO (REAL BUG)'}",
          flush=True)
    return invariant
