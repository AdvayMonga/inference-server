"""A/B the bucketed decode CUDA graphs against the old single-max-batch graph (A100/E4B).

Baseline (main) captured ONE graph at max_batch_size and replayed it at full size every step,
so 4 active rows with max_batch=256 ran a 256-row forward — 64x padding. This measures what
that cost and proves the bucketed replay is token-identical.

  PARITY: replaying bucket B must equal running the SAME ops eagerly over the SAME B-wide
          static buffers. Comparing against UNPADDED eager is NOT a valid gate: a [B,1,H] GEMM
          picks a different cuBLAS kernel per B, so pad width alone shifts logits by ~0.5 and
          flips near-tie argmaxes — reproduced with no graph at all in diag_pad_numerics_modal.
  PERF:   ms/step at each n, bucketed vs forced-max-bucket (the old behavior, same process).

    venv/bin/modal run --detach scripts/bench_decode_buckets_modal.py
"""

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "MODEL_NAME": os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it"),
        "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "256"),
        "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
        "CONTEXT_WINDOW": "8192",
        "CUSTOM_BACKEND_COMPILE": os.environ.get("CUSTOM_BACKEND_COMPILE", "0"),
    })
    .add_local_python_source("inference_server")   # must be last (no build steps after)
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

    torch.set_grad_enabled(False)   # these call model() directly; without this every
                                    # raw forward retains an autograd graph -> OOM
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

    # ---- PARITY: graph replay vs eager over the SAME padded static buffers ----
    from inference_server.backends.custom_torch_backend import _GraphCtx

    print("=== parity: graph replay vs width-matched eager ===", flush=True)
    backend._graph_on = True
    if not backend._graphs:
        backend._capture_all_graphs()
    ok = True
    for n in [1, 3, 5, 17, 32]:
        bucket = backend._decode_bucket(n)
        state, cur = build(n)
        state.prepare_step()
        tokens = torch.tensor([[t] for t in cur], device=dev)
        pos = torch.tensor([[int(state.seq_lens[i])] for i in range(n)], device=dev)

        graphed = backend._replay_decode(tokens, pos, state, bucket)[:, -1, :].float().clone()
        # _replay_decode already staged everything into this bucket's buffers; rerun eagerly
        # over those exact buffers so the only variable is graph-replay vs live dispatch.
        g = backend._graphs[bucket]
        ctx = _GraphCtx(backend.pools, g["seqlens"], g["bt"], backend._block_size)
        eager = backend._decode_fwd(g["tokens"], position_ids=g["pos"],
                                    paged_ctx=ctx)[:n, -1, :].float()
        free(state)

        same_tok = torch.equal(graphed.argmax(-1), eager.argmax(-1))
        diff = float((graphed - eager).abs().max())
        ok &= same_tok
        print(f"  n={n:<4} bucket={bucket:<4} argmax {'MATCH' if same_tok else 'MISMATCH'}"
              f"  maxdiff={diff:.6f}", flush=True)
    print(f"parity: {'PASS' if ok else 'FAIL'}", flush=True)

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
