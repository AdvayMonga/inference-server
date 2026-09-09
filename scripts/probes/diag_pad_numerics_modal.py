"""Is the bucket-dependent divergence a padding BUG or batch-width GEMM numerics?

Q2 of diag_bucket_parity_modal showed the same 17 rows produce different tokens when replayed
at bucket 32/64 vs 128/256 — reproducibly, and unchanged by the graph memory pool. Two
candidate causes:

  (A) BUG: padding rows corrupt real rows' state (block tables / scratch scatter).
  (B) NUMERICS: a [B,1,H] @ [H,H] GEMM picks a different cuBLAS kernel (different accumulation
      order) per B, so logits differ by ~1 ULP-ish and a near-tie argmax flips.

This isolates them by running EAGER at two padded widths — no CUDA graph anywhere. If eager@32
and eager@128 disagree exactly like graphed@32 and graphed@128 do, the graph and the padding
logic are both exonerated and it is (B).
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
app = modal.App("pad-numerics-diag", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

N = 17


@app.function(gpu="A100-80GB", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=3600)
def run():
    import torch

    from inference_server.backends import create_backend
    from inference_server.backends.custom_torch_backend import _GraphCtx
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PagedKVCache

    torch.set_grad_enabled(False)   # these call model() directly; without this every
                                    # raw forward retains an autograd graph -> OOM
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)
    model, dev = backend.model, backend.device
    cols = backend._graph_max_cols

    scratch = [None] * len(backend.pools)
    for L, pool in enumerate(backend.pools):
        if pool is not None:
            scratch[L] = pool.alloc()

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

    def eager_padded(width, steps=8):
        """Run the real forward EAGERLY over a `width`-row padded batch (no CUDA graph).
        Mirrors exactly what _replay_decode stages into the static buffers."""
        state, cur = build(N)
        seq = [[] for _ in range(N)]
        first_logits = None
        for st in range(steps):
            g_tokens = torch.zeros(width, 1, dtype=torch.long, device=dev)
            g_pos = torch.zeros(width, 1, dtype=torch.long, device=dev)
            g_seq = torch.ones(width, dtype=torch.long, device=dev)
            g_bt = [None] * len(backend.pools)
            for L, pool in enumerate(backend.pools):
                if pool is None:
                    continue
                g_bt[L] = torch.full((width, cols), scratch[L], dtype=torch.long, device=dev)
            state.prepare_step()
            n = state.n_rows
            g_tokens[:n].copy_(torch.tensor([[t] for t in cur], device=dev))
            g_pos[:n].copy_(torch.tensor([[int(state.seq_lens[i])] for i in range(n)], device=dev))
            g_seq[:n].copy_(state.seq_lens)
            for L, pool in enumerate(backend.pools):
                if pool is None:
                    continue
                bt = state.block_tables[L]
                g_bt[L][:n, :bt.shape[1]].copy_(bt.clamp_min(0))
            ctx = _GraphCtx(backend.pools, g_seq, g_bt, backend._block_size)
            logits = model(g_tokens, position_ids=g_pos, paged_ctx=ctx)[:N, -1, :].float()
            state.advance()
            if st == 0:
                first_logits = logits.clone()
            cur = [int(t) for t in logits.argmax(-1).tolist()]
            for i, t in enumerate(cur):
                seq[i].append(t)
        free(state)
        return first_logits, seq

    backend._graph_on = False
    print("=== EAGER (no CUDA graph anywhere), same 17 rows, different pad widths ===", flush=True)
    out = {}
    for width in (17, 32, 64, 128, 256):
        out[width] = eager_padded(width)
        print(f"  width {width:>3}: row16 = {out[width][1][16]}", flush=True)

    print("\n=== step-0 logit diff for row 16, vs unpadded width=17 ===", flush=True)
    base = out[17][0][16]
    for width in (32, 64, 128, 256):
        d = (base - out[width][0][16]).abs().max()
        print(f"  width {width:>3}: maxdiff={float(d):.5f}", flush=True)

    top2 = base.topk(2).values
    print(f"\nrow16 step-0 top1-top2 gap = {float(top2[0] - top2[1]):.5f} "
          f"(a gap this small flips on any epsilon)", flush=True)

    agree = len({tuple(out[w][1][16]) for w in out}) == 1
    print(f"\nEAGER is width-invariant: {'YES -> padding logic is a real BUG' if agree else 'NO -> batch-width GEMM numerics; graphs+padding exonerated'}",
          flush=True)

    # Cross-check: do the other 16 rows (confident argmax) stay identical across widths?
    stable = all(out[17][1][i] == out[w][1][i] for w in out for i in range(N) if i != 16)
    print(f"rows 0-15 identical across all widths: {stable}", flush=True)
    return agree
