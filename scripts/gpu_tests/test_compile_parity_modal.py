"""Parity: does torch.compile (CUSTOM_BACKEND_COMPILE) change decode outputs? It fuses bf16
element-wise ops, which can reorder reductions and flip rare argmax near-ties — same class as our
Triton kernel's ~1% noise — so the bar is greedy-token AGREEMENT, not byte-identity.

Compares the production path (compiled + CUDA graph) vs the uncompiled reference (graph, eager
math), both driven through the real decode_step_batched over N rows for K greedy steps.

    BENCH_GPU=A100-80GB BENCH_MODEL=google/gemma-4-E4B-it \\
        venv/bin/modal run scripts/gpu_tests/test_compile_parity_modal.py
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
N, SEQ, K = 8, 64, 40

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"BENCH_GPU": GPU, "BENCH_MODEL": MODEL,
          "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096"})
    .add_local_python_source("inference_server")
)
app = modal.App("compile-parity", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def run():
    import torch
    from inference_server.backends import create_backend
    from inference_server.models.paged_kv_cache import PagedKVCache

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    model, dev = backend.model, backend.device

    def greedy_run():
        # fresh state (deterministic prompts → identical across runs), K greedy decode steps
        state, cur = None, []
        for i in range(N):
            prompt = [2] + [100 + ((i * 7 + j) % 50000) for j in range(SEQ - 1)]
            c = PagedKVCache(pools=backend.pools)
            tok = int(model(torch.tensor([prompt], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            state = backend.splice_into_batched(state, c, c.seq_len)
            cur.append(tok)
        seqs = [[t] for t in cur]
        for _ in range(K):
            tokens = torch.tensor([[t] for t in cur], device=dev)
            pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            nxt, _ = backend.decode_step_batched(tokens, state, None, pos)
            cur = [int(nxt[i]) for i in range(N)]
            for i in range(N):
                seqs[i].append(cur[i])
        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)
        return seqs

    # reference: uncompiled, graph on
    backend._compile_on, backend._decode_fwd = False, model
    backend._graph_on, backend._graph = True, None
    ref = greedy_run()

    # production: compiled + graph
    backend._decode_fwd = torch.compile(model, dynamic=False)
    backend._graph = None  # force recapture over the compiled forward
    cmp = greedy_run()

    total = N * (K + 1)
    agree = sum(int(a == b) for rs, cs in zip(ref, cmp) for a, b in zip(rs, cs))
    print(f"\ngreedy-token agreement (compiled+graph vs uncompiled): "
          f"{agree}/{total} = {agree / total * 100:.1f}%", flush=True)
    for i in range(min(N, 3)):
        print(f"  row {i} ref: {ref[i][:12]}", flush=True)
        print(f"  row {i} cmp: {cmp[i][:12]}", flush=True)
    print("\nPASS" if agree / total >= 0.97 else "\nFAIL (<97% agreement)", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
