"""int8 weight-only vs bf16 decode throughput (A10G), at full batch under the CUDA graph.

Decode is weight-bandwidth-bound (HANDOFF 2026-06-13: 1481 tok/s @ N=32 bf16). int8 weights
halve the bytes read/step → expect ~1.5-2× decode tok/s. Reports ms/step, tok/s, peak GPU
memory, and GPU token agreement vs bf16 (near-lossless expected; near-ties may flip).

    venv/bin/modal run scripts/bench_quant_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("quant-decode-bench", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
N = 32          # full batch (where the graph amortizes weight reads)
STEPS = 50
WARMUP = 12


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def measure(quant: str):
    """One config per container (two full models won't co-fit on a 22 GB A10G)."""
    import os
    # Headroom for torch.compile autotuning (int8 path); smaller pools don't affect decode speed.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUSTOM_BACKEND_BLOCKS"] = "256"
    os.environ["CUSTOM_BACKEND_SLIDING_BLOCKS"] = "256"
    import time, torch
    from inference_server.models.paged_kv_cache import PagedKVCache

    def make_prompt(i):
        return [2, 651, 6037, 576, 100 + i, 603, 8, 235248 - i]

    os.environ["CUSTOM_BACKEND_QUANT"] = quant
    from inference_server.backends import create_backend
    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    weight_mem = torch.cuda.memory_allocated() / 1e9
    model, dev = backend.model, backend.device

    state, cur = None, []
    for i in range(N):
        c = PagedKVCache(pools=backend.pools)
        tok = int(model(torch.tensor([make_prompt(i)], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
        state = backend.splice_into_batched(state, c, c.seq_len)
        cur.append(tok)
    traj = [list(cur)]

    def step():
        nonlocal cur
        tokens = torch.tensor([[t] for t in cur], device=dev)
        pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
        nxt, _ = backend.decode_step_batched(tokens, state, None, pos)
        cur = [int(nxt[i]) for i in range(len(cur))]
        traj.append(list(cur))

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1000

    return {"ms": ms, "tok_s": N * 1000.0 / ms, "weight_gb": weight_mem, "traj": traj[:6]}


@app.local_entrypoint()
def main():
    bf16 = measure.remote("")
    int8 = measure.remote("int8")

    print(f"\n{'config':>8}  {'ms/step':>9}  {'tok/s':>9}  {'mem GB':>8}")
    for name, r in (("bf16", bf16), ("int8", int8)):
        print(f"{name:>8}  {r['ms']:>9.2f}  {r['tok_s']:>9.1f}  {r['weight_gb']:>8.2f}")
    print(f"\nspeedup int8/bf16: {int8['tok_s']/bf16['tok_s']:.2f}x   "
          f"mem: {int8['weight_gb']/bf16['weight_gb']:.2f}x")

    agree = sum(a == b for ra, rb in zip(bf16["traj"], int8["traj"]) for a, b in zip(ra, rb))
    total = sum(len(r) for r in bf16["traj"])
    print(f"GPU token agreement (first {len(bf16['traj'])} steps): {agree}/{total} = {agree/total:.3f}")
