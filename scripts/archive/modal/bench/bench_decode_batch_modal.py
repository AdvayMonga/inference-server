"""Full-batch decode sweep (A10G): confirm decode is bandwidth-bound + the max-batch graph
amortizes weight reads across rows.

Historical note: this ran when ONE graph was captured at max_batch_size and padded every
smaller batch up to it, so graph-ON ms/step was ~FLAT across real N (fixed 32-row weight reads)
while graph-OFF grew with N — the bandwidth fingerprint. Decode graphs are now bucketed per row
count, so graph-ON tracks N too; the ON/OFF gap here is now dispatch overhead, not padding.

    venv/bin/modal run scripts/bench/bench_decode_batch_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("decode-batch-sweep", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
NS = [1, 4, 8, 16, 32]
STEPS = 50
WARMUP = 10


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def run():
    import time, torch
    from inference_server.backends import create_backend
    from inference_server.models.paged_kv_cache import PagedKVCache

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    model, dev = backend.model, backend.device
    maxb = backend._graph_max_rows
    print(f"max_batch (graph capture rows) = {maxb}\n")

    def make_prompt(i):
        # distinct short prompts so rows don't collapse to one shared cache
        return [2, 651, 6037, 576, 100 + i, 603, 8, 235248 - i]

    def measure(n, graph_on):
        backend._graph_on = graph_on and dev.type == "cuda"
        backend._graphs.clear()  # force re-capture per config
        state, cur = None, []
        for i in range(n):
            c = PagedKVCache(pools=backend.pools)
            tok = int(model(torch.tensor([make_prompt(i)], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            state = backend.splice_into_batched(state, c, c.seq_len)
            cur.append(tok)

        def step():
            nonlocal cur
            tokens = torch.tensor([[t] for t in cur], device=dev)
            pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            nxt, _ = backend.decode_step_batched(tokens, state, None, pos)
            cur = [int(nxt[i]) for i in range(len(cur))]

        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / STEPS * 1000

        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)
        return ms

    for graph_on in (True, False):
        print(f"--- graph {'ON' if graph_on else 'OFF (eager)'} ---")
        print(f"{'N':>4}  {'ms/step':>9}  {'tok/s':>9}  {'ms/tok':>8}")
        for n in NS:
            ms = measure(n, graph_on)
            toks = n * 1000.0 / ms
            print(f"{n:>4}  {ms:>9.2f}  {toks:>9.1f}  {ms/n:>8.2f}")
        print()


@app.local_entrypoint()
def main():
    run.remote()
