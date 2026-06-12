"""GPU integration test: persistent-state batched decode == per-row decode (M-cudagraphs Step 1).

Drives the real backend path — prefill → splice_into_batched (BatchedDecodeState) →
decode_step_batched — and checks it produces the same tokens as decoding each row alone
through the validated single-cache path. Run:

    venv/bin/modal run scripts/test_paged_kernel_integration_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("paged-state-integration", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
PROMPTS = [
    [2, 651, 6037, 576, 6081, 603, 8, 235248],
    [2, 1841, 603, 573],
    [2, 23274, 1432, 573, 2778, 692, 6],
]
STEPS = 12


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=600)
def run():
    import torch
    from inference_server.backends import create_backend
    from inference_server.models.paged_kv_cache import PagedKVCache

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    model, dev = backend.model, backend.device

    def reference():
        """Per-row decode through the single-cache model path (validated)."""
        gen = []
        for p in PROMPTS:
            c = PagedKVCache(pools=backend.pools)
            tok = int(model(torch.tensor([p], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            seq = [tok]
            for _ in range(STEPS - 1):
                logits = model(torch.tensor([[tok]], device=dev),
                               position_ids=torch.tensor([[c.seq_len]], device=dev), kv_cache=c)
                tok = int(logits[:, -1, :].argmax(-1))
                seq.append(tok)
            c.free_all()
            gen.append(seq)
        return gen

    def batched_via_state():
        """Real backend path: prefill → splice into BatchedDecodeState → decode_step_batched."""
        state, cur = None, []
        for p in PROMPTS:
            c = PagedKVCache(pools=backend.pools)
            tok = int(model(torch.tensor([p], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            state = backend.splice_into_batched(state, c, c.seq_len)
            cur.append(tok)
        gen = [[t] for t in cur]
        for _ in range(STEPS - 1):
            tokens = torch.tensor([[t] for t in cur], device=dev)
            pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            nxt, state = backend.decode_step_batched(tokens, state, None, pos)
            cur = [int(nxt[i]) for i in range(len(cur))]
            for i, t in enumerate(cur):
                gen[i].append(t)
        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)
        return gen

    ref = reference()
    got = batched_via_state()
    ok = ref == got
    for i, (r, g) in enumerate(zip(ref, got)):
        print(f"row {i}: ref={r}")
        print(f"row {i}: got={g}  {'OK' if r == g else 'DIFF'}")
    print("STATE INTEGRATION", "OK" if ok else "FAIL")
    return ok


@app.local_entrypoint()
def main():
    print("state integration:", run.remote())
