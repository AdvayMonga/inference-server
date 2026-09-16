"""Integration test: CUDA prefill_batch (paged prefill kernel) vs per-row prefill (SDPA).

Checks the WIRING (the kernel is correctness-tested in isolation elsewhere): both the cold path
(fresh cache → full prefill) and the warm path (cache hit → suffix-only) must produce the right
tokens. Kernel uses online softmax → ~1% logit noise vs SDPA, so we check first-token argmax
agreement + decode agreement (allowing rare near-tie flips), not byte-identity.

    venv/bin/modal run scripts/gpu_tests/test_prefill_batch_kernel_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("prefill-batch-kernel-test", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
TEXTS = ["The capital of France is", "Water boils at a temperature of",
         "The opposite of hot is", "Two plus two equals"]


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def run():
    import torch
    from inference_server.backends import create_backend
    from inference_server.models.paged_kv_cache import PrefixCache

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    prompts = [backend.tokenizer(t, return_tensors="pt").input_ids[0].tolist() for t in TEXTS]

    def decode(cache, first, n=5):
        out, tok = [first], first
        for _ in range(n):
            logits = backend.model(torch.tensor([[tok]], device=backend.device), kv_cache=cache)
            tok = int(logits[:, -1, :].argmax(-1))
            out.append(tok)
        return out

    def singles(ps):
        backend.prefix_cache = PrefixCache(pools=backend.pools)
        return [backend.prefill(p) for p in ps]

    ref = singles(prompts)
    ref_first = [r[1] for r in ref]
    ref_dec = [decode(r[0], r[1]) for r in ref]

    def check(label, batched):
        first = [b[1] for b in batched]
        kvlen = [b[2] for b in batched]
        agree = sum(a == b for a, b in zip(first, ref_first))
        dec_ok = sum(decode(b[0], b[1]) == d for b, d in zip(batched, ref_dec))
        print(f"{label}: kvlen_ok={kvlen == [len(p) for p in prompts]}  "
              f"first-token agree {agree}/{len(prompts)}  decode-match {dec_ok}/{len(prompts)}")
        return agree

    # COLD: fresh cache → kernel does a full prefill
    backend.prefix_cache = PrefixCache(pools=backend.pools)
    a1 = check("cold (full)", backend.prefill_batch(prompts))

    # WARM: populate cache, then kernel does suffix-only
    backend.prefix_cache = PrefixCache(pools=backend.pools)
    for p in prompts:
        backend.prefill(p)
    a2 = check("warm (suffix-only)", backend.prefill_batch(prompts))

    print("PREFILL_BATCH KERNEL", "OK" if a1 >= 3 and a2 >= 3 else "FAIL")


@app.local_entrypoint()
def main():
    run.remote()
