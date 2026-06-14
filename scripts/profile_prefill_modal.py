"""Confirm the TTFT diagnosis: is it eager, serial prefill?

Hypothesis: prefill runs EAGER (the CUDA graph is decode-only) at ~65 ms/call, and admission
serializes N prefills per wave, so the last-admitted request's TTFT ≈ N × 65 ms (matches the
sweep's 1841 ms @ N=32). Measures, on A10G/E2B:
  - one prefill (uncached 60-tok, and cached = 1-token suffix)
  - one graphed decode step (for the ratio)
  - 32 back-to-back prefills (the admission-wave cost vs the measured TTFT)

    venv/bin/modal run scripts/profile_prefill_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"CUSTOM_BACKEND_BLOCKS": "2048", "CUSTOM_BACKEND_SLIDING_BLOCKS": "1200", "MAX_BATCH_SIZE": "32"})
    .add_local_python_source("inference_server")
)
app = modal.App("profile-prefill", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
PROMPT_TOKENS = 60


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def run():
    import time, torch, statistics
    from inference_server.backends import create_backend

    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    dev = backend.device

    def sync():
        torch.cuda.synchronize()

    def time_prefill(prompt, n=20):
        ts = []
        for _ in range(n):
            sync(); t0 = time.perf_counter()
            cache, _, _ = backend.prefill(list(prompt))
            sync(); ts.append((time.perf_counter() - t0) * 1000)
            cache.free_all()  # free row blocks
        return statistics.median(ts)

    # warmup
    for i in range(5):
        c, _, _ = backend.prefill([60000 + i] + list(range(100, 100 + PROMPT_TOKENS)))
        c.free_all()

    # uncached: distinct prompt each call (lead token varies)
    cnt = {"n": 0}
    def uncached():
        cnt["n"] += 1
        return [40000 + cnt["n"]] + list(range(100, 100 + PROMPT_TOKENS))
    uncached_ms = statistics.median([_t(backend, uncached(), sync) for _ in range(20)])

    # cached: same prompt repeated (after first, it's a 1-token-suffix cache hit)
    fixed = [55555] + list(range(100, 100 + PROMPT_TOKENS))
    backend.prefill(list(fixed)); # populate cache
    cached_ms = time_prefill(fixed, n=20)

    # graphed decode step (1 row)
    from inference_server.models.paged_kv_cache import PagedKVCache
    c = PagedKVCache(pools=backend.pools)
    tok = int(backend.model(torch.tensor([fixed], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
    state = backend.splice_into_batched(None, c, c.seq_len)
    cur = [tok]
    def decode_once():
        nonlocal cur
        tokens = torch.tensor([[t] for t in cur], device=dev)
        pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
        nxt, _ = backend.decode_step_batched(tokens, state, None, pos)
        cur = [int(nxt[0])]
    for _ in range(5): decode_once()
    dts = []
    for _ in range(20):
        sync(); t0 = time.perf_counter(); decode_once(); sync()
        dts.append((time.perf_counter() - t0) * 1000)
    decode_ms = statistics.median(dts)

    # 32 back-to-back prefills (the admission wave)
    caches = []
    sync(); t0 = time.perf_counter()
    for i in range(32):
        cc, _, _ = backend.prefill([42000 + i] + list(range(100, 100 + PROMPT_TOKENS)))
        caches.append(cc)
    sync(); wave_ms = (time.perf_counter() - t0) * 1000
    for cc in caches:
        cc.free_all()

    # batched prefill scaling — dispatch-bound (flat per-row) or compute-bound (total grows with K)?
    print()
    for Kb in (1, 8, 32):
        prompts = [[43000 + j] + list(range(100, 100 + PROMPT_TOKENS)) for j in range(Kb)]
        sync(); t0 = time.perf_counter()
        res = backend.prefill_batch(prompts)
        sync(); bt = (time.perf_counter() - t0) * 1000
        for cc, _, _ in res:
            cc.free_all()
        print(f"prefill_batch K={Kb:2d} (full): {bt:6.1f} ms  (per-row {bt / Kb:.1f})")

    print(f"\nprefill uncached (60-tok):  {uncached_ms:6.1f} ms")
    print(f"prefill cached (1-tok hit): {cached_ms:6.1f} ms")
    print(f"graphed decode step:        {decode_ms:6.1f} ms")
    print(f"prefill / decode ratio:     {uncached_ms / decode_ms:5.1f}x")
    print(f"32 back-to-back prefills:   {wave_ms:6.1f} ms  (per={wave_ms/32:.1f})")
    print(f"  → vs measured TTFT @ N=32: 1841 ms")


def _t(backend, prompt, sync):
    import time
    sync(); t0 = time.perf_counter()
    cache, _, _ = backend.prefill(list(prompt))
    sync(); dt = (time.perf_counter() - t0) * 1000
    cache.free_all()
    return dt


@app.local_entrypoint()
def main():
    run.remote()
