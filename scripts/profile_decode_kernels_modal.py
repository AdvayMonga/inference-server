"""Break down one decode step (custom-cuda) by GPU kernel — locate the 43ms. The A100/E4B
head-to-head showed we run decode at ~18% of A100 bandwidth vs vLLM's ~68%; this says where
the other 82% goes (GEMV weight-reads vs the element-wise/launch tail), before torch.compile.

Two measurements:
  - graph ON  → real ms/step + achieved GB/s (% of peak). The headline efficiency number.
  - eager     → per-kernel CUDA self-time breakdown. The graph replays THESE same kernels with
                the launch gaps removed, so the eager mix shows what fusion would collapse.
                (Under the graph the profiler sees only one opaque cudaGraphLaunch.)

    BENCH_GPU=A100-80GB BENCH_MODEL=google/gemma-4-E4B-it \\
        venv/bin/modal run --detach scripts/profile_decode_kernels_modal.py
"""

import os

import modal

GPU = os.environ.get("BENCH_GPU", "A100-80GB")
MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
N = int(os.environ.get("PROFILE_N", "32"))
SEQ = int(os.environ.get("PROFILE_SEQ", "160"))
PEAK_GBPS = float(os.environ.get("PEAK_GBPS", "2039"))  # A100-80GB SXM4 HBM2e
STEPS, WARMUP, PROFILE_STEPS = 50, 10, 20

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    # Bake BENCH_* + pools into the image env — the remote fn reads them as module globals and
    # Modal re-imports this module in the container (no local-shell env). See the BENCH_MODEL bug.
    .env({"BENCH_GPU": GPU, "BENCH_MODEL": MODEL,
          "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096"})
    .add_local_python_source("inference_server")
)
app = modal.App("profile-decode-kernels", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=900)
def run():
    import time, torch
    from torch.profiler import ProfilerActivity, profile
    from inference_server.backends import create_backend
    from inference_server.models.paged_kv_cache import PagedKVCache

    print(f"GPU={GPU} MODEL={MODEL} N={N} SEQ={SEQ}", flush=True)
    backend = create_backend("custom-cuda")
    backend.load_model(MODEL)
    model, dev = backend.model, backend.device
    wbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"param bytes = {wbytes / 2**30:.2f} GiB (read once per decode step)\n", flush=True)

    def build_state():
        state, cur = None, []
        for i in range(N):
            prompt = [2] + [100 + ((i * 7 + j) % 50000) for j in range(SEQ - 1)]
            c = PagedKVCache(pools=backend.pools)
            tok = int(model(torch.tensor([prompt], device=dev), kv_cache=c)[:, -1, :].argmax(-1))
            state = backend.splice_into_batched(state, c, c.seq_len)
            cur.append(tok)

        def step():
            tokens = torch.tensor([[t] for t in cur], device=dev)
            pos = torch.tensor([[int(state.seq_lens[i])] for i in range(state.n_rows)], device=dev)
            nxt, _ = backend.decode_step_batched(tokens, state, None, pos)
            for i in range(len(cur)):
                cur[i] = int(nxt[i])

        return state, step

    def teardown(state):
        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)

    # --- 1. real graphed step: ms/step + achieved bandwidth ---
    backend._graph_on, backend._graph = True, None
    state, step = build_state()
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1000
    gbps = wbytes / (ms / 1000) / 1e9
    print(f"[graph ON] {ms:.2f} ms/step  |  {N * 1000 / ms:.0f} tok/s  |  "
          f"{gbps:.0f} GB/s = {gbps / PEAK_GBPS * 100:.0f}% of {PEAK_GBPS:.0f} GB/s peak\n", flush=True)
    teardown(state)

    # --- 2. eager step: per-kernel breakdown (what the graph internally runs) ---
    backend._graph_on, backend._graph = False, None
    state, step = build_state()
    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        step()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / STEPS * 1000
    print(f"[eager] {eager_ms:.2f} ms/step (no graph; gap vs graphed = launch-bubble removed)\n",
          flush=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILE_STEPS):
            step()
        torch.cuda.synchronize()
    teardown(state)

    ka = prof.key_averages()

    def dev_us(evt):  # GPU self-time, across torch versions
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(evt, attr, None)
            if v:
                return v
        return 0.0

    print(ka.table(sort_by="self_cuda_time_total", row_limit=30), flush=True)

    # Bucket GPU self-time → GEMM (weight reads) vs attention vs element-wise/launch tail.
    buckets = {"gemm/matmul": 0.0, "attention": 0.0, "elementwise/norm/copy": 0.0, "other": 0.0}
    total = 0.0
    for evt in ka:
        t = dev_us(evt)
        if t <= 0:
            continue
        total += t
        name = evt.key.lower()
        if any(s in name for s in ("gemm", "cutlass", "ampere", "sm80", "cublas", "matmul", "addmm", "wgrad", "dot")):
            buckets["gemm/matmul"] += t
        elif any(s in name for s in ("attention", "paged", "softmax", "flash", "attn")):
            buckets["attention"] += t
        elif any(s in name for s in ("elementwise", "vectorized", "norm", "reduce", "rope", "rotary",
                                     "cat", "copy", "index", "gather", "scatter", "mul", "add", "gelu", "fill")):
            buckets["elementwise/norm/copy"] += t
        else:
            buckets["other"] += t

    print(f"\n=== GPU self-time buckets (per {PROFILE_STEPS} steps, total {total / 1000:.1f} ms) ===",
          flush=True)
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {v / 1000:>8.2f} ms  {v / total * 100:>5.1f}%", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
