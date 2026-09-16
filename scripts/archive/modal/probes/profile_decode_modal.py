"""Profile the decode step to size the CUDA-graph win before building it.

A CUDA graph removes CPU-side per-step cost (Python prep + kernel-launch overhead) but not
GPU compute. So the ceiling = the CPU fraction of each step. We measure, per batch size N:
  - t_step : full decode_step_batched, synced (real per-step latency = CPU + GPU)
  - t_cpu  : CPU time to *issue* the step without waiting (Python + launches, GPU overlaps)
If t_cpu ≈ t_step the step is CPU/launch-bound → graphs win big. If t_cpu << t_step it's
GPU-bound → graphs won't help much. Run:  venv/bin/modal run scripts/probes/profile_decode_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({"BACKEND": "custom-cuda", "CUSTOM_BACKEND_BLOCKS": "2048",
          "CUSTOM_BACKEND_SLIDING_BLOCKS": "1200"})
    .add_local_python_source("inference_server")
)
app = modal.App("profile-decode", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=600)
def run():
    import time
    import torch
    from inference_server.backends import create_backend

    backend = create_backend("custom-cuda")
    backend.load_model("google/gemma-4-E2B-it")
    dev = backend.device

    def perf(N, prompt_len=480, K=40):
        state = None
        for i in range(N):
            ids = [2] + [1000 + (j % 5000) for j in range(prompt_len - 1)]
            kv, _, _ = backend.prefill(ids, session_id=f"s{i}")
            state = backend.splice_into_batched(state, kv, kv.seq_len)

        def one_step():
            cur = torch.tensor([[100]] * N, device=dev)
            pos = state.seq_lens.unsqueeze(1)
            backend.decode_step_batched(cur, state, None, pos)

        for _ in range(5):  # warmup (JIT compile kernels)
            one_step()
        torch.cuda.synchronize()

        # t_step: synced each iteration
        t0 = time.perf_counter()
        for _ in range(K):
            one_step()
            torch.cuda.synchronize()
        t_step = (time.perf_counter() - t0) / K * 1e3

        # t_cpu: drain, then time issuing one step without waiting (CPU issue cost)
        t_cpu_total = 0.0
        for _ in range(K):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_step()
            t_cpu_total += time.perf_counter() - t0
        torch.cuda.synchronize()
        t_cpu = t_cpu_total / K * 1e3

        for i in range(state.n_rows - 1, -1, -1):
            backend.remove_row_from_cache(state, i)
        return t_step, t_cpu

    print(f"{'N':>4} {'t_step(ms)':>11} {'t_cpu(ms)':>10} {'cpu_frac':>9}  (graph ceiling ≈ cpu_frac)")
    for N in [1, 4, 8, 16, 32]:
        t_step, t_cpu = perf(N)
        print(f"{N:>4} {t_step:>11.2f} {t_cpu:>10.2f} {t_cpu / t_step:>8.0%}")


@app.local_entrypoint()
def main():
    run.remote()
