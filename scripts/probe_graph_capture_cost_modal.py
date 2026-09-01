"""How much startup does the decode-graph bucket ladder cost, and does one dynamic compile fix it?

Bucketing made every bucket a fresh static shape, so with CUSTOM_BACKEND_COMPILE=1 each one
pays its own Inductor compile. A 6-bucket sweep sat >40 min before its first measurement.
This isolates capture cost only (load model -> capture ladder -> report), no serving.

  BENCH_COMPILE=0|1   torch.compile off/on
  BENCH_DYNAMIC=false|auto|true   how the decode forward is compiled (auto = one static compile
                                  then one dynamic recompile that covers every later bucket)

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/probe_graph_capture_cost_modal.py
"""

import os

import modal

MODEL = os.environ.get("BENCH_MODEL", "google/gemma-4-E4B-it")
_env = {
    "MODEL_NAME": MODEL,
    "MAX_BATCH_SIZE": os.environ.get("MAX_BATCH_SIZE", "32"),
    "CUSTOM_BACKEND_COMPILE": os.environ.get("BENCH_COMPILE", "1"),
    "CUSTOM_BACKEND_BLOCKS": "8192", "CUSTOM_BACKEND_SLIDING_BLOCKS": "4096",
    "KV_CACHE_NUM_BLOCKS": "16384", "CONTEXT_WINDOW": "8192",
    "BENCH_DYNAMIC": os.environ.get("BENCH_DYNAMIC", "false"),
    "CUSTOM_BACKEND_GRAPH_BUCKETS": os.environ.get("CUSTOM_BACKEND_GRAPH_BUCKETS", ""),
}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env(_env)
    .add_local_python_source("inference_server")
)
app = modal.App("graph-capture-cost", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=os.environ.get("BENCH_GPU", "A100-80GB"),
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=5400)
def run():
    import time

    import torch

    from inference_server.backends import create_backend
    from inference_server.config import settings
    from inference_server.models.paged_kv_cache import PagedKVCache

    torch.set_grad_enabled(False)
    backend = create_backend("custom-cuda")
    backend.load_model(settings.model_name)

    mode = os.environ.get("BENCH_DYNAMIC", "false")
    if backend._compile_on and mode != "false":
        # Recompile the decode forward with a non-static batch dim so one artifact covers
        # every bucket. "auto" = torch's automatic dynamic shapes (static first, then dynamic).
        backend._decode_fwd = torch.compile(backend.model,
                                            dynamic=(None if mode == "auto" else True))
    print(f"[probe] compile={backend._compile_on} dynamic={mode} "
          f"buckets={backend._graph_buckets}", flush=True)

    c = PagedKVCache(pools=backend.pools)
    backend.model(torch.tensor([[2, 651, 6037, 576, 100]], device=backend.device), kv_cache=c)
    state = backend.splice_into_batched(None, c, c.seq_len)

    t0 = time.perf_counter()
    backend._capture_all_graphs()
    total = time.perf_counter() - t0
    print(f"[probe] TOTAL capture = {total:.1f}s for {len(backend._graphs)} buckets", flush=True)

    # sanity: every captured bucket replays and produces a token
    for b in sorted(backend._graphs):
        tok = torch.tensor([[5]], device=backend.device)
        pos = torch.tensor([[int(state.seq_lens[0])]], device=backend.device)
        state.prepare_step()
        out = backend._replay_decode(tok, pos, state, b)
        state.advance()
        print(f"[probe]   bucket {b:>4} replay OK, argmax={int(out[0, -1].argmax())}", flush=True)
    return total
