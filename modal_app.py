"""Modal deployment entrypoint — wraps the FastAPI server in a GPU container.

One container per GPU; `@modal.concurrent` lets our continuous-batch scheduler
do the batching (without it, Modal would shard requests across containers and
bypass the scheduler entirely — defeating the whole project).
"""

import modal

# CUDA-enabled torch first, from PyTorch's index — avoids the silent CPU-wheel
# fallback. Subsequent pyproject install sees torch already satisfies the spec
# and leaves it alone.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .env({
        "BACKEND": "custom-cuda",                 # hand-written Gemma 4 forward + paged KV + kernel
        # Windowed KV storage lets sliding pools be smaller (capped at the 512 window) so the
        # binding full pools can be larger at ~equal memory → more concurrent long-context reqs.
        "CUSTOM_BACKEND_BLOCKS": "2048",          # full-attention pools (grow with sequence)
        "CUSTOM_BACKEND_SLIDING_BLOCKS": "1200",  # sliding pools (capped at window → sized smaller)
        "MAX_ACTIVE_KV_TOKENS": "48000",          # coarse token cap; per-pool window-aware gate is the real limit
    })
    .add_local_python_source("inference_server")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App("inference-server", image=image)


@app.function(
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=256)
@modal.asgi_app()
def fastapi_app():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
    from inference_server.server import app as web_app
    return web_app
