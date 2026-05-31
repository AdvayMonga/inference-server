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
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .env({"COMPILE_MODEL": "true"})
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
    from inference_server.server import app as web_app
    return web_app
