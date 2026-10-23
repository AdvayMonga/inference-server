# inference-server

A single-GPU LLM inference system on PyTorch, Triton and CUDA. It serves
Qwen3-30B-A3B behind an OpenAI-compatible API. It is measured and tuned in
[BlameGraph](https://github.com/AdvayMonga/BlameGraph).

- Continuous batching with mixed prefill/decode steps and async decode
- Paged KV cache with a radix-tree prefix cache
- Hand-written Qwen3-MoE forward: Triton MoE kernels, FlashInfer prefill, CUDA-graph decode, a decode megakernel
- FP8 experts, optional FP8 KV cache
- `/v1/completions` (with `prompt_logprobs`), `/v1/chat/completions`, streaming, Prometheus `/metrics`

## Install

```bash
uv sync --locked --extra dev --extra flashinfer
# Linux + CUDA: swap in the CUDA torch wheel at the locked version
uv pip install --reinstall "torch==$(awk '/^name = "torch"$/{getline; gsub(/version = |"/,""); print; exit}' uv.lock)" \
  --index-url https://download.pytorch.org/whl/cu130
```

## Run

```bash
BACKEND=custom-cuda uvicorn inference_server.server:app --port 8000
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-30B-A3B", "prompt": "Hello", "max_tokens": 32}'
```

Settings are environment variables; see [`.env.example`](.env.example) and
[`config.py`](src/inference_server/config.py).

## Test

```bash
pytest -q            # CPU suite
pytest -m heavy      # loads the real model
```
