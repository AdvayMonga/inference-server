"""Gating check for the vLLM head-to-head: can vLLM load + run OUR exact model on A10G?

Three things this surfaces:
  1. Does vLLM recognize the architecture (gemma-3n-class: per-layer embeddings, MatFormer)?
  2. Can it download the (gated) model?
  3. Can it actually run it on A10G — the full-attention layers have head_dim 512, the same
     ceiling that ruled out FlashAttention-2 for us. vLLM's attn backend may hit it too.

If this fails, there's no apples-to-apples head-to-head with this model. Run:
    venv/bin/modal run scripts/gpu_tests/vllm_compat_check_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    # flashinfer's sampler JIT-compiles a CUDA kernel (needs nvcc, absent from the slim image).
    # Disable it → vLLM's native torch sampler; TRITON_ATTN (auto-selected for head_dim 512)
    # compiles via Triton, no nvcc needed.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
app = modal.App("vllm-compat-check", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1800)
def run():
    import traceback
    try:
        import vllm
        print("vllm version:", vllm.__version__)
    except Exception as e:
        print("vllm import FAILED:", e)
        return

    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=MODEL, dtype="bfloat16", max_model_len=2048,
            gpu_memory_utilization=0.85, enforce_eager=True,
        )
        out = llm.generate(["The capital of France is"],
                           SamplingParams(max_tokens=16, temperature=0.0))
        print("GENERATION:", repr(out[0].outputs[0].text))
        print("VLLM_SUPPORTS_MODEL: YES")
    except Exception as e:
        traceback.print_exc()
        print("VLLM_SUPPORTS_MODEL: NO")
        print(f"REASON: {type(e).__name__}: {str(e)[:600]}")


@app.local_entrypoint()
def main():
    run.remote()
