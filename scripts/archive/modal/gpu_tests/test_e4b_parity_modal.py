"""Gating check for the A100 phase: does our custom Gemma 4 forward run E4B correctly?

The custom forward is config-driven (layers/dims/sharing from HF config), so E4B *should* construct
and load — but it's unverified. Compares last-position logits (custom vs HF) on A100-80GB. Pass =
argmax agreement + high cosine (the custom forward was byte-matched to HF on E2B; E4B should hold).

    venv/bin/modal run scripts/gpu_tests/test_e4b_parity_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("e4b-parity", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E4B-it"


@app.function(gpu="A100-80GB", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=1200)
def run():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from inference_server.models.gemma4 import GemmaForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")

    try:
        custom = GemmaForCausalLM.from_hf(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"E4B CONSTRUCT/LOAD FAILED: {type(e).__name__}: {str(e)[:400]}")
        return
    print(f"custom E4B built: {custom.model.num_layers} layers, hidden={custom.model.hidden_size}")
    with torch.no_grad():
        cl = custom(ids)[:, -1, :].float()
    del custom
    torch.cuda.empty_cache()

    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    with torch.no_grad():
        hl = hf(ids).logits[:, -1, :].float()

    diff = (cl - hl).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(cl, hl).item()
    ok = cos > 0.999 and int(cl.argmax()) == int(hl.argmax())
    print(f"E4B parity: max|diff|={diff:.4f}  cos={cos:.6f}  argmax_eq={int(cl.argmax()) == int(hl.argmax())}")
    print("E4B ON CUSTOM FORWARD", "OK" if ok else "MISMATCH — needs investigation")


@app.local_entrypoint()
def main():
    run.remote()
