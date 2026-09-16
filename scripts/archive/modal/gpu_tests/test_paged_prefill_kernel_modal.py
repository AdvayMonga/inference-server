"""Isolated correctness test for the paged PREFILL kernel (multi-query, no gather).

Builds random per-sequence KV in scattered blocks (varied prefix/suffix lengths + GQA), runs
paged_prefill_attention, and compares against a gather + per-query causal-softmax reference.
The check lives in `checks.py`, shared with `cuda_gate.py`; this file is the Modal launcher.

    venv/bin/modal run scripts/gpu_tests/test_paged_prefill_kernel_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server", "checks")
)
app = modal.App("paged-prefill-kernel-test", image=image)


@app.function(gpu="A10G", timeout=600)
def run():
    import checks

    passed, detail = checks.paged_prefill_parity()
    print("PAGED PREFILL KERNEL", "OK" if passed else "FAIL", detail)
    return passed


@app.local_entrypoint()
def main():
    print("paged prefill:", run.remote())
