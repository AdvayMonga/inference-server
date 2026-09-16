"""Isolated GPU parity test for the Triton paged-decode kernel (M2.3).

Validates the kernel against a pure-torch reference on synthetic data — no model — so
kernel correctness is proven before it's wired into the forward. The checks themselves live
in `checks.py`, shared with `cuda_gate.py` (the RunPod instrument); this file is the Modal
launcher. Run:

    venv/bin/modal run scripts/gpu_tests/test_paged_kernel_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server", "checks")
)

app = modal.App("paged-kernel-test", image=image)

DECODE_CHECKS = ("paged_decode_parity", "paged_decode_window", "paged_decode_no_recompile")


@app.function(gpu="A10G")
def run():
    import checks

    ok = True
    for name in DECODE_CHECKS:
        passed, detail = checks.CHECKS[name]()
        print(name, "OK" if passed else "FAIL", detail)
        ok = ok and passed
    return ok


@app.local_entrypoint()
def main():
    print("kernel parity:", run.remote())
