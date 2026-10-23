"""Backend factory — returns the right backend based on config."""

from inference_server.backends.base import InferenceBackend


def create_backend(backend_name: str) -> InferenceBackend:
    """Create an inference backend by name: custom-cuda, custom-mps or custom-cpu."""
    if backend_name.startswith("custom-"):
        # The hand-written Qwen3-MoE forward with paged KV + prefix sharing, scheduler-driven.
        from inference_server.backends.custom_torch_backend import CustomTorchBackend
        device = backend_name.split("-", 1)[1]
        return CustomTorchBackend(device=device)
    raise ValueError(f"Unknown backend: {backend_name}. Available: custom-{{cuda,mps,cpu}}")
