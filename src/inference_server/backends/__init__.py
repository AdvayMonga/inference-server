"""Backend factory — returns the right backend based on config."""

from inference_server.backends.base import InferenceBackend


def create_backend(backend_name: str) -> InferenceBackend:
    """Create an inference backend by name. Add new backends here."""
    if backend_name in ("cuda", "mps", "cpu"):
        from inference_server.backends.torch_backend import TorchBackend
        return TorchBackend(device=backend_name)
    elif backend_name == "mlx":
        from inference_server.backends.mlx_backend import MLXBackend
        return MLXBackend()
    elif backend_name.startswith("custom-"):
        # custom-cuda / custom-mps / custom-cpu — uses our hand-written Gemma 4 forward.
        # No KV cache, no batched decode yet (M2 adds those).
        from inference_server.backends.custom_torch_backend import CustomTorchBackend
        device = backend_name.split("-", 1)[1]
        return CustomTorchBackend(device=device)
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Available: cuda, mps, cpu, mlx, custom-{{cuda,mps,cpu}}")
