"""Backend factory — returns the right backend based on config."""

from inference_server.backends.base import InferenceBackend


def create_backend(backend_name: str) -> InferenceBackend:
    """Create an inference backend by name. Add new backends here."""
    if backend_name == "mps":
        from inference_server.backends.mps import MPSBackend
        return MPSBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Available: mps")
