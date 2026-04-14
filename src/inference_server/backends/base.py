"""Abstract base class defining the contract all inference backends must follow."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from inference_server.kv_cache.cache_manager import CacheManager


class InferenceBackend(ABC):
    """Interface for model backends. Server code only talks through this."""

    cache_manager: "CacheManager | None" = None

    def set_cache_manager(self, cache_manager: "CacheManager") -> None:
        """Attach a cache manager for prefix caching. Optional."""
        self.cache_manager = cache_manager

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load model weights onto the target device. Called once at startup."""
        ...

    @abstractmethod
    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        """Run full autoregressive generation, return all generated token IDs."""
        ...

    @abstractmethod
    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        """Run batched autoregressive generation for multiple requests at once."""
        ...

    @abstractmethod
    def generate_step(
        self, token_ids: list[int], kv_cache: object | None = None
    ) -> tuple[int, object]:
        """Run a single generation step. Return (next_token_id, updated_kv_cache)."""
        ...

    @abstractmethod
    def stream(self, token_ids: list[int], max_tokens: int) -> Generator[int, None, None]:
        """Yield token IDs one at a time as they are generated."""
        ...
