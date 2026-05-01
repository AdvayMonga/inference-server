"""Abstract base class defining the contract all inference backends must follow."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from inference_server.kv_cache.cache_manager import CacheManager


class InferenceBackend(ABC):
    """Interface for model backends. Server code only talks through this."""

    cache_adapter: "CacheManager | None" = None
    last_cache_hit_tokens: int = 0  # tokens served from cache on the most recent request

    def set_cache_adapter(self, adapter: "CacheManager") -> None:
        """Attach a CacheManager for prefix caching. Optional."""
        self.cache_adapter = adapter

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        """Load model weights onto the target device. Called once at startup."""
        ...

    @abstractmethod
    def generate(self, token_ids: list[int], max_tokens: int,
                  template_prefix_len: int = 0,
                  session_id: str = "default") -> list[int]:
        """Run full autoregressive generation, return all generated token IDs."""
        ...

    @abstractmethod
    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int],
        session_ids: list[str] | None = None,
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
    def stream(self, token_ids: list[int], max_tokens: int,
                template_prefix_len: int = 0,
                session_id: str = "default") -> Generator[int, None, None]:
        """Yield token IDs one at a time as they are generated."""
        ...

    # --- Continuous-batching primitives (used by ContinuousBatchScheduler) ---
    # Backends that cannot expose these (e.g. MLX via mlx_lm) raise NotImplementedError.

    def prefill(
        self, token_ids: list[int], session_id: str = "default"
    ) -> tuple[object, int, int]:
        """Prefill one request. Returns (per_row_kv_cache, first_token_id, kv_length)."""
        raise NotImplementedError(f"{type(self).__name__} does not support prefill()")

    def decode_step_batched(
        self,
        current_tokens: object,    # tensor [B, 1]
        batched_kv: object,
        attention_mask: object,    # tensor [B, S]
        position_ids: object,      # tensor [B, 1]
    ) -> tuple[object, object]:
        """One decode step. Returns (next_tokens [B], updated batched_kv)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support decode_step_batched()"
        )

    def stack_caches_left_padded(
        self, per_row_caches: list, max_kv_len: int
    ) -> object:
        """Left-pad and stack per-row KV caches into a batched cache."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support stack_caches_left_padded()"
        )

    def remove_row_from_cache(self, batched_kv: object, row_idx: int) -> object:
        """Drop row `row_idx` from the batched cache."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support remove_row_from_cache()"
        )

    def splice_into_batched(
        self, batched_kv: object | None, new_kv: object, new_kv_len: int,
    ) -> object:
        """Append a new row's KV to a batched cache, left-padding to align seq lengths.

        Returns the updated batched cache. If `batched_kv` is None, returns a batched
        cache containing just the new row.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support splice_into_batched()"
        )

    def kv_length(self, kv: object) -> int:
        """Return the current sequence length of a KV cache."""
        raise NotImplementedError(f"{type(self).__name__} does not support kv_length()")

    def is_eos(self, token_id: int) -> bool:
        """True if `token_id` is an end-of-sequence token."""
        raise NotImplementedError(f"{type(self).__name__} does not support is_eos()")

    @property
    def device_str(self) -> str:
        """Device identifier for tensor creation by the scheduler."""
        raise NotImplementedError(f"{type(self).__name__} does not expose device_str")
