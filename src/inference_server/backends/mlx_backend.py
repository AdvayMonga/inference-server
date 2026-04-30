"""MLX (Apple Silicon native) inference backend."""

import logging
from typing import Generator

import mlx.core as mx
import mlx_lm

from inference_server.backends.base import InferenceBackend

logger = logging.getLogger(__name__)

# Thinking channel token IDs (Gemma 4)
THINK_START = 100  # <|channel>
THINK_END = 101    # <channel|>


class MLXBackend(InferenceBackend):
    """Inference backend using Apple's MLX framework."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._eos_ids: set[int] = set()

    def load_model(self, model_name: str) -> None:
        """Load model via mlx-lm."""
        self.model, self.tokenizer = mlx_lm.load(model_name)

        # Get EOS tokens from tokenizer (MLX model doesn't have .config like HF)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, list):
            self._eos_ids = set(eos)
        elif eos is not None:
            self._eos_ids = {eos, 106}  # include <turn|> for Gemma 4
        else:
            self._eos_ids = {1, 106}

    def generate(self, token_ids: list[int], max_tokens: int,
                  template_prefix_len: int = 0,
                  session_id: str = "default") -> list[int]:
        """Full generation with thinking filter."""
        prompt = self.tokenizer.decode(token_ids)
        visible = []
        in_thinking = False

        for token in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens * 4):
            token_id = token.token

            if token_id in self._eos_ids:
                break

            if token_id == THINK_START:
                in_thinking = True
                continue
            elif token_id == THINK_END:
                in_thinking = False
                continue

            if not in_thinking:
                visible.append(token_id)
                if len(visible) >= max_tokens:
                    break

        return visible

    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int],
        session_ids: list[str] | None = None,
    ) -> list[list[int]]:
        """Batch generation — runs sequentially for MLX (single-user optimized)."""
        sids = session_ids if session_ids is not None else ["default"] * len(batch_token_ids)
        return [
            self.generate(ids, mt, session_id=sid)
            for ids, mt, sid in zip(batch_token_ids, max_tokens, sids)
        ]

    def generate_step(
        self, token_ids: list[int], kv_cache: object | None = None
    ) -> tuple[int, object]:
        """Single step — not used directly with MLX."""
        prompt = self.tokenizer.decode(token_ids)
        for token in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=1):
            return token.token, None
        return 0, None

    def stream(self, token_ids: list[int], max_tokens: int,
                template_prefix_len: int = 0,
                session_id: str = "default") -> Generator[int, None, None]:
        """Yield token IDs one at a time with thinking filter."""
        prompt = self.tokenizer.decode(token_ids)
        visible_count = 0
        in_thinking = False

        for token in mlx_lm.stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens * 4):
            token_id = token.token

            if token_id in self._eos_ids:
                break

            if token_id == THINK_START:
                in_thinking = True
                continue
            elif token_id == THINK_END:
                in_thinking = False
                continue

            if not in_thinking:
                yield token_id
                visible_count += 1
                if visible_count >= max_tokens:
                    break
