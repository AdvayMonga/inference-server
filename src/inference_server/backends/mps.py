"""MPS (Apple Silicon) inference backend."""

import logging
import threading
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_server.backends.base import InferenceBackend
from inference_server.kv_cache.hf_format import (
    blocks_to_dynamic_cache,
    dynamic_cache_to_per_layer_3d,
)

logger = logging.getLogger(__name__)


class MPSBackend(InferenceBackend):
    """Inference backend for Apple Silicon GPUs via Metal Performance Shaders."""

    # Thinking channel token IDs (Gemma 4)
    THINK_START = 100  # <|channel>
    THINK_END = 101    # <channel|>

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("mps")
        self._eos_ids: set[int] = set()
        self._lock = threading.Lock()

    def load_model(self, model_name: str) -> None:
        """Load model and tokenizer onto MPS device."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

        # Optional: compile the model for fused GPU kernels
        from inference_server.config import settings
        if settings.compile_model:
            logger.info("Compiling model with torch.compile (first run will be slow)...")
            self.model = torch.compile(self.model)

        eos = self.model.config.eos_token_id
        if isinstance(eos, list):
            self._eos_ids = set(eos)
        else:
            self._eos_ids = {eos}

    def generate(self, token_ids: list[int], max_tokens: int,
                  template_prefix_len: int = 0) -> list[int]:
        """Two-phase generation: prefill (cacheable) then decode (unique per response)."""
        with self._lock:
            try:
                kv_cache, next_token_id = self._prefill_with_cache(token_ids)

                # --- Phase 2: Decode ---
                visible = []
                in_thinking = False

                with torch.no_grad():
                    for _ in range(max_tokens * 4):
                        if next_token_id in self._eos_ids:
                            break

                        if next_token_id == self.THINK_START:
                            in_thinking = True
                        elif next_token_id == self.THINK_END:
                            in_thinking = False
                        elif not in_thinking:
                            visible.append(next_token_id)
                            if len(visible) >= max_tokens:
                                break

                        input_tensor = torch.tensor([[next_token_id]], device=self.device)
                        outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                        kv_cache = outputs.past_key_values
                        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

                return visible
            finally:
                if self.cache_adapter is not None:
                    self.cache_adapter.release(token_ids)

    def _prefill_with_cache(self, token_ids: list[int]) -> tuple[object, int]:
        """Run prefill using CacheManager prefix lookup. Returns (kv_cache, first_token_id)."""
        cache = self.cache_adapter
        matched = 0
        kv_cache = None

        if cache is not None:
            lookup_matched, prefix_blocks = cache.lookup(token_ids)
            if lookup_matched > 0:
                kv_cache, matched = blocks_to_dynamic_cache(prefix_blocks)
                logger.debug(f"Prefix cache hit: {matched}/{len(token_ids)} tokens")

        with torch.no_grad():
            if matched == len(token_ids):
                # Full hit — run last token forward to prime decode
                input_tensor = torch.tensor([[token_ids[-1]]], device=self.device)
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
            elif matched > 0:
                # Partial hit — prefill the suffix only
                suffix = token_ids[matched:]
                input_tensor = torch.tensor([suffix], device=self.device)
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
            else:
                # Cold prefill
                input_tensor = torch.tensor([token_ids], device=self.device)
                outputs = self.model(input_tensor, use_cache=True)
                kv_cache = outputs.past_key_values

            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

        # Store the new portion into the cache
        if cache is not None and matched < len(token_ids):
            per_layer_3d = dynamic_cache_to_per_layer_3d(kv_cache)
            cache.store(token_ids, per_layer_3d, skip_tokens=matched)

        return kv_cache, next_token_id

    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        """Batched autoregressive generation with padding and per-request EOS tracking."""
        with self._lock:
            return self._generate_batch_impl(batch_token_ids, max_tokens)

    def _generate_batch_impl(
        self, batch_token_ids: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        """Inner batch generation (called under lock)."""
        batch_size = len(batch_token_ids)
        max_prompt_len = max(len(ids) for ids in batch_token_ids)
        pad_id = self.tokenizer.pad_token_id
        max_gen = max(max_tokens)

        padded = []
        for ids in batch_token_ids:
            padding = [pad_id] * (max_prompt_len - len(ids))
            padded.append(padding + ids)

        input_ids = torch.tensor(padded, device=self.device)
        attention_mask = (input_ids != pad_id).long()

        generated = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        kv_cache = None

        with torch.no_grad():
            for step in range(max_gen):
                outputs = self.model(
                    input_ids, attention_mask=attention_mask,
                    past_key_values=kv_cache, use_cache=True,
                )
                kv_cache = outputs.past_key_values
                next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)

                for i in range(batch_size):
                    if finished[i]:
                        continue
                    tok = next_tokens[i].item()
                    if tok in self._eos_ids or len(generated[i]) >= max_tokens[i]:
                        finished[i] = True
                    else:
                        generated[i].append(tok)

                if all(finished):
                    break

                input_ids = next_tokens.unsqueeze(1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, 1, device=self.device, dtype=torch.long),
                ], dim=1)

        return generated

    def generate_step(
        self, token_ids: list[int], kv_cache: object | None = None
    ) -> tuple[int, object]:
        """Single generation step on MPS."""
        input_tensor = torch.tensor([token_ids], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)

        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
        return next_token_id, outputs.past_key_values

    def stream(self, token_ids: list[int], max_tokens: int,
                template_prefix_len: int = 0) -> Generator[int, None, None]:
        """Two-phase streaming: prefill (cacheable) then decode (yields tokens)."""
        with self._lock:
            try:
                kv_cache, next_token_id = self._prefill_with_cache(token_ids)

                # --- Phase 2: Decode ---
                visible_count = 0
                in_thinking = False

                with torch.no_grad():
                    for _ in range(max_tokens * 4):
                        if next_token_id in self._eos_ids:
                            break

                        if next_token_id == self.THINK_START:
                            in_thinking = True
                        elif next_token_id == self.THINK_END:
                            in_thinking = False
                        elif not in_thinking:
                            yield next_token_id
                            visible_count += 1
                            if visible_count >= max_tokens:
                                break

                        input_tensor = torch.tensor([[next_token_id]], device=self.device)
                        outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                        kv_cache = outputs.past_key_values
                        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
            finally:
                if self.cache_adapter is not None:
                    self.cache_adapter.release(token_ids)

