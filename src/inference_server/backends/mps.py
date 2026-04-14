"""MPS (Apple Silicon) inference backend."""

import logging
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_server.backends.base import InferenceBackend

logger = logging.getLogger(__name__)


class MPSBackend(InferenceBackend):
    """Inference backend for Apple Silicon GPUs via Metal Performance Shaders."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("mps")

    def load_model(self, model_name: str) -> None:
        """Load model and tokenizer onto MPS device."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        """Full autoregressive generation with optional prefix cache reuse."""
        skip_tokens = 0
        kv_cache = None

        # Try to reuse cached KV state for the prompt prefix
        if self.cache_manager is not None:
            skip_tokens, cached_blocks = self.cache_manager.lookup(token_ids)
            if skip_tokens > 0:
                kv_cache = self.cache_manager.build_kv_from_blocks(cached_blocks)
                logger.debug(f"Reusing {skip_tokens} cached tokens, computing {len(token_ids) - skip_tokens} new")

        # Prefill: process uncached prompt tokens
        uncached_ids = token_ids[skip_tokens:]
        if uncached_ids:
            input_tensor = torch.tensor([uncached_ids], device=self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
        else:
            # Entire prompt was cached — run one step to get the first generated token
            input_tensor = torch.tensor([[token_ids[-1]]], device=self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

        generated = []
        if next_token_id == self.tokenizer.eos_token_id:
            self._cache_and_release(token_ids, kv_cache, skip_tokens)
            return generated

        generated.append(next_token_id)

        # Decode: generate remaining tokens one at a time
        with torch.no_grad():
            for _ in range(max_tokens - 1):
                input_tensor = torch.tensor([[next_token_id]], device=self.device)
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                generated.append(next_token_id)

        # Store the prompt's KV state in cache for future requests
        self._cache_and_release(token_ids, kv_cache, skip_tokens)

        return generated

    def _cache_and_release(self, token_ids: list[int], kv_cache: object, skip_tokens: int) -> None:
        """Store prompt KV state in cache and release blocks."""
        if self.cache_manager is None:
            return

        # Extract KV tensors from HuggingFace format: tuple of (K, V) per layer
        if kv_cache is not None:
            kv_tensors = [(k.squeeze(0), v.squeeze(0)) for k, v in kv_cache]
            self.cache_manager.store(token_ids, kv_tensors, skip_tokens=skip_tokens)

        # Release the lookup ref we acquired
        if skip_tokens > 0:
            self.cache_manager.release(token_ids)

    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        """Batched autoregressive generation with padding and per-request EOS tracking."""
        batch_size = len(batch_token_ids)
        max_prompt_len = max(len(ids) for ids in batch_token_ids)
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
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
                    if tok == eos_id or len(generated[i]) >= max_tokens[i]:
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

    def stream(self, token_ids: list[int], max_tokens: int) -> Generator[int, None, None]:
        """Yield tokens one at a time with optional prefix cache reuse."""
        skip_tokens = 0
        kv_cache = None

        if self.cache_manager is not None:
            skip_tokens, cached_blocks = self.cache_manager.lookup(token_ids)
            if skip_tokens > 0:
                kv_cache = self.cache_manager.build_kv_from_blocks(cached_blocks)

        uncached_ids = token_ids[skip_tokens:]
        if uncached_ids:
            input_tensor = torch.tensor([uncached_ids], device=self.device)
        else:
            input_tensor = torch.tensor([[token_ids[-1]]], device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
            kv_cache = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

            if next_token_id != self.tokenizer.eos_token_id:
                yield next_token_id

                for _ in range(max_tokens - 1):
                    input_tensor = torch.tensor([[next_token_id]], device=self.device)
                    outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                    kv_cache = outputs.past_key_values
                    next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

                    if next_token_id == self.tokenizer.eos_token_id:
                        break

                    yield next_token_id

        self._cache_and_release(token_ids, kv_cache, skip_tokens)
