"""MPS (Apple Silicon) inference backend."""

from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_server.backends.base import InferenceBackend


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
        """Full autoregressive generation for a single request."""
        input_tensor = torch.tensor([token_ids], device=self.device)
        generated = []
        kv_cache = None

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                generated.append(next_token_id)
                input_tensor = torch.tensor([[next_token_id]], device=self.device)

        return generated

    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        """Batched autoregressive generation with padding and per-request EOS tracking."""
        batch_size = len(batch_token_ids)
        max_prompt_len = max(len(ids) for ids in batch_token_ids)
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        max_gen = max(max_tokens)

        # Left-pad prompts so they all end at the same position
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

                # Next input is just the new tokens (KV cache has the history)
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
        """Yield tokens one at a time from MPS."""
        input_tensor = torch.tensor([token_ids], device=self.device)
        kv_cache = None

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_tensor, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                yield next_token_id
                input_tensor = torch.tensor([[next_token_id]], device=self.device)
