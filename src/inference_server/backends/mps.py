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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        """Full autoregressive generation on MPS."""
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
