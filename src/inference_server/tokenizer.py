"""Tokenization pipeline — text to token IDs and back, with validation."""

import re

from transformers import AutoTokenizer


class Tokenizer:
    """Wraps HuggingFace tokenizer with validation and context window enforcement."""

    def __init__(self, model_name: str, context_window: int):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._context_window = context_window
        self._template_prefix_len = self._calc_template_prefix_len()

    def _calc_template_prefix_len(self) -> int:
        """Count shared template tokens that are identical for all prompts."""
        if not getattr(self._tokenizer, "chat_template", None):
            return 0
        try:
            ids_a = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "A"}],
                return_dict=True, return_tensors=None, add_generation_prompt=True,
            )["input_ids"]
            ids_b = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "B"}],
                return_dict=True, return_tensors=None, add_generation_prompt=True,
            )["input_ids"]
            shared = 0
            for a, b in zip(ids_a, ids_b):
                if a == b:
                    shared += 1
                else:
                    break
            return shared
        except Exception:
            return 0

    @property
    def template_prefix_len(self) -> int:
        """Number of template tokens shared by all prompts (not cacheable across prompts)."""
        return self._template_prefix_len

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs. Raises if input exceeds context window."""
        if not text:
            raise ValueError("Input text is empty")

        token_ids = self._tokenizer.encode(text, add_special_tokens=True)

        if len(token_ids) > self._context_window:
            raise ValueError(
                f"Input is {len(token_ids)} tokens, exceeds context window of {self._context_window}"
            )

        return token_ids

    def encode_chat(self, text: str, thinking: bool = True) -> list[int]:
        """Wrap text in chat template and encode. Falls back to plain encode if no template."""
        if not text:
            raise ValueError("Input text is empty")

        if not getattr(self._tokenizer, "chat_template", None):
            return self.encode(text)

        messages = [{"role": "user", "content": text}]
        inputs = self._tokenizer.apply_chat_template(
            messages, return_dict=True, return_tensors=None,
            add_generation_prompt=True, enable_thinking=thinking,
        )
        token_ids = inputs["input_ids"]

        if len(token_ids) > self._context_window:
            raise ValueError(
                f"Input is {len(token_ids)} tokens, exceeds context window of {self._context_window}"
            )

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID — used for streaming one token at a time."""
        return self._tokenizer.decode([token_id], skip_special_tokens=True)

    @staticmethod
    def strip_thinking(text: str) -> str:
        """Remove thinking blocks from model output (supports multiple formats)."""
        # Gemma 4 format: <|channel>thought\n...<channel|>
        stripped = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
        # Also handle <think>...</think> format
        stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL)
        return stripped.strip()

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token ID for this model."""
        return self._tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
