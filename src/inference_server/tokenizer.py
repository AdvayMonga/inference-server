"""Tokenization pipeline — text to token IDs and back, with validation."""

from transformers import AutoTokenizer


class Tokenizer:
    """Wraps HuggingFace tokenizer with validation and context window enforcement."""

    def __init__(self, model_name: str, context_window: int):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._context_window = context_window

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

    def encode_chat(self, text: str) -> list[int]:
        """Wrap text in the chat template and encode. Used for instruction-tuned models."""
        if not text:
            raise ValueError("Input text is empty")

        messages = [{"role": "user", "content": text}]
        inputs = self._tokenizer.apply_chat_template(
            messages, return_dict=True, return_tensors=None, add_generation_prompt=True
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

    @property
    def eos_token_id(self) -> int:
        """The end-of-sequence token ID for this model."""
        return self._tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size
