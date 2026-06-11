"""Custom-forward Torch backend (M1) — uses our hand-written Gemma 4 model.

What works: single-prompt non-batched non-cached `generate()` / `stream()`. Proves
the custom forward serves real tokens end-to-end.

What's deliberately NOT here yet (M2 builds these on top of paged KV):
  - past_key_values / DynamicCache plumbing
  - decode_step_batched, prefill, prefill_chunk
  - prefix-cache lookup/store

Scheduler is restricted to the legacy generate/stream path when this backend is active.
"""

from __future__ import annotations

import logging
import threading
from typing import Generator

import torch

from inference_server.backends.base import InferenceBackend
from inference_server.sampling import GREEDY, SamplingParams, sample

logger = logging.getLogger(__name__)


class CustomTorchBackend(InferenceBackend):
    """PyTorch backend driven by our custom GemmaForCausalLM forward."""

    THINK_START = 100
    THINK_END = 101

    def __init__(self, device: str = "cuda", model_name: str = "google/gemma-4-E2B-it"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._eos_ids: set[int] = set()
        self._lock = threading.Lock()
        self.cache_adapter = None
        self.last_cache_hit_tokens = 0

    def load_model(self, model_name: str) -> None:
        from transformers import AutoTokenizer
        from inference_server.models.gemma4 import GemmaForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GemmaForCausalLM.from_hf(model_name, dtype=torch.bfloat16).to(self.device).eval()
        eos = self.tokenizer.eos_token_id
        self._eos_ids = {eos} if isinstance(eos, int) else set(eos)

    def set_cache_adapter(self, adapter) -> None:
        self.cache_adapter = adapter  # not used yet — M2

    # ---- Required interface ----

    @torch.no_grad()
    def generate(
        self, token_ids: list[int], max_tokens: int,
        template_prefix_len: int = 0,
        session_id: str = "default",
        sampling: SamplingParams = GREEDY,
    ) -> list[int]:
        """Autoregressive generation without KV cache — re-runs full forward each step.
        Slow but correct; M2 wires up paged KV.
        """
        with self._lock:
            ids = list(token_ids)
            visible: list[int] = []
            in_thinking = False

            for _ in range(max_tokens * 4):
                input_tensor = torch.tensor([ids], device=self.device)
                logits = self.model(input_tensor)
                tok = int(sample(logits[:, -1, :], sampling).item())
                ids.append(tok)

                if tok in self._eos_ids:
                    break
                if tok == self.THINK_START:
                    in_thinking = True
                elif tok == self.THINK_END:
                    in_thinking = False
                elif not in_thinking:
                    visible.append(tok)
                    if len(visible) >= max_tokens:
                        break
            return visible

    @torch.no_grad()
    def stream(
        self, token_ids: list[int], max_tokens: int,
        template_prefix_len: int = 0,
        session_id: str = "default",
        sampling: SamplingParams = GREEDY,
    ) -> Generator[int, None, None]:
        with self._lock:
            ids = list(token_ids)
            visible_count = 0
            in_thinking = False

            for _ in range(max_tokens * 4):
                input_tensor = torch.tensor([ids], device=self.device)
                logits = self.model(input_tensor)
                tok = int(sample(logits[:, -1, :], sampling).item())
                ids.append(tok)

                if tok in self._eos_ids:
                    break
                if tok == self.THINK_START:
                    in_thinking = True
                elif tok == self.THINK_END:
                    in_thinking = False
                elif not in_thinking:
                    yield tok
                    visible_count += 1
                    if visible_count >= max_tokens:
                        break

    @torch.no_grad()
    def generate_step(
        self, token_ids: list[int], kv_cache=None,
        sampling: SamplingParams = GREEDY,
    ) -> tuple[int, object]:
        # No KV cache yet — ignore the kv_cache argument (M2)
        input_tensor = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_tensor)
        tok = int(sample(logits[:, -1, :], sampling).item())
        return tok, None

    def generate_batch(self, batch_token_ids, max_tokens, session_ids=None):
        # Naive — loop generate() per row; concurrent batching belongs in M2.
        out = []
        for ids, mt in zip(batch_token_ids, max_tokens):
            out.append(self.generate(ids, mt, session_id=(session_ids or ["default"])[len(out)]))
        return out

    def is_eos(self, token_id: int) -> bool:
        return token_id in self._eos_ids

    @property
    def device_str(self) -> str:
        return str(self.device)
