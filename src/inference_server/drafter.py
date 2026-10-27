"""A draft model for speculative decoding (SPEC_DRAFT_MODEL, e.g. Qwen/Qwen3-0.6B: the target's tokenizer): it proposes
the next tokens greedily and the target verifies them in one forward (scheduler._decode_step_spec), so the output is
the target's greedy decode exactly. One model and one static KV cache, shared: one request's context is in the cache at
a time, and a request whose context was displaced is prefilled again."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class DraftModel:
    """The draft model with a static KV cache of `max_len` positions; one-token steps replay a CUDA graph on CUDA."""

    def __init__(self, model, max_len: int):
        from transformers import StaticCache
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.max_len = max_len
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.owner = None                                       # the ModelDrafter whose context the cache holds
        self._tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self._graph = self._out = None
        self._fwd()                                             # allocates the cache's layers (position 0, overwritten)
        if self.device.type == "cuda":
            try:
                self._capture()
            except Exception:  # noqa: BLE001 - the eager step is the same computation
                logger.exception("draft model graph capture failed; drafting eagerly")
                self._graph = None

    @classmethod
    def load(cls, name: str, device, max_len: int, dtype=torch.bfloat16) -> "DraftModel":
        from transformers import AutoModelForCausalLM
        return cls(AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device), max_len)

    def _at(self, pos: torch.Tensor) -> None:
        """Every layer writes its next K/V at `pos`: a static cache layer writes at its own running count
        (`cumulative_length`), not at the cache_position passed in, so a position rewritten after rejected drafts
        would land elsewhere (transformers 5.x)."""
        for layer in self.cache.layers:
            if layer.is_initialized:
                layer.cumulative_length.copy_(pos.reshape(layer.cumulative_length.shape))

    @torch.no_grad()
    def _fwd(self) -> torch.Tensor:
        self._at(self._pos)
        out = self.model(self._tok, past_key_values=self.cache, cache_position=self._pos, use_cache=True)
        return out.logits[0, -1].argmax()

    def _capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._fwd()                                     # position 0 of an empty cache: overwritten later
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._out = self._fwd()
        self._graph = g

    @torch.no_grad()
    def prefill(self, ids: list[int], start: int) -> None:
        """Write the K/V of `ids` at positions start.. (eager, one forward)."""
        if not ids:
            return
        t = torch.tensor([ids], dtype=torch.long, device=self.device)
        pos = torch.arange(start, start + len(ids), device=self.device)
        self._at(pos[:1])
        self.model(t, past_key_values=self.cache, cache_position=pos, use_cache=True)

    @torch.no_grad()
    def step(self, tok: int, pos: int) -> int:
        """Feed `tok` at `pos`; the greedy next token."""
        self._tok.fill_(tok)
        self._pos.fill_(pos)
        if self._graph is not None:
            self._graph.replay()
            return int(self._out.item())
        return int(self._fwd().item())


class ModelDrafter:
    """One request's drafting state: its context `ids` (the last one not yet fed to the draft model) and how many
    leading ids have correct K/V in the shared cache. Same interface as scheduler.NgramDrafter."""

    def __init__(self, draft: DraftModel, ids: list[int]):
        self.d = draft
        self.ids = list(ids)
        self.cached = 0
        self._drafts: list[int] = []

    def propose(self, k: int) -> list[int]:
        d = self.d
        if d.owner is not self:                                 # another request's context is in the cache
            d.owner, self.cached = self, 0
        n = len(self.ids)
        if n + k > d.max_len or k <= 0:
            return []
        d.prefill(self.ids[self.cached:n - 1], self.cached)
        out, tok = [], self.ids[-1]
        for i in range(k):
            tok = d.step(tok, n - 1 + i)
            out.append(tok)
        self.cached = n                                         # ids and out[:-1] are in the cache
        self._drafts = out
        return out

    def observe(self, tokens: list[int]) -> None:
        """The tokens that follow the context (accepted drafts and the target's own token, or plain decode's)."""
        m = 0
        for a, b in zip(tokens, self._drafts[:-1]):             # drafts that were fed have K/V in the cache
            if a != b:
                break
            m += 1
        self.cached = min(self.cached, len(self.ids)) + m
        self.ids.extend(tokens)
        self._drafts = []
