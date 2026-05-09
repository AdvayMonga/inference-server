"""PyTorch-based inference backend. Device-agnostic: cuda, mps, or cpu."""

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


class TorchBackend(InferenceBackend):
    """HuggingFace + PyTorch backend. Works on cuda, mps, or cpu."""

    # Thinking channel token IDs (Gemma 4)
    THINK_START = 100  # <|channel>
    THINK_END = 101    # <channel|>

    def __init__(self, device: str = "cuda"):
        self.model = None
        self.tokenizer = None
        self.device = torch.device(device)
        self._eos_ids: set[int] = set()
        self._lock = threading.Lock()

    def load_model(self, model_name: str) -> None:
        """Load model and tokenizer onto the configured device."""
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
                  template_prefix_len: int = 0,
                  session_id: str = "default") -> list[int]:
        """Two-phase generation: prefill (cacheable) then decode (unique per response)."""
        with self._lock:
            try:
                kv_cache, next_token_id = self._prefill_with_cache(token_ids, session_id)

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
                    self.cache_adapter.release(token_ids, session_id=session_id)

    def _prefill_with_cache(self, token_ids: list[int],
                             session_id: str = "default") -> tuple[object, int]:
        """Run prefill using CacheManager prefix lookup. Returns (kv_cache, first_token_id)."""
        cache = self.cache_adapter
        matched = 0
        kv_cache = None

        if cache is not None:
            lookup_matched, prefix_blocks = cache.lookup(token_ids, session_id=session_id)
            if lookup_matched > 0:
                kv_cache, matched = blocks_to_dynamic_cache(prefix_blocks)
                logger.debug(f"Prefix cache hit: {matched}/{len(token_ids)} tokens")

        self.last_cache_hit_tokens = matched

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
            cache.store(token_ids, per_layer_3d, skip_tokens=matched, session_id=session_id)

        return kv_cache, next_token_id

    def generate_batch(
        self, batch_token_ids: list[list[int]], max_tokens: list[int],
        session_ids: list[str] | None = None,
    ) -> list[list[int]]:
        """Batched autoregressive generation with per-row prefix caching."""
        with self._lock:
            return self._generate_batch_impl(batch_token_ids, max_tokens, session_ids)

    def _generate_batch_impl(
        self, batch_token_ids: list[list[int]], max_tokens: list[int],
        session_ids: list[str] | None = None,
    ) -> list[list[int]]:
        """Per-row prefill (with cache), batched decode."""
        from transformers.cache_utils import DynamicCache

        batch_size = len(batch_token_ids)
        if session_ids is None:
            session_ids = ["default"] * batch_size
        max_gen = max(max_tokens)

        # Phase 1: per-row prefill (sequential — each row hits cache independently)
        per_row_caches: list[DynamicCache] = []
        first_tokens: list[int] = []
        real_kv_lens: list[int] = []

        try:
            for ids, sid in zip(batch_token_ids, session_ids):
                kv_i, tok_i = self._prefill_with_cache(ids, sid)
                per_row_caches.append(kv_i)
                first_tokens.append(tok_i)
                real_kv_lens.append(kv_i.layers[0].keys.shape[2])

            max_kv_len = max(real_kv_lens)

            # Phase 2: stack per-row caches into a batched DynamicCache (left-pad along seq)
            batched_cache = self._stack_caches_left_padded(per_row_caches, max_kv_len)

            # Initial attention mask: 0 over left-pad, 1 over real KV
            attention_mask = torch.zeros(
                batch_size, max_kv_len, device=self.device, dtype=torch.long
            )
            for i, L in enumerate(real_kv_lens):
                attention_mask[i, max_kv_len - L:] = 1

            generated: list[list[int]] = [[] for _ in range(batch_size)]
            finished: list[bool] = [False] * batch_size
            current_tokens = torch.tensor(first_tokens, device=self.device).unsqueeze(1)

            # Phase 3: batched decode with explicit per-row position_ids
            with torch.no_grad():
                for step in range(max_gen):
                    for i in range(batch_size):
                        if finished[i]:
                            continue
                        tok = current_tokens[i, 0].item()
                        if tok in self._eos_ids or len(generated[i]) >= max_tokens[i]:
                            finished[i] = True
                        else:
                            generated[i].append(tok)

                    if all(finished):
                        break

                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(batch_size, 1, device=self.device, dtype=torch.long),
                    ], dim=1)
                    position_ids = torch.tensor(
                        [[real_kv_lens[i] + step] for i in range(batch_size)],
                        device=self.device, dtype=torch.long,
                    )

                    outputs = self.model(
                        current_tokens,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=batched_cache,
                        use_cache=True,
                    )
                    batched_cache = outputs.past_key_values
                    current_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            return generated
        finally:
            if self.cache_adapter is not None:
                for ids, sid in zip(batch_token_ids, session_ids):
                    self.cache_adapter.release(ids, session_id=sid)

    def _stack_caches_left_padded(self, per_row_caches: list, max_kv_len: int):
        """Build a batched DynamicCache by left-padding each row's KV along seq dim."""
        from transformers.cache_utils import DynamicCache

        if not per_row_caches:
            return None
        num_layers = len(per_row_caches[0].layers)
        batched = DynamicCache()
        for layer in range(num_layers):
            ks, vs = [], []
            for cache in per_row_caches:
                k = cache.layers[layer].keys     # [1, H, L, D]
                v = cache.layers[layer].values
                pad_len = max_kv_len - k.shape[2]
                if pad_len > 0:
                    pad_shape = (k.shape[0], k.shape[1], pad_len, k.shape[3])
                    k = torch.cat([torch.zeros(pad_shape, device=k.device, dtype=k.dtype), k], dim=2)
                    v = torch.cat([torch.zeros(pad_shape, device=v.device, dtype=v.dtype), v], dim=2)
                ks.append(k)
                vs.append(v)
            batched.update(torch.cat(ks, dim=0), torch.cat(vs, dim=0), layer)
        return batched

    # --- Continuous-batching primitives ---

    def prefill(
        self, token_ids: list[int], session_id: str = "default"
    ) -> tuple[object, int, int]:
        """Prefill one request through the cache. Caller holds the model lock."""
        kv, first_token = self._prefill_with_cache(token_ids, session_id)
        return kv, first_token, kv.layers[0].keys.shape[2]

    def prefill_lookup(
        self, token_ids: list[int], session_id: str = "default"
    ) -> tuple[object | None, int]:
        """Cache lookup only — no forward pass. Returns (partial_kv, matched)."""
        cache = self.cache_adapter
        matched = 0
        partial_kv = None
        if cache is not None:
            m, prefix_blocks = cache.lookup(token_ids, session_id=session_id)
            if m > 0:
                partial_kv, matched = blocks_to_dynamic_cache(prefix_blocks)
        self.last_cache_hit_tokens = matched
        return partial_kv, matched

    def prefill_chunk(
        self, chunk_token_ids: list[int], partial_kv: object | None
    ) -> tuple[object, int, int]:
        """Forward pass on chunk_token_ids against partial_kv. Returns (kv, last_argmax, kv_len)."""
        with torch.no_grad():
            input_tensor = torch.tensor([chunk_token_ids], device=self.device)
            outputs = self.model(input_tensor, past_key_values=partial_kv, use_cache=True)
            kv = outputs.past_key_values
            last_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        return kv, last_token, kv.layers[0].keys.shape[2]

    def prefill_store(
        self, token_ids: list[int], full_kv: object, matched: int,
        session_id: str = "default",
    ) -> None:
        """Store the uncached portion of full_kv into the cache."""
        cache = self.cache_adapter
        if cache is None or matched >= len(token_ids):
            return
        per_layer_3d = dynamic_cache_to_per_layer_3d(full_kv)
        cache.store(token_ids, per_layer_3d, skip_tokens=matched, session_id=session_id)

    def decode_step_batched(
        self,
        current_tokens: torch.Tensor,
        batched_kv: object,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        """One forward pass over the running batch."""
        with torch.no_grad():
            outputs = self.model(
                current_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=batched_kv,
                use_cache=True,
            )
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        return next_tokens, outputs.past_key_values

    def stack_caches_left_padded(self, per_row_caches: list, max_kv_len: int) -> object:
        return self._stack_caches_left_padded(per_row_caches, max_kv_len)

    def remove_row_from_cache(self, batched_kv: object, row_idx: int) -> object:
        """Drop one row from a batched DynamicCache along the batch dim."""
        from transformers.cache_utils import DynamicCache
        new_cache = DynamicCache()
        for layer_idx, layer in enumerate(batched_kv.layers):
            k = torch.cat([layer.keys[:row_idx], layer.keys[row_idx + 1:]], dim=0)
            v = torch.cat([layer.values[:row_idx], layer.values[row_idx + 1:]], dim=0)
            new_cache.update(k, v, layer_idx)
        return new_cache

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        """Append a new row's KV with left-padding so all rows align."""
        from transformers.cache_utils import DynamicCache
        if batched_kv is None:
            return self._stack_caches_left_padded([new_kv], new_kv_len)

        existing_len = self.kv_length(batched_kv)
        max_len = max(existing_len, new_kv_len)
        ex_pad = max_len - existing_len
        new_pad = max_len - new_kv_len

        out = DynamicCache()
        for li, ex_layer in enumerate(batched_kv.layers):
            ek, ev = ex_layer.keys, ex_layer.values
            if ex_pad > 0:
                shp = (ek.shape[0], ek.shape[1], ex_pad, ek.shape[3])
                ek = torch.cat([torch.zeros(shp, device=ek.device, dtype=ek.dtype), ek], dim=2)
                ev = torch.cat([torch.zeros(shp, device=ev.device, dtype=ev.dtype), ev], dim=2)
            nk = new_kv.layers[li].keys
            nv = new_kv.layers[li].values
            if new_pad > 0:
                shp = (nk.shape[0], nk.shape[1], new_pad, nk.shape[3])
                nk = torch.cat([torch.zeros(shp, device=nk.device, dtype=nk.dtype), nk], dim=2)
                nv = torch.cat([torch.zeros(shp, device=nv.device, dtype=nv.dtype), nv], dim=2)
            out.update(torch.cat([ek, nk], dim=0), torch.cat([ev, nv], dim=0), li)
        return out

    def kv_length(self, kv: object) -> int:
        return kv.layers[0].keys.shape[2]

    def is_eos(self, token_id: int) -> bool:
        return token_id in self._eos_ids

    @property
    def device_str(self) -> str:
        return str(self.device)

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
                template_prefix_len: int = 0,
                session_id: str = "default") -> Generator[int, None, None]:
        """Two-phase streaming: prefill (cacheable) then decode (yields tokens)."""
        with self._lock:
            try:
                kv_cache, next_token_id = self._prefill_with_cache(token_ids, session_id)

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
                    self.cache_adapter.release(token_ids, session_id=session_id)

