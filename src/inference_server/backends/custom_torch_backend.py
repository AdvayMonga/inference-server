"""Custom-forward Torch backend — uses our hand-written Gemma 4 model.

Drives the full stack: paged KV cache + cross-session prefix sharing under both the
legacy direct `generate()` / `stream()` path and the `ContinuousBatchScheduler`.

Scheduler integration (M2.4): the continuous-batching primitives (`prefill`,
`prefill_lookup/chunk/store`, `decode_step_batched`, `splice_into_batched`,
`remove_row_from_cache`, `kv_length`) represent the batched KV as a plain
`list[PagedKVCache]` — one paged cache per active row. Decode loops the forward
once per row (row-by-row, not GPU-batched): correct, and exercises admit / prefix-hit
/ evict end-to-end. Each row's cache holds only its real tokens, so there is no
left-padding and the scheduler's `attention_mask` is ignored. Real batched decode
(single forward over all rows) is a later step.
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
        import os
        from transformers import AutoTokenizer
        from inference_server.models.gemma4 import GemmaForCausalLM
        from inference_server.models.paged_kv_cache import make_pools_for_gemma

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GemmaForCausalLM.from_hf(model_name, dtype=torch.bfloat16).to(self.device).eval()
        eos = self.tokenizer.eos_token_id
        self._eos_ids = {eos} if isinstance(eos, int) else set(eos)

        # Pre-allocate paged block pools shared across sessions. Sliding pools can be sized
        # smaller (capped at the window) to free memory for the full pools — see kv_reserve.
        n_blocks = int(os.environ.get("CUSTOM_BACKEND_BLOCKS", "512"))
        sliding_blocks = int(os.environ.get("CUSTOM_BACKEND_SLIDING_BLOCKS", str(n_blocks)))
        bsz = int(os.environ.get("CUSTOM_BACKEND_BLOCK_SIZE", "16"))
        self.pools = make_pools_for_gemma(
            self.model, num_blocks_per_pool=n_blocks, block_size=bsz, sliding_blocks=sliding_blocks,
        )
        self._reserved = [0] * len(self.pools)  # per-pool blocks reserved by admitted requests

        from inference_server.models.paged_kv_cache import PrefixCache
        self.prefix_cache = PrefixCache(pools=self.pools)

    def set_cache_adapter(self, adapter) -> None:
        # Custom backend caches internally (self.prefix_cache + paged pools), not via the
        # HF-format CacheManager. Leave cache_adapter=None so the scheduler skips its
        # KV-pressure gate + release for this backend instead of driving the wrong cache.
        pass

    # ---- Required interface ----

    @torch.no_grad()
    def generate(
        self, token_ids: list[int], max_tokens: int,
        template_prefix_len: int = 0,
        session_id: str = "default",
        sampling: SamplingParams = GREEDY,
    ) -> list[int]:
        """Prefix-cache lookup → prefill only the suffix → incremental decode."""
        from inference_server.models.paged_kv_cache import PagedKVCache

        with self._lock:
            matched, shared = self.prefix_cache.lookup(token_ids)
            self.last_cache_hit_tokens = matched
            cache = PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=matched)
            try:
                # Prefill only the un-cached suffix; if matched == len(token_ids) we still need
                # at least one forward to produce next-token logits, so leave the last token unmatched.
                if matched >= len(token_ids):
                    matched = len(token_ids) - 1
                    # Trim shared blocks so we don't double-count the last token's position
                    # (this is rare — only when the entire prompt is in cache)
                prompt_suffix = token_ids[matched:]
                prompt = torch.tensor([prompt_suffix], device=self.device)
                logits = self.model(prompt, kv_cache=cache)
                tok = int(sample(logits[:, -1, :], sampling).item())

                visible: list[int] = []
                in_thinking = False

                for _ in range(max_tokens * 4):
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

                    step = torch.tensor([[tok]], device=self.device)
                    logits = self.model(step, kv_cache=cache)
                    tok = int(sample(logits[:, -1, :], sampling).item())
                # Store the prompt's full blocks for future sharing (before releasing our refs).
                # Skip if a sliding layer evicted (prompt > window) — we only cache prefixes
                # that fit fully in the window (no partial/offset blocks to share).
                if not cache.any_evicted:
                    self.prefix_cache.store(token_ids, cache.block_tables)
                return visible
            finally:
                cache.free_all()

    @torch.no_grad()
    def stream(
        self, token_ids: list[int], max_tokens: int,
        template_prefix_len: int = 0,
        session_id: str = "default",
        sampling: SamplingParams = GREEDY,
    ) -> Generator[int, None, None]:
        from inference_server.models.paged_kv_cache import PagedKVCache

        with self._lock:
            matched, shared = self.prefix_cache.lookup(token_ids)
            self.last_cache_hit_tokens = matched
            cache = PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=matched)
            try:
                if matched >= len(token_ids):
                    matched = len(token_ids) - 1
                prompt_suffix = token_ids[matched:]
                prompt = torch.tensor([prompt_suffix], device=self.device)
                logits = self.model(prompt, kv_cache=cache)
                tok = int(sample(logits[:, -1, :], sampling).item())

                visible_count = 0
                in_thinking = False

                for _ in range(max_tokens * 4):
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

                    step = torch.tensor([[tok]], device=self.device)
                    logits = self.model(step, kv_cache=cache)
                    tok = int(sample(logits[:, -1, :], sampling).item())
                if not cache.any_evicted:   # only cache prefixes that fit fully in the window
                    self.prefix_cache.store(token_ids, cache.block_tables)
            finally:
                cache.free_all()

    @torch.no_grad()
    def generate_step(
        self, token_ids: list[int], kv_cache=None,
        sampling: SamplingParams = GREEDY,
    ) -> tuple[int, object]:
        """One forward pass. Pass our KVCache to incremental-decode; pass None to do full prefill."""
        from inference_server.models.gemma4 import KVCache
        if kv_cache is None:
            kv_cache = KVCache(num_layers=self.model.model.num_layers)
        input_tensor = torch.tensor([token_ids], device=self.device)
        logits = self.model(input_tensor, kv_cache=kv_cache)
        tok = int(sample(logits[:, -1, :], sampling).item())
        return tok, kv_cache

    def generate_batch(self, batch_token_ids, max_tokens, session_ids=None):
        # Naive — loop generate() per row; concurrent batching belongs in M2.
        out = []
        for ids, mt in zip(batch_token_ids, max_tokens):
            out.append(self.generate(ids, mt, session_id=(session_ids or ["default"])[len(out)]))
        return out

    # ---- Continuous-batching primitives (driven by ContinuousBatchScheduler) ----
    # Caller (scheduler) holds self._lock — these must not re-acquire it.
    # `batched_kv` is a plain list[PagedKVCache], one per active row.

    @torch.no_grad()
    def prefill(self, token_ids, session_id="default"):
        """Prefix-cache lookup → forward on the suffix → store. Returns (cache, first_token, kv_len)."""
        from inference_server.models.paged_kv_cache import PagedKVCache
        matched, shared = self.prefix_cache.lookup(token_ids)
        self.last_cache_hit_tokens = matched
        cache = PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=matched)
        if matched >= len(token_ids):
            matched = len(token_ids) - 1  # leave ≥1 token so we get next-token logits
        suffix = token_ids[matched:]
        logits = self.model(torch.tensor([suffix], device=self.device), kv_cache=cache)
        first_token = int(sample(logits[:, -1, :], GREEDY).item())
        if not cache.any_evicted:   # only cache prefixes that fit fully in the window
            self.prefix_cache.store(token_ids, cache.block_tables)
        return cache, first_token, cache.seq_len

    @torch.no_grad()
    def prefill_lookup(self, token_ids, session_id="default"):
        """Cache lookup only — no forward pass. Returns (seeded PagedKVCache, matched)."""
        from inference_server.models.paged_kv_cache import PagedKVCache
        matched, shared = self.prefix_cache.lookup(token_ids)
        self.last_cache_hit_tokens = matched
        cache = PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=matched)
        return cache, matched

    @torch.no_grad()
    def prefill_chunk(self, chunk_token_ids, partial_kv, sampling: SamplingParams = GREEDY):
        """Forward pass on a chunk against partial_kv. Returns (cache, last_token, kv_len)."""
        from inference_server.models.paged_kv_cache import PagedKVCache
        if partial_kv is None:
            partial_kv = PagedKVCache(pools=self.pools)
        logits = self.model(torch.tensor([chunk_token_ids], device=self.device), kv_cache=partial_kv)
        last_token = int(sample(logits[:, -1, :], sampling).item())
        return partial_kv, last_token, partial_kv.seq_len

    def prefill_store(self, token_ids, full_kv, matched, session_id="default"):
        """Store the prompt's full blocks for future sharing (PrefixCache dedupes by prefix)."""
        if not full_kv.any_evicted:   # only cache prefixes that fit fully in the window
            self.prefix_cache.store(token_ids, full_kv.block_tables)

    @torch.no_grad()
    def decode_step_batched(self, current_tokens, batched_kv, attention_mask,
                            position_ids, sampling_per_row=None):
        """One batched forward over all rows. `attention_mask` (scheduler-built) is ignored;
        the window/causality come from per-row seq_lens inside the model.

        CUDA: `batched_kv` is a persistent `BatchedDecodeState` — block tables live as GPU
        tensors, so the per-step hot path is pure GPU (no Python rebuild). CPU/MPS: a
        `list[PagedKVCache]` gathered into a left-padded masked SDPA (portable reference)."""
        if self.device.type == "cuda":
            state = batched_kv
            n = state.n_rows
            state.prepare_step()  # alloc any boundary-crossing block (once per step)
            logits = self.model(current_tokens, position_ids=position_ids, paged_ctx=state)
            state.advance()       # bump seq_lens + evict out-of-window blocks
        else:
            from inference_server.models.paged_kv_cache import BatchedPagedKVCache
            rows = batched_kv
            n = len(rows)
            ctx = BatchedPagedKVCache(rows)
            seq_lens = [r.seq_len for r in rows]
            lmax = max(seq_lens)
            mask = torch.zeros(n, 1, 1, lmax + 1, dtype=torch.bool, device=self.device)
            for i, L in enumerate(seq_lens):
                mask[i, 0, 0, lmax - L:] = True
            logits = self.model(current_tokens, position_ids=position_ids, kv_cache=ctx, attn_mask=mask)

        last = logits[:, -1, :]  # [N, V]
        if sampling_per_row is None:
            next_tokens = sample(last, GREEDY)
        else:
            next_tokens = torch.stack([sample(last[i], sampling_per_row[i]) for i in range(n)])
        return next_tokens, batched_kv

    def splice_into_batched(self, batched_kv, new_kv, new_kv_len):
        """Add a newly-prefilled row. CUDA: ingest into the persistent BatchedDecodeState.
        CPU: append to the list of per-row caches."""
        if self.device.type == "cuda":
            from inference_server.models.paged_kv_cache import BatchedDecodeState
            state = batched_kv if batched_kv is not None else BatchedDecodeState(self.pools, self.device)
            state.add_row(new_kv)
            return state
        if batched_kv is None:
            return [new_kv]
        batched_kv.append(new_kv)
        return batched_kv

    def remove_row_from_cache(self, batched_kv, row_idx):
        """Drop one row and release its blocks back to the pool."""
        if self.device.type == "cuda":
            batched_kv.remove_row(row_idx)
        else:
            batched_kv.pop(row_idx).free_all()
        return batched_kv

    def kv_length(self, kv):
        """Longest row's seq_len (used only for the scheduler's mask bookkeeping)."""
        if self.device.type == "cuda":
            return int(kv.seq_lens.max().item()) if kv.n_rows else 0
        return max((c.seq_len for c in kv), default=0)

    # --- Window-aware per-pool KV admission ---

    def _kv_footprints(self, prompt_len: int, max_tokens: int) -> list[int]:
        """Worst-case blocks this request occupies in each pool. Sliding pools are capped at
        the window (they evict out-of-window blocks); full pools grow with the sequence."""
        tokens = prompt_len + max_tokens
        fps = []
        for pool in self.pools:
            if pool is None:
                fps.append(0)
                continue
            blocks = (tokens + pool.block_size - 1) // pool.block_size
            if pool.window is not None:
                window_blocks = pool.window // pool.block_size + 1  # +1 for window/block straddle
                blocks = min(blocks, window_blocks)
            fps.append(blocks)
        return fps

    def kv_reserve(self, prompt_len: int, max_tokens: int) -> bool:
        """Admit only if this request's per-pool footprint fits in EVERY pool's free blocks.
        Commits the reservation on success. Single-threaded (scheduler worker) → no lock."""
        fps = self._kv_footprints(prompt_len, max_tokens)
        for i, pool in enumerate(self.pools):
            if pool is not None and self._reserved[i] + fps[i] > pool.num_blocks:
                return False
        for i in range(len(self.pools)):
            self._reserved[i] += fps[i]
        return True

    def kv_release(self, prompt_len: int, max_tokens: int) -> None:
        fps = self._kv_footprints(prompt_len, max_tokens)
        for i in range(len(self.pools)):
            self._reserved[i] -= fps[i]

    def is_eos(self, token_id: int) -> bool:
        return token_id in self._eos_ids

    @property
    def device_str(self) -> str:
        return str(self.device)
