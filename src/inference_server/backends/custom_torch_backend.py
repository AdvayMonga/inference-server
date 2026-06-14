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


class _GraphCtx:
    """paged_ctx backed by FIXED static buffers, for CUDA-graph capture. Always processes
    `maxN` rows (inactive rows are dummies pointing at a scratch block). Same append /
    block_table_tensor interface the model uses, but reading the static buffers — so the
    captured graph reads/writes fixed addresses and a replay just needs fresh values copied in.
    """

    def __init__(self, pools, seq_lens, block_tables, bs):
        self.pools = pools
        self.seq_lens = seq_lens          # static [maxN] long
        self.block_tables = block_tables  # static per-layer [maxN, MAX_COLS] long
        self.bs = bs

    def append(self, layer_idx, k_new, v_new):
        pool = self.pools[layer_idx]
        rows = torch.arange(self.seq_lens.shape[0], device=self.seq_lens.device)
        lb = self.seq_lens // self.bs
        slot = self.seq_lens % self.bs
        bid = self.block_tables[layer_idx][rows, lb]
        pool.k[bid, :, slot, :] = k_new[:, :, 0, :]
        pool.v[bid, :, slot, :] = v_new[:, :, 0, :]

    def block_table_tensor(self, layer_idx):
        return self.block_tables[layer_idx].to(torch.int32).clamp_min(0), (self.seq_lens + 1).to(torch.int32)


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

        # Weight-only int8: store Linear weights as int8 (per-channel), read 2× fewer bytes/step.
        # The bandwidth lever for decode; dequant is fused into the GEMV (see models/quant.py).
        if os.environ.get("CUSTOM_BACKEND_QUANT", "").lower() == "int8":
            from inference_server.models.quant import quantize_model_int8
            # On CUDA, route layers through the shared compiled GEMV so Inductor fuses the dequant
            # (keeps weight int8 in HBM — the bandwidth win); composes with the decode CUDA graph.
            on_cuda = self.device.type == "cuda"
            nq = quantize_model_int8(self.model, compiled=on_cuda)
            logger.info("Quantized %d Linear layers to int8 weight-only%s", nq,
                        " (+compiled)" if on_cuda else "")

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

        # CUDA-graph decode: capture one graph at max_batch and replay (pad smaller batches).
        # We're dispatch-bound (~1000 tiny launches/step), so one graph beats per-bucket capture.
        from inference_server.config import settings
        self._block_size = bsz
        self._graph_max_rows = settings.max_batch_size
        self._graph_max_cols = (settings.context_window + bsz - 1) // bsz
        self._graph_on = self.device.type == "cuda" and os.environ.get("CUSTOM_BACKEND_CUDA_GRAPH", "1") == "1"
        self._graph = None  # captured lazily on first decode

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
    def prefill_batch(self, prompts, session_ids=None):
        """Prefill K prompts in ONE left-padded forward instead of K serial eager ones — the TTFT
        lever (prefill is dispatch-bound, ~4× a decode step, so batching pays the dispatch once).
        Returns a list of (PagedKVCache, first_token, kv_len), one per prompt.

        v1: full prefill, no prefix-cache lookup (per-row cache hits give ragged suffixes that don't
        batch cleanly); each prompt is stored after, so the single-prefill path stays warm. Left-pad
        so real tokens are right-aligned; a per-row causal+padding mask keeps rows independent; each
        row's real-token KV is sliced out and written to its own paged cache."""
        from inference_server.models.gemma4 import KVCache
        from inference_server.models.paged_kv_cache import PagedKVCache
        dev = self.device
        K = len(prompts)
        lens = [len(p) for p in prompts]
        Lmax = max(lens)
        nlayers = self.model.model.num_layers

        input_ids = torch.zeros(K, Lmax, dtype=torch.long, device=dev)
        position_ids = torch.zeros(K, Lmax, dtype=torch.long, device=dev)
        mask = torch.zeros(K, 1, Lmax, Lmax, dtype=torch.bool, device=dev)
        idx = torch.arange(Lmax, device=dev)
        causal = idx[None, :] <= idx[:, None]  # [q, kk] → kk ≤ q
        for k, p in enumerate(prompts):
            off = Lmax - lens[k]
            input_ids[k, off:] = torch.tensor(p, device=dev)
            position_ids[k, off:] = torch.arange(lens[k], device=dev)
            mask[k, 0] = (idx[None, :] >= off) & causal  # attend real keys up to the query

        kv = KVCache(num_layers=nlayers)
        logits = self.model(input_ids, position_ids=position_ids, kv_cache=kv, attn_mask=mask)

        results = []
        for k in range(K):
            off = Lmax - lens[k]
            first = int(sample(logits[k:k + 1, -1, :], GREEDY).item())
            cache = PagedKVCache(pools=self.pools)
            for i in range(nlayers):
                kvi = kv.get(i)
                if kvi is None:  # KV-shared layer — reads the source layer's pool, nothing to store
                    continue
                ki, vi = kvi
                cache.append(i, ki[k:k + 1, :, off:, :], vi[k:k + 1, :, off:, :])
            if not cache.any_evicted:
                self.prefix_cache.store(prompts[k], cache.block_tables)
            results.append((cache, first, lens[k]))
        return results

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
            if self._graph_on:
                if self._graph is None:
                    self._capture_graph()
                logits = self._replay_decode(current_tokens, position_ids, state)
            else:
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

    # --- CUDA-graph decode capture/replay (single max-batch graph; see DECISIONS) ---

    def _capture_graph(self) -> None:
        """Capture the decode forward once over fixed static buffers (maxN rows). Warmup runs
        eagerly (stabilizes allocator + JITs the Triton kernel) before capture. On any failure
        we fall back to eager decode."""
        maxN, cols, dev = self._graph_max_rows, self._graph_max_cols, self.device
        self._g_tokens = torch.zeros(maxN, 1, dtype=torch.long, device=dev)
        self._g_pos = torch.zeros(maxN, 1, dtype=torch.long, device=dev)
        self._g_seqlens = torch.ones(maxN, dtype=torch.long, device=dev)
        self._g_bt = [None] * len(self.pools)
        self._scratch = [None] * len(self.pools)
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            self._scratch[L] = pool.alloc()  # dummy/padding rows point here; held for the process
            self._g_bt[L] = torch.full((maxN, cols), self._scratch[L], dtype=torch.long, device=dev)
        ctx = _GraphCtx(self.pools, self._g_seqlens, self._g_bt, self._block_size)
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warmup (compile kernels, stabilize allocator)
                    self.model(self._g_tokens, position_ids=self._g_pos, paged_ctx=ctx)
            torch.cuda.current_stream().wait_stream(s)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._g_out = self.model(self._g_tokens, position_ids=self._g_pos, paged_ctx=ctx)
            logger.info("Captured CUDA decode graph (max_rows=%d, max_cols=%d)", maxN, cols)
        except Exception:
            logger.exception("CUDA graph capture failed — falling back to eager decode")
            self._graph_on = False
            self._graph = None

    def _replay_decode(self, current_tokens, position_ids, state):
        """Copy this step's inputs into the static buffers (real rows + scratch dummies), replay
        the captured graph, return logits[:n_rows]. The graph reads the static buffers, so its
        scatter/kernel run on the fresh values — no per-op dispatch."""
        n = state.n_rows
        self._g_tokens[:n].copy_(current_tokens)
        self._g_pos[:n].copy_(position_ids)
        self._g_seqlens[:n].copy_(state.seq_lens)
        self._g_seqlens[n:] = 1
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            self._g_bt[L].fill_(self._scratch[L])           # reset dummies + stale columns to scratch
            bt = state.block_tables[L]
            self._g_bt[L][:n, :bt.shape[1]].copy_(bt.clamp_min(0))
        if self._graph is None:  # capture fell back to eager
            return self.model(current_tokens, position_ids=position_ids, paged_ctx=state)
        self._graph.replay()
        return self._g_out[:n]

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
