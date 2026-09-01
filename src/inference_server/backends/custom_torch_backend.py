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
import os
import threading
from typing import Generator

import torch

from inference_server.backends.base import InferenceBackend
from inference_server.sampling import GREEDY, SamplingParams, sample

logger = logging.getLogger(__name__)

# Suffix-length buckets for K=1 CUDA-graph prefill (≤512 so sliding layers never evict mid-prefill).
_PREFILL_BUCKETS = (64, 128, 256, 512)


def _decode_buckets(max_rows: int) -> tuple[int, ...]:
    """Row buckets for decode CUDA graphs: powers of two up to max_rows (<=2x padding waste).
    Override with CUSTOM_BACKEND_GRAPH_BUCKETS="1,4,16,64" to trade padding for capture time."""
    override = os.environ.get("CUSTOM_BACKEND_GRAPH_BUCKETS", "")
    if override:
        vals = sorted({min(int(x), max_rows) for x in override.split(",") if x.strip()})
        return tuple(vals) or (max_rows,)
    out, b = [], 1
    while b < max_rows:
        out.append(b)
        b *= 2
    out.append(max_rows)
    return tuple(out)


def _prefill_bucket(s: int) -> int | None:
    for b in _PREFILL_BUCKETS:
        if s <= b:
            return b
    return None


class _PrefillGraphCtx:
    """Static-buffer paged ctx for K=1 CUDA-graph prefill at a fixed suffix bucket S, no prefix
    hit (matched=0). Mirrors _GraphCtx: append scatters the S bucket tokens' K/V using the static
    per-pool block-table buffers; the kernel reads those + the static prefix/suffix lens. Block
    allocation happens OUTSIDE the graph (at replay); the graph only touches the static buffers."""

    is_prefill = True

    def __init__(self, pools, block_tables, prefix_lens, suffix_lens, bucket, bs):
        self.pools = pools
        self.block_tables = block_tables      # per-pool static [1, cols] long (None for shared)
        self.prefix_lens = prefix_lens        # static [1] int32 (always 0 — no prefix hit)
        self.suffix_lens = suffix_lens        # static [1] int32
        self.bs = bs
        self._pos = torch.arange(bucket, device=prefix_lens.device)   # 0..bucket-1 (prefix_len=0)

    def append(self, layer_idx, k_new, v_new):
        bt = self.block_tables[layer_idx][0]              # [cols]
        bids = bt[self._pos // self.bs]
        slots = self._pos % self.bs
        pool = self.pools[layer_idx]
        pool.k[bids, :, slots, :] = k_new[0].transpose(0, 1)   # [bucket, Hkv, D]
        pool.v[bids, :, slots, :] = v_new[0].transpose(0, 1)

    def block_table_tensor(self, layer_idx):
        return self.block_tables[layer_idx].to(torch.int32).clamp_min(0)


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


class _PrefillCtx:
    """paged_ctx for batched prefill — multi-query, kernel-backed (no gather). Holds per-row paged
    caches (each seeded with its cached prefix); the model appends the suffix K/V and the prefill
    kernel attends over the paged prefix+suffix. `is_prefill=True` routes the model's attention here.
    Suffix is right-padded (real tokens at [0, suffix_len)); padding queries are skipped in-kernel."""

    is_prefill = True

    def __init__(self, caches, pools, prefix_lens, suffix_lens, device):
        self.caches = caches                       # list[PagedKVCache], one per row
        self.pools = pools
        self._suffix = suffix_lens                 # python list (for the per-row append slice)
        self.prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
        self.suffix_lens = torch.tensor(suffix_lens, dtype=torch.int32, device=device)
        self.device = device

    def append(self, layer_idx, k_new, v_new):
        # k_new/v_new: [B, Hkv, Smax, D] (right-padded). Append each row's real suffix to its cache.
        for b, cache in enumerate(self.caches):
            s = self._suffix[b]
            cache.append(layer_idx, k_new[b:b + 1, :, :s, :], v_new[b:b + 1, :, :s, :])

    def block_table_tensor(self, layer_idx):
        rows = [c.block_tables[layer_idx] for c in self.caches]
        maxb = max((len(r) for r in rows), default=1)
        bt = torch.zeros(len(rows), maxb, dtype=torch.int32, device=self.device)
        for b, r in enumerate(rows):
            if r:
                bt[b, :len(r)] = torch.tensor([x if x >= 0 else 0 for x in r],
                                              dtype=torch.int32, device=self.device)
        return bt


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

        # CUDA-graph decode: one graph per ROW BUCKET, replay the smallest bucket >= n_rows.
        # A single max-batch graph padded every step to max_batch_size (64x waste at 4 active
        # rows with max_batch=256), which dominated TPOT at the low concurrency the SLO lives at.
        from inference_server.config import settings
        self._block_size = bsz
        self._graph_max_rows = settings.max_batch_size
        self._graph_max_cols = (settings.context_window + bsz - 1) // bsz
        self._graph_on = self.device.type == "cuda" and os.environ.get("CUSTOM_BACKEND_CUDA_GRAPH", "1") == "1"
        self._graph_buckets = _decode_buckets(self._graph_max_rows)
        self._graphs: dict[int, dict] = {}   # bucket -> captured graph + its static buffers
        self._scratch = None                 # one padding block per pool, shared by every bucket

        # torch.compile the DECODE forward only (Inductor fuses the ~50% element-wise/norm/RoPE
        # tail the profiler found — the bandwidth lever on A100). Decode-only because prefill's
        # ragged shapes would recompile-storm a shared compile; decode is the static [maxN,1]
        # graph shape. Default mode = fusion, no internal cudagraphs → composes with our manual
        # CUDA graph. The graph captures the fused kernel stream. CPU path stays uncompiled.
        self._compile_on = self.device.type == "cuda" and os.environ.get("CUSTOM_BACKEND_COMPILE", "0") == "1"
        if self._compile_on:
            # mode=None → default fusion (no internal cudagraphs, composes with our manual graph).
            # "max-autotune-no-cudagraphs" autotunes GEMMs + fuses harder; the -no-cudagraphs
            # variant is REQUIRED (plain max-autotune adds its own cudagraphs → conflicts capture).
            mode = os.environ.get("CUSTOM_BACKEND_COMPILE_MODE") or None
            self._decode_fwd = torch.compile(self.model, dynamic=False, mode=mode)
            logger.info("torch.compile enabled for decode forward (mode=%s)", mode or "default")
        else:
            self._decode_fwd = self.model

        # K=1 CUDA-graph prefill — replays the forward instead of paying ~73ms eager dispatch
        # (the unloaded-TTFT floor). Captured lazily per suffix bucket on first use (capture is
        # ~seconds, unlike torch.compile). Flag-gated OFF until verified; deploy unaffected.
        self._prefill_graph_on = self.device.type == "cuda" and \
            os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "0") == "1"
        self._prefill_graphs: dict = {}

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
        """Prefix-cache-aware batched prefill — the TTFT lever. Per row: cache-lookup → seed the
        cached prefix → then ONE batched forward over the (much shorter) SUFFIXES, each attending to
        its own prefix. For repeated/shared prompts the suffix is ~1 token, so the forward is tiny
        (dispatch-bound) instead of re-prefilling the full prompt. No cache hit → suffix = full prompt
        (the v1 behaviour). Returns per-row (PagedKVCache, first_token, kv_len). Device-agnostic
        (gather + masked SDPA). Left-pad both prefix and suffix; a per-row mask keeps rows independent
        (suffix query attends to its real prefix keys + causal suffix keys).

        CUDA uses the gather-free paged prefill kernel instead (`_prefill_batch_kernel`)."""
        if self.device.type == "cuda":
            return self._prefill_batch_kernel(prompts)
        from inference_server.models.gemma4 import KVCache
        from inference_server.models.paged_kv_cache import PagedKVCache
        dev = self.device
        K = len(prompts)
        nlayers = self.model.model.num_layers

        caches, matched, suffixes = [], [], []
        for p in prompts:
            m, shared = self.prefix_cache.lookup(p)
            if m >= len(p):
                m = len(p) - 1  # leave ≥1 token so we get next-token logits
            caches.append(PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=m))
            matched.append(m)
            suffixes.append(p[m:])
        self.last_cache_hit_tokens = matched[-1] if matched else 0

        Smax = max(len(s) for s in suffixes)
        Pmax = max(matched) if matched else 0

        input_ids = torch.zeros(K, Smax, dtype=torch.long, device=dev)
        position_ids = torch.zeros(K, Smax, dtype=torch.long, device=dev)
        for k in range(K):
            s, m = suffixes[k], matched[k]
            input_ids[k, Smax - len(s):] = torch.tensor(s, device=dev)
            position_ids[k, Smax - len(s):] = torch.arange(m, m + len(s), device=dev)

        # Seed a batched KVCache with each row's gathered prefix, left-padded to Pmax (per layer).
        seeded = KVCache(num_layers=nlayers)
        if Pmax > 0:
            for i in range(nlayers):
                rows = [caches[k].get(i) for k in range(K)]
                if all(r is None for r in rows):
                    continue  # shared layer, or no row has a prefix here
                ref = next(r for r in rows if r is not None)[0]
                H, D = ref.shape[1], ref.shape[3]
                kbuf = torch.zeros(K, H, Pmax, D, dtype=ref.dtype, device=dev)
                vbuf = torch.zeros(K, H, Pmax, D, dtype=ref.dtype, device=dev)
                for k, r in enumerate(rows):
                    if r is not None:
                        kbuf[k, :, Pmax - matched[k]:, :] = r[0][0]
                        vbuf[k, :, Pmax - matched[k]:, :] = r[1][0]
                seeded.k[i], seeded.v[i] = kbuf, vbuf

        # Mask [K,1,Smax,Pmax+Smax]: suffix query attends to its real prefix keys + causal suffix keys.
        Ktot = Pmax + Smax
        mask = torch.zeros(K, 1, Smax, Ktot, dtype=torch.bool, device=dev)
        qi = torch.arange(Smax, device=dev)
        ki = torch.arange(Ktot, device=dev)
        for k in range(K):
            L, m = len(suffixes[k]), matched[k]
            soff = Smax - L
            real_q = (qi >= soff)[:, None]                                       # [Smax,1]
            prefix_key = ((ki < Pmax) & (ki >= Pmax - m))[None, :]               # [1,Ktot]
            suffix_key = ((ki >= Pmax + soff))[None, :] & ((ki[None, :] - Pmax) <= qi[:, None])
            mask[k, 0] = real_q & (prefix_key | suffix_key)

        logits = self.model(input_ids, position_ids=position_ids, kv_cache=seeded, attn_mask=mask)

        results = []
        for k in range(K):
            L = len(suffixes[k])
            first = int(sample(logits[k:k + 1, -1, :], GREEDY).item())
            cache = caches[k]
            for i in range(nlayers):
                kvi = seeded.get(i)
                if kvi is None:  # KV-shared layer — reads the source layer's pool
                    continue
                ki_t, vi_t = kvi  # [K,H,Pmax+Smax,D]; this row's real suffix = the last L positions
                tot = ki_t.shape[2]
                cache.append(i, ki_t[k:k + 1, :, tot - L:, :], vi_t[k:k + 1, :, tot - L:, :])
            if not cache.any_evicted:
                self.prefix_cache.store(prompts[k], cache.block_tables)
            results.append((cache, first, len(prompts[k])))
        return results

    @torch.no_grad()
    def _prefill_batch_kernel(self, prompts):
        """CUDA batched prefill via the paged prefill kernel — no gather. Per row: cache-lookup →
        seed prefix blocks → one forward over the right-padded SUFFIXES, where each layer appends
        the suffix K/V to the blocks and the kernel attends over paged prefix+suffix."""
        # K=1 CUDA-graph fast path: cache-miss prompt ≤512 → replay a captured graph (no eager
        # dispatch). Multi-row, cache-hit, or long prompts fall through to the eager kernel path.
        if self._prefill_graph_on and len(prompts) == 1:
            p = prompts[0]
            matched0, _ = self.prefix_cache.lookup(p)
            bucket = _prefill_bucket(len(p))
            if matched0 == 0 and bucket is not None:
                if bucket not in self._prefill_graphs:
                    self._capture_prefill_graph(bucket)
                self.last_cache_hit_tokens = 0
                return [self._replay_prefill(p, bucket)]
        from inference_server.models.paged_kv_cache import PagedKVCache
        dev = self.device
        caches, matched, suffixes = [], [], []
        for p in prompts:
            m, shared = self.prefix_cache.lookup(p)
            if m >= len(p):
                m = len(p) - 1
            caches.append(PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=m))
            matched.append(m)
            suffixes.append(p[m:])
        self.last_cache_hit_tokens = matched[-1] if matched else 0

        Smax = max(len(s) for s in suffixes)
        input_ids = torch.zeros(len(prompts), Smax, dtype=torch.long, device=dev)
        position_ids = torch.zeros(len(prompts), Smax, dtype=torch.long, device=dev)
        for k, s in enumerate(suffixes):
            input_ids[k, :len(s)] = torch.tensor(s, device=dev)               # right-padded
            position_ids[k, :len(s)] = torch.arange(matched[k], matched[k] + len(s), device=dev)

        ctx = _PrefillCtx(caches, self.pools, matched, [len(s) for s in suffixes], dev)
        logits = self.model(input_ids, position_ids=position_ids, paged_ctx=ctx)

        results = []
        for k in range(len(prompts)):
            first = int(sample(logits[k:k + 1, len(suffixes[k]) - 1, :], GREEDY).item())
            if not caches[k].any_evicted:
                self.prefix_cache.store(prompts[k], caches[k].block_tables)
            results.append((caches[k], first, len(prompts[k])))
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
            if self._graph_on and not self._graphs:
                self._capture_all_graphs()   # all buckets up-front, largest first (shared pool)
            if self._graph_on:
                logits = self._replay_decode(current_tokens, position_ids, state,
                                             self._decode_bucket(n))
            else:
                logits = self._decode_fwd(current_tokens, position_ids=position_ids, paged_ctx=state)
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

    # --- CUDA-graph decode capture/replay (one graph per row bucket; see DECISIONS) ---

    def _capture_graph(self, bucket: int) -> None:
        """Capture the decode forward over static buffers sized to `bucket` rows. Warmup runs
        eagerly (stabilizes allocator + JITs the Triton kernel) before capture. Falls back to eager.

        Each bucket gets its OWN CUDA-graph memory pool. Sharing one is only safe when graphs are
        replayed in capture order, and we replay whichever bucket fits the current row count.
        ~250MB total here (the [B,1,262144] lm_head output dominates) buys that guarantee.

        NOTE on determinism: which bucket a step lands in now depends on how many requests are
        in flight, and a [B,1,H] GEMM picks a different cuBLAS kernel (different accumulation
        order) per B. So a prompt's tokens can differ with concurrency. This is a property of
        bucketing, not of graphs: scripts/diag_pad_numerics_modal.py reproduces it EAGER, with
        no graph anywhere, purely by changing the pad width. vLLM behaves the same way. Output
        is still deterministic at a fixed batch width."""
        import time
        t_b = time.perf_counter()
        cols, dev = self._graph_max_cols, self.device
        if self._scratch is None:
            self._scratch = [None] * len(self.pools)
            for L, pool in enumerate(self.pools):
                if pool is not None:
                    self._scratch[L] = pool.alloc()  # dummy/padding rows point here; held for the process
        g_tokens = torch.zeros(bucket, 1, dtype=torch.long, device=dev)
        g_pos = torch.zeros(bucket, 1, dtype=torch.long, device=dev)
        g_seqlens = torch.ones(bucket, dtype=torch.long, device=dev)
        g_bt = [None] * len(self.pools)
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            g_bt[L] = torch.full((bucket, cols), self._scratch[L], dtype=torch.long, device=dev)
        ctx = _GraphCtx(self.pools, g_seqlens, g_bt, self._block_size)
        if os.environ.get("CUSTOM_BACKEND_EXPLAIN", "0") == "1":
            self._log_graph_breaks(ctx, g_tokens, g_pos)
        try:
            if self._compile_on:
                # Force the Dynamo trace + Inductor compile on the DEFAULT stream first — it must
                # NOT happen during capture (below) or on the side stream. Each bucket is a new
                # static shape, so each pays its own compile.
                for _ in range(3):
                    self._decode_fwd(g_tokens, position_ids=g_pos, paged_ctx=ctx)
                torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warmup (compile kernels, stabilize allocator)
                    self._decode_fwd(g_tokens, position_ids=g_pos, paged_ctx=ctx)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):   # private pool per bucket — see docstring
                out = self._decode_fwd(g_tokens, position_ids=g_pos, paged_ctx=ctx)
            self._graphs[bucket] = {"graph": graph, "tokens": g_tokens, "pos": g_pos,
                                    "seqlens": g_seqlens, "bt": g_bt, "out": out}
            print(f"[graphs]   bucket {bucket} captured in {time.perf_counter() - t_b:.1f}s",
                  flush=True)
        except Exception:
            logger.exception("CUDA graph capture failed — falling back to eager decode")
            self._graph_on = False
            self._graphs.clear()

    def _capture_all_graphs(self) -> None:
        """Capture every bucket up-front, largest first. Done on the first decode (inside
        warmup) so no request eats a mid-serving capture stall."""
        import time
        t0 = time.perf_counter()
        for bucket in sorted(self._graph_buckets, reverse=True):
            if not self._graph_on:
                return
            self._capture_graph(bucket)
        # print, not logger.info: bench containers do not configure logging and drop INFO.
        # Capture time matters — with torch.compile on, every bucket is a fresh static shape
        # and pays its own Inductor compile, so this scales with len(buckets).
        print(f"[graphs] captured {len(self._graphs)} decode graphs {sorted(self._graphs)} "
              f"in {time.perf_counter() - t0:.1f}s", flush=True)

    def _decode_bucket(self, n: int) -> int:
        """Smallest captured bucket >= n rows."""
        for b in self._graph_buckets:
            if b >= n:
                return b
        return self._graph_buckets[-1]

    @torch.no_grad()
    def _log_graph_breaks(self, ctx, g_tokens, g_pos) -> None:
        """Count torch.compile graph breaks in the decode forward (Inductor fusion headroom).
        Each break is a fence Inductor can't fuse across — expect one per layer at the raw Triton
        attention call + the Python scatter. Run via CUSTOM_BACKEND_EXPLAIN=1."""
        try:
            import torch._dynamo as dynamo
            from collections import Counter
            exp = dynamo.explain(self.model)(g_tokens, position_ids=g_pos, paged_ctx=ctx)
            # print (not logger.info) — bench containers don't configure logging, so INFO is dropped.
            print(f"[graph-breaks] {exp.graph_break_count} breaks, {exp.graph_count} graphs, "
                  f"{exp.op_count} ops captured", flush=True)
            reasons = Counter(getattr(r, "reason", str(r)) for r in exp.break_reasons)
            for reason, count in reasons.most_common():
                print(f"[graph-breaks]   {count}x {reason}", flush=True)
        except Exception as e:
            print(f"[graph-breaks] explain failed (diagnostic only): {e}", flush=True)

    def _replay_decode(self, current_tokens, position_ids, state, bucket):
        """Copy this step's inputs into `bucket`'s static buffers (real rows + scratch dummies),
        replay that bucket's graph, return logits[:n_rows]. The graph reads the static buffers,
        so its scatter/kernel run on the fresh values — no per-op dispatch."""
        n = state.n_rows
        g = self._graphs[bucket]
        g["tokens"][:n].copy_(current_tokens)
        g["pos"][:n].copy_(position_ids)
        g["seqlens"][:n].copy_(state.seq_lens)
        g["seqlens"][n:] = 1
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            g["bt"][L].fill_(self._scratch[L])           # reset dummies + stale columns to scratch
            bt = state.block_tables[L]
            g["bt"][L][:n, :bt.shape[1]].copy_(bt.clamp_min(0))
        g["graph"].replay()
        return g["out"][:n]

    def _capture_prefill_graph(self, bucket: int) -> None:
        """Capture the prefill forward once over static [1, bucket] buffers. Scratch blocks back the
        block tables during capture; replay copies in the real block IDs + tokens."""
        dev, bs, cols = self.device, self._block_size, bucket // self._block_size
        ids = torch.zeros(1, bucket, dtype=torch.long, device=dev)
        pos = torch.arange(bucket, device=dev).unsqueeze(0)
        prefix_lens = torch.zeros(1, dtype=torch.int32, device=dev)
        suffix_lens = torch.full((1,), bucket, dtype=torch.int32, device=dev)   # all real during capture
        bts = [None] * len(self.pools)
        scratch = [None] * len(self.pools)
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            scratch[L] = pool.alloc()                                            # held for the process
            bts[L] = torch.full((1, cols), scratch[L], dtype=torch.long, device=dev)
        ctx = _PrefillGraphCtx(self.pools, bts, prefix_lens, suffix_lens, bucket, bs)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(ids, position_ids=pos, paged_ctx=ctx)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(ids, position_ids=pos, paged_ctx=ctx)
        self._prefill_graphs[bucket] = {"graph": g, "ids": ids, "suffix_lens": suffix_lens,
                                        "bts": bts, "out": out}
        logger.info("Captured CUDA prefill graph (bucket=%d)", bucket)

    def _replay_prefill(self, prompt_ids, bucket):
        """K=1, no-prefix prefill via graph replay. Allocate blocks, copy IDs into static buffers,
        replay, return (cache, first_token, kv_len)."""
        from inference_server.models.paged_kv_cache import PagedKVCache
        dev, cols, S = self.device, bucket // self._block_size, len(prompt_ids)
        g = self._prefill_graphs[bucket]
        cache = PagedKVCache(pools=self.pools)
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            bids = [pool.alloc() for _ in range(cols)]
            cache.block_tables[L] = bids
            cache.seq_lens[L] = S
            g["bts"][L][0].copy_(torch.tensor(bids, device=dev))
        g["ids"].zero_()
        g["ids"][0, :S].copy_(torch.tensor(prompt_ids, device=dev))
        g["suffix_lens"].fill_(S)                                               # padding queries skipped
        g["graph"].replay()
        first = int(sample(g["out"][:, S - 1, :], GREEDY).item())
        if not cache.any_evicted:
            self.prefix_cache.store(prompt_ids, cache.block_tables)
        return cache, first, S

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
