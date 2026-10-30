"""Custom-forward Torch backend — uses our hand-written Qwen3-MoE model.

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
import time
from typing import Generator

import torch

from inference_server.backends.base import DeviceToken, InferenceBackend, stop_token_ids
from inference_server.models import fused
from inference_server.models.paged_kv_cache import step_inputs, to_device, write_token
from inference_server.sampling import GREEDY, SamplingParams, sample, sample_batched

logger = logging.getLogger(__name__)

# Suffix-length buckets for K=1 CUDA-graph prefill. The ladder reaches past 512 because the p95
# prompt on our workload is ~830 tokens and those were the requests still stuck on the ~100ms
# eager path — i.e. exactly the ones setting TTFT p95.
#
# The ladder is finer than powers of two because a graph processes its WHOLE bucket: a 600-token
# suffix in a 1024 bucket wastes 40% of the forward, and prefill is now the entire TTFT tail
# (measured at rate 1: queue p95 21ms vs prefill p95 203ms). Prefill graphs do not go through
# torch.compile — all eight capture in a few seconds — so extra buckets are nearly free here,
# unlike the DECODE ladder where each bucket costs a ~140s compile.
_PREFILL_BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024)


def _decode_buckets(max_rows: int, coarse: bool = False) -> tuple[int, ...]:
    """Row buckets for decode CUDA graphs.

    Default is powers of two (<=2x padding waste). With torch.compile on, EVERY bucket is a
    distinct static shape and costs its own compile — measured at ~140s each, linear in bucket
    count (2 buckets 285s, 5 buckets 574s). A persistent Inductor cache only recovers ~19% of
    that, because most of it is Dynamo tracing rather than codegen. So when compiling, step by
    4x instead: <=4x padding (n=1 costs 20.2ms/step at bucket 8 vs 17.7ms at bucket 1, ~14%)
    for half the buckets and half the startup.

    Override with CUSTOM_BACKEND_GRAPH_BUCKETS="1,4,16,64" to pick the trade yourself.
    """
    override = os.environ.get("CUSTOM_BACKEND_GRAPH_BUCKETS", "")
    if override:
        vals = sorted({min(int(x), max_rows) for x in override.split(",") if x.strip()})
        return tuple(vals) or (max_rows,)
    if coarse:
        out, b = [], 4
        while b < max_rows:
            out.append(b)
            b *= 4
        out.append(max_rows)
        return tuple(dict.fromkeys(out))
    # Starts at 2, not 1: with torch.compile on, a batch-1 shape forces its own Inductor
    # specialization (Dynamo never makes a dim that can be 1 dynamic), measured at ~126s of
    # extra startup — to buy 4% at n=1 (17.73 vs 18.53 ms/step). Not worth it.
    out, b = [], 2
    while b < max_rows:
        out.append(b)
        b *= 2
    out.append(max_rows)
    return tuple(dict.fromkeys(out))


def _prefill_bucket(s: int) -> int | None:
    for b in _PREFILL_BUCKETS:
        if s <= b:
            return b
    return None


def real_tokens(suffix_lens: list[int]) -> list[int]:
    """Flat indices of the real tokens in a right-padded [rows, longest] grid."""
    s = max(suffix_lens)
    return [b * s + i for b, n in enumerate(suffix_lens) for i in range(n)]


class _PrefillGraphCtx:
    """Static-buffer paged ctx for K=1 CUDA-graph prefill at a fixed suffix bucket S, with or
    without a prefix-cache hit. Mirrors _GraphCtx: append scatters the S bucket tokens' K/V using
    the static per-pool block-table buffers; the kernel reads those + the static prefix/suffix
    lens. Block allocation happens OUTSIDE the graph (at replay); the graph only touches the
    static buffers.

    `prefix_lens` is a RUNTIME value, so the absolute write positions are derived inside the
    captured graph (prefix + 0..S-1) rather than baked at capture. That is what lets one captured
    graph serve any prefix length, which matters because with a warm prefix cache almost every
    request is a hit and the old matched==0 gate meant the graph never fired."""

    is_prefill = True

    def __init__(self, pools, block_table, prefix_lens, suffix_lens, bucket, bs):
        self.pools = pools
        self.bt = block_table                 # static [1, cols] long, one table for every layer
        self.prefix_lens = prefix_lens        # static [1] int32 — set per replay
        self.suffix_lens = suffix_lens        # static [1] int32 — set per replay
        self.bs = bs
        self._ar = torch.arange(bucket, device=prefix_lens.device)   # 0..bucket-1

    def append(self, layer_idx, k_new, v_new):
        # Absolute positions computed IN-GRAPH from the runtime prefix length.
        pos = self.prefix_lens[0].long() + self._ar        # [bucket]
        bids = self.bt[0][pos // self.bs]
        slots = pos % self.bs
        pool = self.pools[layer_idx]
        pool.k[bids, :, slots, :] = k_new[0].transpose(0, 1).to(pool.k.dtype)   # [bucket, Hkv, D]
        pool.v[bids, :, slots, :] = v_new[0].transpose(0, 1).to(pool.v.dtype)

    def block_table_tensor(self, layer_idx):
        return self.bt.to(torch.int32).clamp_min(0)


class _GraphCtx:
    """paged_ctx backed by FIXED static buffers, for CUDA-graph capture. Always processes
    `maxN` rows (inactive rows are dummies pointing at a scratch block). Same append /
    block_table_tensor interface the model uses, but reading the static buffers — so the
    captured graph reads/writes fixed addresses and a replay just needs fresh values copied in.
    """

    def __init__(self, pools, seq_lens, block_table, bs):
        self.pools = pools
        self.seq_lens = seq_lens          # static [maxN] long
        self.bt = block_table             # static [maxN, MAX_COLS] long, one table for every layer
        self.bs = bs
        self._live = [L for L, p in enumerate(pools) if p is not None]

    def append(self, layer_idx, k_new, v_new):
        if layer_idx == self._live[0]:   # per-step values once per forward, not per layer (captured as such)
            s = self.step_inputs()
            self._rows, self._lb, self._slot = s["kv_rows"], s["kv_block"], s["kv_slot"]
            self._bt32, self._sl32 = s["attn_block_tables"], s["attn_seq_lens"]
        write_token(self.pools[layer_idx], self.bt, self._rows, self._lb, self._slot, k_new, v_new)

    def block_table_tensor(self, layer_idx):
        return self._bt32, self._sl32

    def step_inputs(self) -> dict:
        return step_inputs(self.bt, self.seq_lens, self.bs)


class _PrefillCtx:
    """paged_ctx for batched prefill — multi-query, kernel-backed (no gather). Holds per-row paged
    caches (each seeded with its cached prefix); the model appends the suffix K/V and the prefill
    kernel attends over the paged prefix+suffix. `is_prefill=True` routes the model's attention here.
    Suffix is right-padded (real tokens at [0, suffix_len)); padding queries are skipped in-kernel."""

    is_prefill = True
    packed = False

    def __init__(self, caches, pools, prefix_lens, suffix_lens, device, scratch=None):
        self.caches = caches                       # list[PagedKVCache], one per row
        self.pools = pools
        self._suffix = suffix_lens                 # python list of per-row real suffix lengths
        self._prefix = list(prefix_lens)           # and cached prefix lengths (prefill_attention.sdpa_prefill)
        self.prefix_lens = to_device(list(prefix_lens), device, torch.int32)      # pinned: no wait on the stream
        self.suffix_lens = to_device(list(suffix_lens), device, torch.int32)
        self.device = device
        self._scratch = scratch                    # the padding block id (padding writes land here), every pool
        self._ar = torch.arange(max(suffix_lens) if suffix_lens else 1, device=device)
        self._valid = self._ar.unsqueeze(0) < self.suffix_lens.long().unsqueeze(1)   # [K, Smax]
        # Flat indices of the real tokens in the [K, Smax] grid when there is padding: the MoE runs only those
        # (a 54-row mixed step padded to 2048 once sent 110k tokens through the experts and OOMed on 3.4 GiB).
        # Built on the host from the lengths: nonzero() on the device waits for every kernel queued before it.
        padded = not self.packed and suffix_lens and sum(suffix_lens) < len(suffix_lens) * max(suffix_lens)
        self.tokens = to_device(real_tokens(suffix_lens), device) if padded else None
        self._bt_cache = {}                        # layer -> the wave's [K, maxb] table, built once
        self._wave_table = None                    # the one block table (a block id is the same in every layer)
        self._wave_pos = None                      # (scratch id, block ids [K, S], slots [K, S]) of the suffix tokens
        self.alloc_s = 0.0                         # host seconds in block allocation over the wave
        self._blocks_done = False

    def _ensure_blocks(self):
        """Reserve every row's blocks up front, once a wave (Python only, no GPU work; `alloc_s` totals it, the
        prefix cache's reclaim included)."""
        if self._blocks_done:
            return
        t0 = time.perf_counter()
        bs = next(p for p in self.pools if p is not None).block_size
        for b, cache in enumerate(self.caches):
            bt = cache.blocks
            need = (cache.seq_len + self._suffix[b] + bs - 1) // bs
            while len(bt) < need:
                bt.append(cache.alloc.alloc())
        self._blocks_done = True
        self.alloc_s += time.perf_counter() - t0

    def _wave_bt(self, layer_idx):
        """The wave's [K, maxb] block table (long), built at the first layer, and whether it existed already."""
        same = self._wave_table is not None
        if not same:
            self._ensure_blocks()
            self._wave_table = self._table().long()
        self._bt_cache[layer_idx] = self._wave_table
        return self._wave_table, same

    def _table(self):
        """[K, maxb] int32 block table — ONE host->device copy, not one per row."""
        rows = [c.blocks for c in self.caches]
        maxb = max((len(r) for r in rows), default=1)
        pad = self._scratch if self._scratch is not None else 0
        flat = [r + [pad] * (maxb - len(r)) for r in rows]
        return to_device(flat, self.device, torch.int32)

    def append(self, layer_idx, k_new, v_new):
        """One scatter for the whole wave. Was a Python loop over rows calling
        PagedKVCache.append, i.e. K host->device copies + K scatter launches PER LAYER — 34% of
        a K=8 prefill forward (probe_prefill_ctx_overhead_modal.py). Padding positions are
        pointed at the pool's scratch block so they cannot touch a real row's KV."""
        pool = self.pools[layer_idx]
        bs = pool.block_size
        bt, same = self._wave_bt(layer_idx)

        S = k_new.shape[2]
        scratch = self._scratch
        if not same or self._wave_pos is None or self._wave_pos[0] != scratch:
            # Where every row's suffix lands: the same for every layer (same prefix lengths, same block ids, scratch
            # blocks allocated in lockstep), so built once a wave from device tensors. The per-layer
            # torch.tensor(list) it replaces was a synchronous host->device copy: 41% of the scheduler thread's time
            # under long prompts sat in it, and the host could never run ahead of the GPU by more than one layer.
            abs_pos = self.prefix_lens.long().unsqueeze(1) + self._ar[:S]                     # [K, S]
            valid = self._valid[:, :S]
            bids = bt.gather(1, (abs_pos // bs).clamp_(max=bt.shape[1] - 1))
            if scratch is not None:
                bids = torch.where(valid, bids, bids.new_full((), scratch))
            slots = torch.where(valid, abs_pos % bs, torch.zeros_like(abs_pos))
            self._wave_pos = (scratch, bids, slots)
        _, bids, slots = self._wave_pos

        pool.k[bids, :, slots, :] = k_new.permute(0, 2, 1, 3).to(pool.k.dtype)   # [K, S, Hkv, D]
        pool.v[bids, :, slots, :] = v_new.permute(0, 2, 1, 3).to(pool.v.dtype)

        for b, cache in enumerate(self.caches):   # bookkeeping only — no GPU work
            cache.seq_lens[layer_idx] += self._suffix[b]

    def block_table_tensor(self, layer_idx):
        # append() already built it for this layer; rebuilt only when asked for again.
        bt = self._bt_cache.pop(layer_idx, None)
        if bt is None:
            return self._wave_bt(layer_idx)[0].to(torch.int32)
        return bt.to(torch.int32)


class _PackedCtx(_PrefillCtx):
    """The prefill context for one packed forward: every row's chunk concatenated into one row of `total` tokens,
    nothing padded. Attention is per row through FlashInfer's ragged batch prefill (or sdpa_prefill_packed); the
    K/V of token i land at its row's block for position prefix + offset. The padded wave cost rows x longest tokens of
    activations (a 62-row step padded to 8192 OOMed) and capped how many rows a step could carry."""

    packed = True

    def __init__(self, caches, pools, prefix_lens, suffix_lens, device, scratch=None):
        super().__init__(caches, pools, prefix_lens, suffix_lens, device, scratch)
        self.tokens = None                                  # nothing is padded: the MoE sees every token
        lens = self.suffix_lens.long()
        total = sum(suffix_lens)
        rows = torch.repeat_interleave(torch.arange(len(suffix_lens), device=device), lens, output_size=total)
        starts = torch.cumsum(lens, 0) - lens
        self._rows = rows                                                                      # [T] row of token i
        self._pos = self.prefix_lens.long()[rows] + torch.arange(total, device=device) - starts[rows]   # its position

    def append(self, layer_idx, k_new, v_new):
        """k/v_new [1, Hkv, T, D]: every token into its row's block, one scatter per pool."""
        pool = self.pools[layer_idx]
        bs = pool.block_size
        bt, same = self._wave_bt(layer_idx)
        if not same or self._wave_pos is None:
            self._wave_pos = (None, bt[self._rows, self._pos // bs], self._pos % bs)
        _, bids, slots = self._wave_pos
        k_t, v_t = k_new[0].transpose(0, 1), v_new[0].transpose(0, 1)                           # [T, Hkv, D]
        if fused._on(k_t, v_t) and pool.k.dtype in (torch.bfloat16, torch.float8_e4m3fn):    # cast in the kernel
            fused.kv_store_rows(pool.k, pool.v, bids, slots, k_t, v_t)
        else:
            pool.k[bids, :, slots, :] = k_t.to(pool.k.dtype)
            pool.v[bids, :, slots, :] = v_t.to(pool.v.dtype)
        for b, cache in enumerate(self.caches):
            cache.seq_lens[layer_idx] += self._suffix[b]


class _StateRow:
    """A decode row of a BatchedDecodeState seen as a prefill row: its block lists (host mirrors) and lengths, so the
    packed context places its one token like any other row's chunk. Allocation happened in prepare_step."""

    def __init__(self, state, r, n_layers):
        self.blocks = state._host_blocks[r]
        self.alloc = state.alloc
        self.block_tables = [self.blocks if p is not None else None for p in state.pools]
        self.seq_lens = [state._host_len[r]] * n_layers
        self.seq_len = state._host_len[r]


# Prefill chunks of at most this many tokens in total run as decode rows through the captured decode graphs (one row
# per token, all of a request's rows sharing its block table, row i at position prefix + i; every row writes its K/V
# before attention in each layer, so row i attends to exactly positions <= prefix + i: causal prefill). A 1-row,
# 12-token packed prefill was host-bound at ~57 ms (a Python launch stream of ~1000 kernels), and long_prompt, whose
# prompts are 98.7% prefix-cache hits, spent half its time in such steps (H200, 2026-10-19). Measured long_prompt:
# 20 req/s off, 33.3 at 64 tokens, 36.7 at 128 and at 256. 0 = off.
PREFILL_AS_DECODE_TOKENS = int(os.environ.get("CUSTOM_BACKEND_PREFILL_AS_DECODE_TOKENS", "128"))


def decode_rows(caches, chunks, pools):
    """Expand prefill chunks into decode rows: reserve every chunk's blocks (one id per block, for every layer), and
    return (tokens, positions, owner row per token, table [K, cols] of block ids, 0-padded). The caches' block lists
    grow here; their lengths move on only after the forward (`commit_rows`)."""
    bs = next(p for p in pools if p is not None).block_size
    tokens, pos, owner = [], [], []
    for k, (cache, chunk) in enumerate(zip(caches, chunks)):
        p0 = cache.seq_len
        need = (p0 + len(chunk) + bs - 1) // bs
        bt = cache.blocks
        while len(bt) < need:
            bt.append(cache.alloc.alloc())
        tokens += chunk
        pos += range(p0, p0 + len(chunk))
        owner += [k] * len(chunk)
    cols = max(len(c.blocks) for c in caches)
    table = [c.blocks + [0] * (cols - len(c.blocks)) for c in caches]
    return tokens, pos, owner, table


def commit_rows(caches, chunks, pools) -> None:
    """The chunks' K/V are written: every live layer's length moves on by the chunk."""
    for cache, chunk in zip(caches, chunks):
        for L, pool in enumerate(pools):
            if pool is not None:
                cache.seq_lens[L] += len(chunk)


def first_tokens(rows: torch.Tensor, samplings: list, defer: bool) -> list:
    """Sample one token per row of `rows` [K, V]: ints (one device sync), or DeviceTokens whose pinned copies land
    behind one event (`defer`, CUDA): the caller reads them after it has queued more work."""
    if not samplings:
        return []
    toks = sample_batched(rows, samplings)
    if not defer or not toks.is_cuda:
        return toks.tolist()
    host = torch.empty(toks.shape, dtype=toks.dtype, pin_memory=True)
    host.copy_(toks, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    return [DeviceToken(toks, host, j, ev) for j in range(len(samplings))]


class CustomTorchBackend(InferenceBackend):
    """PyTorch backend driven by our custom forward (Qwen3MoeForCausalLM)."""

    defers_first_tokens = True      # prefill_chunk_batch(defer_tokens=True) returns DeviceTokens for final rows

    THINK_START = 100   # replaced at load by the tokenizer's <think> / </think> ids
    THINK_END = 101

    # Each row's paged cache holds only its real tokens and the model derives causality
    # from per-row seq_lens, so the scheduler's [B, S] mask is never read.
    needs_attention_mask = False

    def __init__(self, device: str = "cuda", model_name: str = "Qwen/Qwen3-30B-A3B"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._eos_ids: set[int] = set()
        self._lock = threading.Lock()
        self.last_cache_hit_tokens = 0

    def load_model(self, model_name: str) -> None:
        import os
        from transformers import AutoConfig, AutoTokenizer
        from inference_server.models.paged_kv_cache import make_pools
        from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM

        cfg = AutoConfig.from_pretrained(model_name)
        if cfg.model_type != "qwen3_moe":
            raise ValueError(f"CustomTorchBackend supports qwen3_moe, not {cfg.model_type!r}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Qwen's think markers; ids 100/101 are ordinary characters in its vocab
        self.THINK_START, self.THINK_END = self.tokenizer.convert_tokens_to_ids(["<think>", "</think>"])

        # Shards stream straight onto the device; no HF model built in host memory first.
        self.model = Qwen3MoeForCausalLM.from_safetensors(model_name, device=self.device)
        self.max_positions = int(getattr(cfg.get_text_config(), "max_position_embeddings", 0) or 0)
        self._eos_ids = stop_token_ids(model_name, self.tokenizer)

        if os.environ.get("CUSTOM_BACKEND_FUSE_QKV", "1") == "1":
            for layer in self.model.model.layers:
                layer.self_attn.fuse_qkv()
        # fp8 experts by default on CUDA (CUSTOM_BACKEND_QUANT=bf16 keeps them bf16). The experts are ~97% of the
        # weight bytes; at batch 128 a decode step streams nearly all of them. H200 2026-10-19, with 256 decode rows:
        # saturated 3.9 -> 6.3 req/s, long_prompt 20 -> 20, single_stream 4.57 -> 4.79 ms; dev-tier gate PASS.
        quant = os.environ.get("CUSTOM_BACKEND_QUANT", "fp8" if self.device.type == "cuda" else "").lower()
        if quant == "fp8" and self.device.type == "cuda":
            for layer in self.model.model.layers:
                layer.mlp.experts.quantize_fp8()
            torch.cuda.empty_cache()
            logger.info("MoE experts quantized to fp8 e4m3 (per output channel; activations per token)")
        elif quant == "fp8":
            logger.warning("CUSTOM_BACKEND_QUANT=fp8 needs CUDA; ignoring")
        elif quant == "int8":
            logger.warning("CUSTOM_BACKEND_QUANT=int8 is not supported for qwen3_moe; ignoring")

        # The decode-step megakernel (models/megakernel): one cooperative kernel for the whole step, for decode steps of
        # up to MEGAKERNEL_MAX_ROWS rows; built before the pools are sized (the fp8 expert copies take ~29 GB).
        # Measured 2026-10-18 (H200, v1 per-layer kernel): it beat the graphs at batch 1 (4.1 vs 5.6 ms) and 2 (5.4 vs
        # 5.7); from 4 on both were bound by streaming ~30 distinct experts' weights and the graphs were faster.
        from inference_server.models import megakernel as mk
        self._mega_on = self.device.type == "cuda" and mk.ENABLED and mk.available()
        self._mega_max_rows = int(os.environ.get("MEGAKERNEL_MAX_ROWS", "2"))
        # A measured table of which path wins per row count and context (models/megakernel/routing.py); the rule above
        # where it has no entry.
        from inference_server.models.megakernel import routing
        self._mega_routing = (routing.load(os.environ.get("MEGAKERNEL_ROUTING", routing.DEFAULT), mk.BMAX)
                              if self._mega_on else None)
        self._mega = mk.Model(self.model, self.device) if self._mega_on else None
        if self._mega is not None:
            logger.info("decode-step megakernel on: batch <= %d, experts %s", self._mega_max_rows, "fp8" if self._mega.fp8 else "bf16")

        # A draft model for speculative decoding (SPEC_DRAFT_MODEL, drafter.py), loaded before the pools are sized: its
        # weights and its static cache (max_len = the context window) come out of the KV budget.
        draft_name = os.environ.get("SPEC_DRAFT_MODEL", "")
        self._draft = None
        if draft_name:
            from inference_server.config import settings
            from inference_server.drafter import DraftModel
            self._draft = DraftModel.load(draft_name, self.device, settings.context_window)
            logger.info("draft model %s loaded (static cache %d positions)", draft_name, settings.context_window)

        # The KV pools' element type: KV_CACHE_DTYPE=fp8 (CUDA) stores K/V as float8_e4m3fn with a unit scale, as
        # vLLM's uncalibrated fp8 cache does: twice the tokens per GB (96 KB a token in bf16 for this model) and half
        # the bytes each attention read streams; readers widen to bf16 before any product. Default bf16.
        kv = os.environ.get("KV_CACHE_DTYPE", "bf16").lower()
        self.kv_dtype = torch.float8_e4m3fn if (kv == "fp8" and self.device.type == "cuda") else None
        if kv == "fp8" and self.kv_dtype is None:
            logger.warning("KV_CACHE_DTYPE=fp8 needs CUDA; the pools stay in the model's dtype")

        # Pre-allocate paged block pools shared across sessions — see kv_reserve.
        from inference_server.config import settings
        from inference_server.models.paged_kv_cache import blocks_for_budget
        bsz = int(os.environ.get("CUSTOM_BACKEND_BLOCK_SIZE", "16"))
        if os.environ.get("CUSTOM_BACKEND_BLOCKS"):
            n_blocks = int(os.environ["CUSTOM_BACKEND_BLOCKS"])
        elif self.device.type == "cuda":
            # From what the weights leave free after the largest prefill wave the scheduler admits (its peak is
            # measured, not guessed: a fixed 512 blocks held ~8k tokens, and 90% of the raw free memory OOMed
            # on long prompts). The remaining share is for CUDA graphs and smaller transients.
            peak = self._prefill_peak_bytes(settings.max_prefill_tokens or self.max_positions or 4096)
            free, _ = torch.cuda.mem_get_info(self.device)
            n_blocks = blocks_for_budget(self.model, int((free - peak) * settings.kv_cache_memory_fraction), bsz,
                                         self.kv_dtype)
            logger.info("KV pools: %d blocks of %d tokens (%.0f%% of %.1f GiB free minus a %.1f GiB prefill peak "
                        "at %d tokens)", n_blocks, bsz, 100 * settings.kv_cache_memory_fraction, free / 2**30,
                        peak / 2**30, settings.max_prefill_tokens)
        else:
            n_blocks = 512
        self.pools = make_pools(self.model, num_blocks_per_pool=n_blocks, block_size=bsz, kv_dtype=self.kv_dtype)
        self._alloc = next(p for p in self.pools if p is not None).allocator      # one id space for every layer
        self._reserved = [0] * len(self.pools)  # per-pool blocks reserved by admitted requests
        if self._mega is not None:
            self._mega.bind_pools(self.pools)

        from inference_server.models.paged_kv_cache import PrefixCache, RadixPrefixCache
        cache_cls = RadixPrefixCache if settings.prefix_cache_impl == "radix" else PrefixCache
        self.prefix_cache = cache_cls(
            pools=self.pools,
            max_entries=settings.prefix_cache_max_entries,
            max_block_fraction=settings.prefix_cache_block_fraction,
        )

        # Tuned Triton launch table (scripts/bench/tune_triton_launch.py). Read here, before any
        # graph is captured, because a graph freezes its launch config. Unset = launch as before.
        launch_path = os.environ.get("CUSTOM_BACKEND_LAUNCH_TABLE", "")
        if launch_path and self.device.type == "cuda":
            from inference_server.models import launch_table
            gpu = torch.cuda.get_device_name(self.device)
            n = launch_table.load(launch_path, gpu, model_name)
            (logger.info if n else logger.warning)(
                "Triton launch table %s: %d entries for gpu=%r model=%r", launch_path, n, gpu, model_name)

        # CUDA-graph decode: one graph per ROW BUCKET, replay the smallest bucket >= n_rows.
        # A single max-batch graph padded every step to max_batch_size (64x waste at 4 active
        # rows with max_batch=256), which dominated TPOT at the low concurrency the SLO lives at.
        self._block_size = bsz
        self._graph_max_rows = settings.max_batch_size
        self._graph_max_cols = (settings.context_window + bsz - 1) // bsz
        # Mixed prefill as one packed forward, and decode rows join it (PREFILL_PACKED=0: the right-padded wave and a
        # separate decode step; on CPU/MPS packed is opt-in, for the tests).
        self.packed_prefill = os.environ.get("PREFILL_PACKED", "1" if self.device.type == "cuda" else "0") == "1"
        self._graph_on =self.device.type == "cuda" and \
            os.environ.get("CUSTOM_BACKEND_CUDA_GRAPH", "1") == "1"
        self._compile_on = self.device.type == "cuda" and \
            os.environ.get("CUSTOM_BACKEND_COMPILE", "0") == "1"      # MoE included since 2026-10-17 (an A/B)
        self._graph_buckets = _decode_buckets(self._graph_max_rows, coarse=self._compile_on)
        self._graphs: dict[int, dict] = {}   # bucket -> captured graph + its static buffers
        self._scratch = None                 # the padding block id, shared by every bucket (every pool)

        # torch.compile the DECODE forward only (Inductor fuses the ~50% element-wise/norm/RoPE
        # tail the profiler found — the bandwidth lever on A100). Decode-only because prefill's
        # ragged shapes would recompile-storm a shared compile; decode is the static [maxN,1]
        # graph shape. Default mode = fusion, no internal cudagraphs → composes with our manual
        # CUDA graph. The graph captures the fused kernel stream. CPU path stays uncompiled.
        if self._compile_on:
            # mode=None → default fusion (no internal cudagraphs, composes with our manual graph).
            # "max-autotune-no-cudagraphs" autotunes GEMMs + fuses harder; the -no-cudagraphs
            # variant is REQUIRED (plain max-autotune adds its own cudagraphs → conflicts capture).
            mode = os.environ.get("CUSTOM_BACKEND_COMPILE_MODE") or None
            # dynamic=None (automatic), NOT dynamic=False. Each bucket is a distinct static
            # shape, so dynamic=False pays a full Inductor compile per bucket (~92s each,
            # 553s for a 6-bucket ladder, ~828s for 9). With automatic dynamic shapes the
            # first bucket compiles static, the second recompiles with the batch dim
            # symbolic, and every later bucket reuses that artifact — measured at 1.8s each.
            # dynamic=True is NOT an option: it dies in Inductor codegen with
            # `NameError: name 's6' is not defined` (same class of bug HANDOFF hit on prefill).
            self._decode_fwd = torch.compile(self.model, mode=mode)
            logger.info("torch.compile enabled for decode forward (mode=%s)", mode or "default")
        else:
            self._decode_fwd = self.model

        # K=1 CUDA-graph prefill — replays the forward instead of paying ~73ms eager dispatch
        # (the unloaded-TTFT floor). Captured lazily per suffix bucket on first use (capture is
        # ~seconds, unlike torch.compile). Flag-gated OFF until verified; deploy unaffected.
        self._prefill_graph_on = self.device.type == "cuda" and \
            os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH", "0") == "1"
        self._prefill_graphs: dict = {}
        if self._graph_on:
            self._capture_all_graphs()       # at load: on the first decode it was a 3.5 s step for the first requests

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
                self.prefix_cache.store(token_ids, cache.blocks)
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
                self.prefix_cache.store(token_ids, cache.blocks)
            finally:
                cache.free_all()

    @torch.no_grad()
    def generate_step(
        self, token_ids: list[int], kv_cache=None,
        sampling: SamplingParams = GREEDY,
    ) -> tuple[int, object]:
        """One forward pass. Pass our KVCache to incremental-decode; pass None to do full prefill."""
        from inference_server.models.common import KVCache
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
        # lm_head on the last suffix position only — see Qwen3MoeForCausalLM.forward(logits_index).
        last = torch.tensor([len(suffix) - 1], device=self.device)
        try:
            logits = self.model(torch.tensor([suffix], device=self.device), kv_cache=cache,
                                logits_index=last)
        except BaseException:
            cache.free_all()            # do not abandon partially allocated blocks
            raise
        first_token = int(sample(logits[:, -1, :], GREEDY).item())
        self.prefix_cache.store(token_ids, cache.blocks)
        return cache, first_token, cache.seq_len

    @torch.no_grad()
    def _prefill_peak_bytes(self, tokens: int) -> int:
        """Peak memory above the weights of one prefill of `tokens` tokens, run through the plain (unpaged) forward
        with the logits of the last position only: what the KV pools must leave room for."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        ids = torch.zeros(1, tokens, dtype=torch.long, device=self.device)
        self.model(ids, logits_index=torch.tensor([tokens - 1], device=self.device))
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) - base
        torch.cuda.empty_cache()
        return peak

    @torch.no_grad()
    def prefill_batch(self, prompts, session_ids=None):
        """Prefix-cache-aware batched prefill — the TTFT lever. Per row: cache-lookup → seed the
        cached prefix → then ONE batched forward over the (much shorter) SUFFIXES, each attending to
        its own prefix. For repeated/shared prompts the suffix is ~1 token, so the forward is tiny
        (dispatch-bound) instead of re-prefilling the full prompt. No cache hit → suffix = full prompt
        (the v1 behaviour). Returns per-row (PagedKVCache, first_token, kv_len). Device-agnostic
        (gather + masked SDPA). Left-pad the prefix, right-pad the suffix, so each row's prefix+suffix
        keys are contiguous; a per-row mask keeps rows independent (suffix query attends to its real
        prefix keys + causal suffix keys).

        CUDA uses the gather-free paged prefill kernel instead (`_prefill_batch_kernel`)."""
        if self.device.type == "cuda":
            return self._prefill_batch_kernel(prompts)
        from inference_server.models.common import KVCache
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
        self.last_batch_cache_hit_tokens = list(matched)

        Smax = max(len(s) for s in suffixes)
        Pmax = max(matched) if matched else 0

        input_ids = torch.zeros(K, Smax, dtype=torch.long, device=dev)
        position_ids = torch.zeros(K, Smax, dtype=torch.long, device=dev)
        for k in range(K):
            s, m = suffixes[k], matched[k]
            input_ids[k, :len(s)] = torch.tensor(s, device=dev)                  # right-padded
            position_ids[k, :len(s)] = torch.arange(m, m + len(s), device=dev)

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
            real_q = (qi < L)[:, None]                                           # [Smax,1]
            prefix_key = ((ki < Pmax) & (ki >= Pmax - m))[None, :]               # [1,Ktot]
            suffix_key = (ki >= Pmax)[None, :] & ((ki[None, :] - Pmax) <= qi[:, None])
            mask[k, 0] = real_q & (prefix_key | suffix_key)

        logits = self.model(input_ids, position_ids=position_ids, kv_cache=seeded, attn_mask=mask)

        results = []
        for k in range(K):
            L = len(suffixes[k])
            first = int(sample(logits[k:k + 1, L - 1, :], GREEDY).item())
            cache = caches[k]
            for i in range(nlayers):
                kvi = seeded.get(i)
                if kvi is None:  # KV-shared layer — reads the source layer's pool
                    continue
                ki_t, vi_t = kvi  # [K,H,Pmax+Smax,D]; this row's real suffix = columns Pmax..Pmax+L
                cache.append(i, ki_t[k:k + 1, :, Pmax:Pmax + L, :], vi_t[k:k + 1, :, Pmax:Pmax + L, :])
            self.prefix_cache.store(prompts[k], cache.blocks)
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
            matched0, shared0 = self.prefix_cache.lookup(p)
            if matched0 >= len(p):
                matched0 = 0            # need >=1 suffix token to get next-token logits
                shared0 = None
            # `bucket` is None for suffixes past the largest bucket — those stay eager.
            bucket = _prefill_bucket(len(p) - matched0)
            if bucket is not None:
                if not self._prefill_graphs:
                    self._capture_all_prefill_graphs()
                if any(k[0] == bucket for k in self._prefill_graphs):
                    self.last_cache_hit_tokens = matched0
                    self.last_batch_cache_hit_tokens = [matched0]
                    return [self._replay_prefill(p, bucket, matched0, shared0)]
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
        self.last_batch_cache_hit_tokens = list(matched)

        Smax = max(len(s) for s in suffixes)
        input_ids = torch.zeros(len(prompts), Smax, dtype=torch.long, device=dev)
        position_ids = torch.zeros(len(prompts), Smax, dtype=torch.long, device=dev)
        for k, s in enumerate(suffixes):
            input_ids[k, :len(s)] = torch.tensor(s, device=dev)               # right-padded
            position_ids[k, :len(s)] = torch.arange(matched[k], matched[k] + len(s), device=dev)

        self._ensure_scratch()
        try:
            return self._prefill_batch_forward(prompts, caches, matched, suffixes,
                                               input_ids, position_ids, dev)
        except BaseException:
            for c in caches:            # same partial-allocation leak as the graphed path
                c.free_all()
            raise

    def _prefill_batch_forward(self, prompts, caches, matched, suffixes,
                               input_ids, position_ids, dev):
        ctx = _PrefillCtx(caches, self.pools, matched, [len(s) for s in suffixes], dev,
                          scratch=self._scratch)
        # One logit row per sequence, not one per token: the vocab projection over a K x Smax
        # right-padded wave is the single largest FLOP+bytes item in prefill and we read K rows.
        last_idx = torch.tensor([len(s) - 1 for s in suffixes], device=dev)
        logits = self.model(input_ids, position_ids=position_ids, paged_ctx=ctx,
                            logits_index=last_idx)

        results = []
        for k in range(len(prompts)):
            first = int(sample(logits[k:k + 1, 0, :], GREEDY).item())
            self.prefix_cache.store(prompts[k], caches[k].blocks)
            results.append((caches[k], first, len(prompts[k])))
        return results

    @torch.no_grad()
    def prefill_lookup(self, token_ids, session_id="default"):
        """Cache lookup only — no forward pass. Returns (seeded PagedKVCache, matched)."""
        from inference_server.models.paged_kv_cache import PagedKVCache
        matched, shared = self.prefix_cache.lookup(token_ids)
        if matched and matched >= len(token_ids):
            # Every token cached: the next token needs a forward over the last one, and feeding it again at position
            # len(ids) put it in the KV twice. Give back the last cached block; it is prefilled like any suffix.
            self._alloc.release(shared.pop())
            matched -= self._block_size
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

    @torch.no_grad()
    def prefill_chunk_batch(self, items, decode=None, defer_tokens=False):
        """`items`: [(partial_kv | None, chunk_ids, is_final, sampling)] -> [(kv, last_token | None, kv_len)]. One
        forward over every chunk: packed into one row (default), or right-padded (PREFILL_PACKED=0); each row attends
        to its own blocks so far plus the chunk. With `decode` = (decode batch, current tokens [R], sampling per row)
        the running decode rows join the packed forward with one token each and the result is (results, next tokens
        [R]): one step for prefill chunks and decode tokens, the weights streamed once for both. On CPU/MPS without
        the packed path, one prefill_chunk per row. On a failure every prefill row's blocks are freed: the scheduler
        rejects them all. `defer_tokens` (CUDA): final rows' tokens come back as DeviceTokens, no sync here."""
        from inference_server.models.paged_kv_cache import BatchedDecodeState, PagedKVCache
        if not self.packed_prefill and self.device.type != "cuda":
            out = []
            for kv, chunk, final, sampling in items:
                kv, tok, n = self.prefill_chunk(chunk, kv, sampling=sampling)
                out.append((kv, tok if final else None, n))
            return out
        dev = self.device
        n_tok = sum(len(c) for _, c, _, _ in items)
        if self.joint_fits(decode[0].n_rows if decode is not None else 0, n_tok, decode):
            out = self._prefill_as_decode(items, defer_tokens, decode)
            if out is not None:
                return out
        caches = [kv if kv is not None else PagedKVCache(pools=self.pools) for kv, _, _, _ in items]
        prefix = [c.seq_len for c in caches]
        chunks = [chunk for _, chunk, _, _ in items]
        lens = [len(c) for c in chunks]
        K = len(items)
        state = None
        if decode is not None:
            assert self.packed_prefill
            batched, current, sampling_rows = decode
            if isinstance(batched, BatchedDecodeState):
                state = batched
                state.prepare_step()                # any block a row crosses into, before its view is taken
                rows = [_StateRow(state, r, len(self.pools)) for r in range(state.n_rows)]
            else:
                rows = list(batched)                # CPU/MPS: the decode rows are PagedKVCache objects
            caches = caches + rows
            prefix = prefix + [r.seq_len for r in rows]
            lens = lens + [1] * len(rows)           # the decode rows' tokens join on the device (no .tolist() sync)
        if self.packed_prefill:                     # one row of every chunk, no padding; logits at each row's last token
            input_ids = to_device([t for c in chunks for t in c], dev)
            if decode is not None:
                input_ids = torch.cat([input_ids, current.view(-1).to(input_ids.device)])
            input_ids = input_ids.unsqueeze(0)
            position_ids = to_device([p + i for p, n in zip(prefix, lens) for i in range(n)], dev).unsqueeze(0)
            ends, acc = [], 0
            for n in lens:
                acc += n
                ends.append(acc - 1)
            last_idx = to_device(ends, dev)
        else:
            S = max(lens)
            input_ids = torch.zeros(len(items), S, dtype=torch.long, device=dev)
            position_ids = torch.zeros(len(items), S, dtype=torch.long, device=dev)
            for k, chunk in enumerate(chunks):
                input_ids[k, :len(chunk)] = torch.tensor(chunk, device=dev)
                position_ids[k, :len(chunk)] = torch.arange(prefix[k], prefix[k] + len(chunk), device=dev)
            last_idx = torch.tensor([n - 1 for n in lens], device=dev)
        self._ensure_scratch()
        t0 = time.perf_counter()
        cuda = dev.type == "cuda"
        ev0 = ev1 = None
        try:
            ctx_cls = _PackedCtx if self.packed_prefill else _PrefillCtx
            ctx = ctx_cls(caches, self.pools, prefix, lens, dev, scratch=self._scratch)
            if cuda:
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
            logits = self.model(input_ids, position_ids=position_ids, paged_ctx=ctx, logits_index=last_idx)
            if cuda:
                ev1.record()
        except BaseException:
            for c in caches[:K]:
                c.free_all()
            raise
        t1 = time.perf_counter()
        finals = [k for k, it in enumerate(items) if it[2]]
        rows = (logits[0] if self.packed_prefill else logits[:, 0])[to_device(finals, dev)] if finals else None
        toks = dict(zip(finals, first_tokens(rows, [items[k][3] for k in finals], defer_tokens)))
        out = [(caches[k], toks.get(k), caches[k].seq_len) for k in range(K)]
        next_tokens = None
        if decode is not None:
            next_tokens = sample_batched(logits[0, K:, :], sampling_rows)
            if state is not None:
                state.advance()                     # the token is written; lengths move on
        t2 = time.perf_counter()
        # Where a step's time went, for the timeline (`prefill_detail`): host work enqueueing the forward (incl. the
        # block allocations inside it), the forward's own device time, and the wait at the first sync.
        self.last_prefill_timing = {"host_enqueue_s": round(t1 - t0, 4), "alloc_s": round(ctx.alloc_s, 4),
                                    "sync_wait_s": round(t2 - t1, 4),
                                    "gpu_forward_s": (round(ev0.elapsed_time(ev1) / 1e3, 4)
                                                      if cuda and ev1.query() else None)}   # a step with no final row has not synced
        return (out, next_tokens) if decode is not None else out

    def joint_fits(self, n_rows: int, n_tokens: int, decode=None) -> bool:
        """Whether a step's prefill tokens (and `n_rows` running decode rows) fit one decode-graph replay."""
        from inference_server.models.paged_kv_cache import BatchedDecodeState
        if decode is not None and (not self.packed_prefill or not isinstance(decode[0], BatchedDecodeState)):
            return False                            # (a fallback from here is the packed joint forward)
        return (self._graph_on and 0 < n_tokens <= PREFILL_AS_DECODE_TOKENS
                and n_rows + n_tokens <= max(self._graph_buckets))

    @torch.no_grad()
    def _prefill_as_decode(self, items, defer_tokens=False, decode=None):
        """Short prefill chunks as one replay of the decode graph (`PREFILL_AS_DECODE_TOKENS`): the same result shape
        as prefill_chunk_batch. With `decode` (batched state, current tokens, sampling per row) the running decode rows
        ride in the same replay, ahead of the prefill rows: one forward per step, the weights streamed once (vLLM's
        mixed batch; separate steps streamed the experts twice). None when the rows do not fit the graphs' block
        tables (the caller runs the packed forward). On a failure every prefill row's blocks are freed."""
        from types import SimpleNamespace
        from inference_server.models.paged_kv_cache import PagedKVCache
        t0 = time.perf_counter()
        state, current, sampling_rows = decode if decode is not None else (None, None, None)
        caches = [kv if kv is not None else PagedKVCache(pools=self.pools) for kv, _, _, _ in items]
        chunks = [chunk for _, chunk, _, _ in items]
        bs = self._block_size
        if max((c.seq_len + len(ch) + bs - 1) // bs for c, ch in zip(caches, chunks)) > self._graph_max_cols:
            return None
        if state is not None:
            state.prepare_step()                    # any block a decode row crosses into
            if state.bt.shape[1] > self._graph_max_cols:
                return None
        try:
            tokens, pos, owner, table = decode_rows(caches, chunks, self.pools)
            if not self._graphs:
                self._capture_all_graphs()
            dev, n = self.device, len(tokens)
            tok_t, pos_t, own_t, tab_t = (to_device(v, dev) for v in (tokens, pos, owner, table))   # table [K, cols]
            bt = tab_t[own_t]                                                                 # [n, cols]
            nd = state.n_rows if state is not None else 0
            if nd:
                cols = max(bt.shape[1], state.bt.shape[1])
                bt = torch.cat([state._grow_cols(state.bt, cols), state._grow_cols(bt, cols)], dim=0)
                tok_t = torch.cat([current.view(-1), tok_t])
                pos_t = torch.cat([state.seq_lens, pos_t])
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            logits = self._replay_decode(tok_t.unsqueeze(1), pos_t.unsqueeze(1),
                                         SimpleNamespace(n_rows=nd + n, seq_lens=pos_t, bt=bt.contiguous()),
                                         self._decode_bucket(nd + n))
            ev1.record()
        except BaseException:
            for c in caches:
                c.free_all()
            raise
        commit_rows(caches, chunks, self.pools)
        t1 = time.perf_counter()
        next_tokens = None
        if nd:
            next_tokens = sample_batched(logits[:nd, -1], sampling_rows)
            state.advance()                         # the decode rows' tokens are written; lengths move on
        ends, end = [], nd - 1
        for _, chunk, _, _ in items:
            end += len(chunk)
            ends.append(end)
        finals = [k for k, it in enumerate(items) if it[2]]
        rows = logits[to_device([ends[k] for k in finals], self.device), -1] if finals else None
        toks = dict(zip(finals, first_tokens(rows, [items[k][3] for k in finals], defer_tokens)))
        out = [(caches[k], toks.get(k), caches[k].seq_len) for k in range(len(items))]
        t2 = time.perf_counter()
        self.last_prefill_timing = {"host_enqueue_s": round(t1 - t0, 4), "alloc_s": 0.0, "sync_wait_s": round(t2 - t1, 4),
                                    "gpu_forward_s": round(ev0.elapsed_time(ev1) / 1e3, 4) if ev1.query() else None,
                                    "as_decode": True, "decode_rows": nd}
        return (out, next_tokens) if decode is not None else out

    def prefill_store(self, token_ids, full_kv, matched, session_id="default"):
        """Store the prompt's full blocks for future sharing (PrefixCache dedupes by prefix)."""
        self.prefix_cache.store(token_ids, full_kv.blocks)

    @torch.no_grad()
    def decode_step_batched(self, current_tokens, batched_kv, attention_mask,
                            position_ids, sampling_per_row=None):
        """One batched forward over all rows. `attention_mask` (scheduler-built) is ignored;
        causality comes from per-row seq_lens inside the model.

        CUDA: `batched_kv` is a persistent `BatchedDecodeState` — block tables live as GPU
        tensors, so the per-step hot path is pure GPU (no Python rebuild). CPU/MPS: a
        `list[PagedKVCache]` gathered into a left-padded masked SDPA (portable reference)."""
        if self.device.type == "cuda":
            state = batched_kv
            n = state.n_rows
            if position_ids is None:                   # the state knows every row's position; no host round trip
                position_ids = state.seq_lens.unsqueeze(1)
            new = state.prepare_step(defer=True)  # alloc any boundary-crossing block (once per step)
            if self._graph_on and not self._graphs:
                self._capture_all_graphs()   # all buckets up-front, largest first (shared pool)
            # Rows past context_window outgrow the static block tables; run that step eagerly.
            fits = state.bt.shape[1] <= self._graph_max_cols
            greedy_tokens = None
            seq_next = None
            if self._mega_on and self._use_megakernel(state):
                if new is not None:
                    new.apply()
                logits, greedy_tokens = self._decode_megakernel(current_tokens, state)
            elif self._graph_on and fits:
                seq_next = torch.empty_like(state.seq_lens)            # staged by the same launch as the inputs
                logits = self._replay_decode(current_tokens, position_ids, state,
                                             self._decode_bucket(n), new=new, seq_next=seq_next)
            else:
                if new is not None:
                    new.apply()
                logits = self._decode_fwd(current_tokens, position_ids=position_ids, paged_ctx=state)
            state.advance(seq_next)       # bump seq_lens
        else:
            from inference_server.models.paged_kv_cache import BatchedPagedKVCache
            greedy_tokens = None
            rows = batched_kv
            n = len(rows)
            ctx = BatchedPagedKVCache(rows)
            seq_lens = [r.seq_len for r in rows]
            if position_ids is None:                   # each row's own position, not the batch's longest
                position_ids = torch.tensor([[L] for L in seq_lens], dtype=torch.long, device=self.device)
            lmax = max(seq_lens)
            mask = torch.zeros(n, 1, 1, lmax + 1, dtype=torch.bool, device=self.device)
            for i, L in enumerate(seq_lens):
                mask[i, 0, 0, lmax - L:] = True
            logits = self.model(current_tokens, position_ids=position_ids, kv_cache=ctx, attn_mask=mask)

        last = logits[:, -1, :]  # [N, V]
        if greedy_tokens is not None and (sampling_per_row is None or all(p.temperature <= 0.0 for p in sampling_per_row)):
            next_tokens = greedy_tokens          # the kernel's own argmax: no sampling kernels between cooperative launches
        elif sampling_per_row is None:
            next_tokens = sample(last, GREEDY)
        else:
            next_tokens = sample_batched(last, sampling_per_row)
        return next_tokens, batched_kv

    # --- speculative decoding: one forward over [current] + drafts as rows at consecutive positions ---

    @torch.no_grad()
    def verify_step(self, batched_kv, r: int, tokens: list[int]) -> list[int]:
        """Row r of the batch fed `tokens` at positions len..len+n-1 in one forward (each token a decode row sharing
        the row's block table, K/V written before attention: causal), returning the greedy next token at each
        position. The row's length does not move: `commit_verify` advances it by what the caller accepted; the K/V of
        rejected positions are overwritten by later steps."""
        from types import SimpleNamespace
        from inference_server.models import megakernel as mk
        n = len(tokens)
        if self.device.type != "cuda":                     # CPU/MPS: the row's PagedKVCache through the model
            cache = batched_kv[r]
            p0 = cache.seq_len
            logits = self.model(torch.tensor([tokens], device=self.device), kv_cache=cache)
            out = logits[0].float().argmax(-1).tolist()
            cache.seq_lens = [p0 if L == 0 else L for L in cache.seq_lens]    # back to p0: commit_verify sets the length
            cache._verify_p0 = p0
            return out
        state = batched_kv
        state.reserve(r, n)
        p0 = state._host_len[r]
        pos = torch.arange(p0, p0 + n, device=self.device)
        bt = state.bt[r:r + 1].expand(n, -1).contiguous()
        tok_t = to_device(tokens, self.device)
        if self._mega_on and n <= self._mega_max_rows:
            _, greedy = mk.decode_step(self._mega, tok_t, pos, bt)
        else:
            if self._graph_on and not self._graphs:
                self._capture_all_graphs()
            logits = self._replay_decode(tok_t.unsqueeze(1), pos.unsqueeze(1),
                                         SimpleNamespace(n_rows=n, seq_lens=pos, bt=bt), self._decode_bucket(n))
            greedy = logits[:, -1].argmax(-1)
        return greedy.tolist()

    def new_drafter(self, ids):
        """A drafter for a request with context `ids` from the draft model (SPEC_DRAFT_MODEL), or None."""
        if getattr(self, "_draft", None) is None:
            return None
        from inference_server.drafter import ModelDrafter
        return ModelDrafter(self._draft, ids)

    def commit_verify(self, batched_kv, r: int, accepted: int) -> None:
        """Row r's length moves on by the `accepted` tokens of its last verify_step."""
        if self.device.type != "cuda":
            cache = batched_kv[r]
            p0 = getattr(cache, "_verify_p0", cache.seq_len)
            cache.seq_lens = [p0 + accepted if p is not None else 0 for p in cache.pools]
            return
        batched_kv.advance_row(r, accepted)

    # --- the decode-step megakernel (MEGAKERNEL=1, batch <= 16): one cooperative kernel for the whole step ---

    def _use_megakernel(self, state) -> bool:
        """The routing table's path for this step's rows and longest context, else n <= MEGAKERNEL_MAX_ROWS."""
        n, table = state.n_rows, self._mega_routing
        path = table.path(n, state.max_len()) if table is not None and n in table.rows else None
        return path == "megakernel" if path is not None else n <= self._mega_max_rows

    def _decode_megakernel(self, current_tokens, state):
        """One launch: embedding, every layer (each writes its K/V for the new token), final norm, lm_head, greedy
        argmax -> (logits [B, 1, V], tokens [B]). Nothing else is launched around it: a cooperative kernel overlaps no
        other kernel, so each small launch beside it cost ~70 us of idle GPU (measured 2026-10-18)."""
        from inference_server.models import megakernel as mk
        logits, tokens = mk.decode_step(self._mega, current_tokens.view(-1), state.seq_lens, state.bt)
        return logits.unsqueeze(1), tokens

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
        self._ensure_scratch()
        g_tokens = torch.zeros(bucket, 1, dtype=torch.long, device=dev)
        g_pos = torch.zeros(bucket, 1, dtype=torch.long, device=dev)
        g_seqlens = torch.ones(bucket, dtype=torch.long, device=dev)
        # One int32 table for every layer (a block id is the same in each layer's pool), refreshed by the staging launch.
        g_bt = torch.full((bucket, cols), self._scratch, dtype=torch.int32, device=dev)
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
            self._graphs[bucket] = {"graph": graph, "tokens": g_tokens, "pos": g_pos, "seqlens": g_seqlens,
                                    "bt": g_bt, "out": out}
            print(f"[graphs]   bucket {bucket} captured in {time.perf_counter() - t_b:.1f}s",
                  flush=True)
        except Exception:
            logger.exception("CUDA graph capture failed — falling back to eager decode")
            self._graph_on = False
            self._graphs.clear()

    def _ensure_scratch(self) -> None:
        """One padding block (the same id in every pool), shared by every captured graph (decode + prefill).
        Dummy/padding rows point here. Held for the process — never released."""
        if self._scratch is not None:
            return
        self._scratch = self._alloc.alloc()

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

    def _replay_decode(self, current_tokens, position_ids, state, bucket, new=None, seq_next=None):
        """Stage this step's inputs into `bucket`'s static buffers (real rows + scratch dummies) in one launch,
        replay that bucket's graph, return logits[:n_rows]. The graph reads the static buffers,
        so its scatter/kernel run on the fresh values — no per-op dispatch. `new`, `seq_next`: as
        fused.stage_decode's (a block allocated this step; the lengths after it)."""
        n = state.n_rows
        g = self._graphs[bucket]
        fused.stage_decode(g, current_tokens, position_ids, state.seq_lens, state.bt, n, self._scratch,
                           new=(new.blk, new.col) if new is not None else None, seq_next=seq_next)
        g["graph"].replay()
        return g["out"][:n]

    def _capture_prefill_graph(self, bucket: int, tiled: bool | None = None) -> None:
        """Capture the prefill forward once over static [1, bucket] buffers. Scratch blocks back the
        block tables during capture; replay copies in the real block IDs + tokens.

        The block table is sized for the whole context, not just the bucket, because with a
        prefix-cache hit the write positions start at prefix_len rather than 0.

        Captured under a SPECIFIC kernel variant. A graph freezes whichever kernel was active at
        capture, so a runtime flag can never change a replay — the only way to A/B a graphed path
        is to capture one graph per variant and choose which to replay."""
        from inference_server.models import paged_attention_kernel as _K
        variant = _K._TILED_PREFILL if tiled is None else tiled
        prev, _K._TILED_PREFILL = _K._TILED_PREFILL, variant
        dev, bs = self.device, self._block_size
        cols = self._graph_max_cols
        ids = torch.zeros(1, bucket, dtype=torch.long, device=dev)
        pos = torch.arange(bucket, device=dev).unsqueeze(0)
        prefix_lens = torch.zeros(1, dtype=torch.int32, device=dev)
        suffix_lens = torch.full((1,), bucket, dtype=torch.int32, device=dev)   # all real during capture
        logits_index = torch.full((1,), bucket - 1, dtype=torch.long, device=dev)
        self._ensure_scratch()
        bt = torch.full((1, cols), self._scratch, dtype=torch.long, device=dev)
        ctx = _PrefillGraphCtx(self.pools, bt, prefix_lens, suffix_lens, bucket, bs)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.model(ids, position_ids=pos, paged_ctx=ctx, logits_index=logits_index)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(ids, position_ids=pos, paged_ctx=ctx, logits_index=logits_index)
        # EVERY tensor allocated BEFORE capture that the graph reads must stay referenced.
        # Tensors allocated DURING capture live in the graph's private pool and are safe; these
        # do not. `pos`, `prefix_lens` and `ctx` (which owns ctx._pos, the scatter index) were
        # dropped here, so the allocator recycled their addresses and a later replay read
        # garbage -> huge q_pos -> out-of-bounds block-table load -> illegal memory access.
        # That is why the FIRST replay always succeeded and a later one crashed.
        _K._TILED_PREFILL = prev
        self._prefill_graphs[(bucket, variant)] = {
                                       "graph": g, "ids": ids, "suffix_lens": suffix_lens,
                                        "bt": bt, "out": out,
                                        "pos": pos, "prefix_lens": prefix_lens, "ctx": ctx,
                                        "logits_index": logits_index}
        logger.info("Captured CUDA prefill graph (bucket=%d, tiled=%s)", bucket, variant)

    def _capture_all_prefill_graphs(self) -> None:
        """Capture every prefill bucket at once, on first use.

        Capturing lazily per bucket meant the first request to reach each bucket ate a
        multi-second stall mid-serving: it showed up as TTFT p95 730ms at rate 1 (where p95 is
        essentially the single worst request) while p50 was 56ms. Prefill graphs do not go
        through torch.compile — capture is seconds, not minutes — so taking it all at once
        during warmup is cheap and makes the tail honest."""
        import time
        from inference_server.models import paged_attention_kernel as _K

        # By default capture only the active variant. PREFILL_GRAPH_VARIANTS=both captures each
        # bucket twice (tiled and untiled) so the GRAPHED path — the production one — can be
        # A/B'd by flipping the flag between arms. Costs ~0.5s and one graph pool per extra
        # capture, so it is worth it only while running an experiment.
        want = ([False, True] if os.environ.get("CUSTOM_BACKEND_PREFILL_GRAPH_VARIANTS") == "both"
                else [_K._TILED_PREFILL])
        t0 = time.perf_counter()
        for b in _PREFILL_BUCKETS:
          for variant in want:
            if (b, variant) not in self._prefill_graphs:
                try:
                    self._capture_prefill_graph(b, tiled=variant)
                except Exception:
                    logger.exception("prefill graph capture failed (bucket=%d) — eager fallback", b)
        print(f"[graphs] captured {len(self._prefill_graphs)} prefill graphs "
              f"{sorted(self._prefill_graphs)} in {time.perf_counter() - t0:.1f}s", flush=True)

    def _replay_prefill(self, prompt_ids, bucket, matched=0, shared=None):
        """K=1 prefill via graph replay, with or without a prefix-cache hit.

        Seeds the cached prefix blocks, allocates blocks for the suffix, stages everything into
        this bucket's static buffers, replays. Returns (cache, first_token, kv_len)."""
        from inference_server.models.paged_kv_cache import PagedKVCache
        S = len(prompt_ids) - matched
        from inference_server.models import paged_attention_kernel as _K
        key = (bucket, _K._TILED_PREFILL)
        if key not in self._prefill_graphs:                 # only one variant was captured
            key = next(k for k in self._prefill_graphs if k[0] == bucket)
        g = self._prefill_graphs[key]
        cache = PagedKVCache(pools=self.pools, shared_prefix=shared, shared_prefix_tokens=matched)
        try:
            return self._replay_prefill_inner(cache, g, prompt_ids, bucket, matched, S)
        except BaseException:
            # Blocks allocated before the failure would otherwise be abandoned with the local
            # cache. Under KV pressure alloc() raises MID-LOOP, so every rejected request used
            # to leak whatever it had already taken — the pool drained to zero and stayed there.
            cache.free_all()
            raise

    def _replay_prefill_inner(self, cache, g, prompt_ids, bucket, matched, S):
        dev, bs = self.device, self._block_size
        total = matched + S
        need = (total + bs - 1) // bs                     # only the blocks this prompt REALLY uses
        bt = cache.blocks                                 # already holds the shared prefix blocks
        while len(bt) < need:
            bt.append(cache.alloc.alloc())
        for L, pool in enumerate(self.pools):
            if pool is not None:
                cache.seq_lens[L] = total
        # Reset the whole static row to scratch FIRST. The graph writes all `bucket`
        # positions, so the entries past this prompt's real blocks must point somewhere
        # harmless — otherwise padding K/V lands in whichever request used this buffer last.
        g["bt"][0].fill_(self._scratch)
        g["bt"][0, :len(bt)].copy_(torch.tensor(bt, device=dev))
        g["ids"].zero_()
        g["ids"][0, :S].copy_(torch.tensor(prompt_ids[matched:], device=dev))
        g["pos"][0].copy_(torch.arange(matched, matched + bucket, device=dev))
        g["prefix_lens"].fill_(matched)
        g["suffix_lens"].fill_(S)                                               # padding queries skipped
        g["logits_index"].fill_(S - 1)                                          # lm_head on that row only
        g["graph"].replay()
        first = int(sample(g["out"][:, 0, :], GREEDY).item())
        self.prefix_cache.store(prompt_ids, cache.blocks)
        return cache, first, matched + S

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
        return self.remove_rows_from_cache(batched_kv, [row_idx])

    def remove_rows_from_cache(self, batched_kv, row_idxs):
        """Drop rows and release their blocks: on CUDA one compaction of the batched state, no stream sync."""
        if self.device.type == "cuda":
            batched_kv.remove_rows(row_idxs)
        else:
            for idx in sorted(row_idxs, reverse=True):
                batched_kv.pop(idx).free_all()
        return batched_kv

    def kv_length(self, kv):
        """Longest row's seq_len (used only for the scheduler's mask bookkeeping)."""
        if self.device.type == "cuda":
            return int(kv.seq_lens.max().item()) if kv.n_rows else 0
        return max((c.seq_len for c in kv), default=0)

    # --- Per-pool KV admission ---

    def _kv_footprints(self, prompt_len: int, max_tokens: int) -> list[int]:
        """Worst-case blocks this request occupies in each pool (prompt + max_tokens, rounded up)."""
        tokens = prompt_len + max_tokens
        return [0 if pool is None else (tokens + pool.block_size - 1) // pool.block_size
                for pool in self.pools]

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

    def kv_capacity_error(self, prompt_len: int, max_tokens: int) -> str | None:
        for fp, pool in zip(self._kv_footprints(prompt_len, max_tokens), self.pools):
            if pool is not None and fp > pool.num_blocks:
                needed = min(prompt_len + max_tokens, fp * pool.block_size)
                return (f"request needs {needed} KV tokens (prompt + max_tokens) but the cache "
                        f"holds {pool.num_blocks * pool.block_size}")
        return None

    @torch.inference_mode()
    def score_logprobs(self, token_ids: list[int], top_k: int = 20) -> list[dict[int, float] | None]:
        """Teacher-forced prompt logprobs in vLLM's shape (entry 0 None): one plain forward under the lock, no cache or graphs."""
        if len(token_ids) < 2:
            return [None] * len(token_ids)
        limit = getattr(self, "max_positions", 0)
        if limit and len(token_ids) > limit:
            raise ValueError(f"{len(token_ids)} tokens exceed the model's {limit} positions")
        ids = torch.tensor([token_ids], device=self.device)
        with self._lock:
            logits = self.model(ids)[0, :-1]          # row i predicts token i + 1
        out: list[dict[int, float] | None] = [None]
        for s in range(0, logits.shape[0], 512):
            lp = torch.log_softmax(logits[s:s + 512].float(), dim=-1)
            nxt = ids[0, s + 1:s + 1 + lp.shape[0]].tolist()        # one sync per chunk, not per row
            tok_lp = lp.gather(1, torch.tensor(nxt, device=lp.device)[:, None])[:, 0].tolist()
            top_v, top_i = lp.topk(min(top_k, lp.shape[-1]), dim=-1)
            for r, (vs, ix) in enumerate(zip(top_v.tolist(), top_i.tolist())):
                d = dict(zip(ix, vs))
                d[nxt[r]] = tok_lp[r]
                out.append(d)
        return out

    def is_eos(self, token_id: int) -> bool:
        return token_id in self._eos_ids

    @property
    def device_str(self) -> str:
        return str(self.device)
