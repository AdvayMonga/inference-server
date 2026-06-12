"""Block-paged KV cache for the custom Gemma 4 forward (M2.2).

Drop-in replacement for `KVCache`: same `get(layer_idx)` / `append(layer_idx, k, v)`
/ `seq_len` interface. Difference: storage is per-layer block pools, each session
holds a block_table of physical block indices.

What this enables:
  - Multiple sessions share a single block pool (no contiguous-allocation overhead).
  - Future: physical blocks can be ref-counted, allowing prefix sharing across sessions
    via the radix tree (M2.2b).
  - Future: custom attention kernel reads K/V via the block table directly (M2.3).

What this doesn't do yet:
  - Eviction / OOM handling (we raise on pool exhaustion).
  - Cross-session sharing — block_tables are per-session, no ref counting yet.

Storage layout per pool: `[num_blocks, num_kv_heads, block_size, head_dim]`
(matches the existing `BlockManager` layout for forward compat).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BlockPool:
    """Physical block pool for one layer's K/V. Shared across sessions, ref-counted."""

    def __init__(
        self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
        dtype: torch.dtype, device: torch.device, window: int | None = None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window = window  # sliding-window size for this layer; None = full attention (never evicts)
        self.k = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self._refcounts: list[int] = [0] * num_blocks
        self._free: list[int] = list(reversed(range(num_blocks)))

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError(f"BlockPool exhausted ({self.num_blocks} blocks)")
        bid = self._free.pop()
        self._refcounts[bid] = 1
        return bid

    def acquire(self, block_idx: int) -> None:
        """Add a reference to a block already owned by something else (cache / other session)."""
        assert self._refcounts[block_idx] > 0, "acquiring a freed block"
        self._refcounts[block_idx] += 1

    def release(self, block_idx: int) -> None:
        """Drop a reference; return to free list when last reference drops."""
        self._refcounts[block_idx] -= 1
        if self._refcounts[block_idx] == 0:
            self._free.append(block_idx)

    @property
    def free_count(self) -> int:
        return len(self._free)


class PrefixCache:
    """Maps aligned token-prefixes → per-layer block ids. Shares blocks across sessions
    via refcount: storing acquires; eviction releases. Sharing happens only on full-block
    boundaries (partial blocks aren't safe to share — slots after seq_len aren't written).
    """

    def __init__(self, pools: list[BlockPool | None]):
        self.pools = pools
        self.block_size = next(p.block_size for p in pools if p is not None)
        # key = tuple of leading tokens aligned to block boundary
        # value = dict[layer_idx] -> list of block ids (one per full block)
        self.entries: dict[tuple[int, ...], dict[int, list[int]]] = {}
        self.hits = 0
        self.lookups = 0

    def lookup(self, token_ids: list[int]) -> tuple[int, dict[int, list[int]]]:
        """Return (matched_tokens, per_layer_block_ids). Acquires blocks for the caller."""
        self.lookups += 1
        n_full_blocks = len(token_ids) // self.block_size
        # Walk back from longest aligned prefix.
        for n in range(n_full_blocks, 0, -1):
            key = tuple(token_ids[: n * self.block_size])
            if key in self.entries:
                self.hits += 1
                layer_blocks = self.entries[key]
                # The caller becomes an additional owner.
                for layer_idx, bids in layer_blocks.items():
                    for bid in bids:
                        self.pools[layer_idx].acquire(bid)
                return n * self.block_size, {k: list(v) for k, v in layer_blocks.items()}
        return 0, {}

    def store(self, token_ids: list[int], block_tables: list[list[int]]) -> None:
        """Store the session's fully-written blocks under its aligned prefix."""
        n_full_blocks = len(token_ids) // self.block_size
        if n_full_blocks == 0:
            return
        key = tuple(token_ids[: n_full_blocks * self.block_size])
        if key in self.entries:
            return  # already cached
        entry: dict[int, list[int]] = {}
        for layer_idx, bids in enumerate(block_tables):
            if self.pools[layer_idx] is None:
                continue
            # Only first n_full_blocks of this layer's table are guaranteed full
            full_bids = bids[:n_full_blocks]
            entry[layer_idx] = list(full_bids)
            # Cache acquires its own ref so blocks survive after the session ends.
            for bid in full_bids:
                self.pools[layer_idx].acquire(bid)
        self.entries[key] = entry


class PagedKVCache:
    """Per-session, per-layer block tables. Shares pools with other sessions.

    Construct with `shared_prefix={layer_idx: [bid, ...]}` and `shared_prefix_tokens`
    to seed the cache from a `PrefixCache.lookup()` hit. The caller is responsible for
    skipping forward computation on the prefix-matched portion of the prompt.
    """

    def __init__(
        self, pools: list[BlockPool | None],
        shared_prefix: dict[int, list[int]] | None = None,
        shared_prefix_tokens: int = 0,
    ):
        self.pools = pools
        self.num_layers = len(pools)
        self.block_tables: list[list[int]] = [[] for _ in range(self.num_layers)]
        self.seq_lens: list[int] = [0] * self.num_layers
        # Per-layer count of evicted (freed, out-of-window) leading blocks. Those slots in
        # block_tables hold -1; the decode kernel's start_b skips them so they're never read.
        self.evicted: list[int] = [0] * self.num_layers
        if shared_prefix:
            for layer_idx, bids in shared_prefix.items():
                self.block_tables[layer_idx] = list(bids)
                self.seq_lens[layer_idx] = shared_prefix_tokens

    @property
    def any_evicted(self) -> bool:
        """True if any sliding layer has freed an out-of-window block (prompt/context > window).
        Used to skip prefix-cache storage — we only cache prefixes that fit fully in the window."""
        return any(e > 0 for e in self.evicted)

    def _evict(self, layer_idx: int) -> None:
        """Free blocks that have fully fallen out of this layer's sliding window."""
        pool = self.pools[layer_idx]
        if pool is None or pool.window is None:
            return
        bs = pool.block_size
        keep_from = max(0, self.seq_lens[layer_idx] - pool.window) // bs  # first in-window logical block
        bt = self.block_tables[layer_idx]
        for i in range(self.evicted[layer_idx], keep_from):
            if bt[i] >= 0:
                pool.release(bt[i])
                bt[i] = -1
        if keep_from > self.evicted[layer_idx]:
            self.evicted[layer_idx] = keep_from

    @property
    def seq_len(self) -> int:
        """Length of any populated layer's sequence (all source layers stay in sync)."""
        for L in self.seq_lens:
            if L > 0:
                return L
        return 0

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Materialize K, V views by gathering blocks. Shape: [1, H, seq_len, D]."""
        seq_len = self.seq_lens[layer_idx]
        if seq_len == 0:
            return None
        pool = self.pools[layer_idx]
        bt = self.block_tables[layer_idx]
        # Gather all blocks → [N_blocks, H, block_size, D]. Evicted slots (-1) map to block 0;
        # their positions are out-of-window and get masked out by the caller's window mask.
        idx = torch.tensor([b if b >= 0 else 0 for b in bt], device=pool.k.device, dtype=torch.long)
        k_gathered = pool.k.index_select(0, idx)
        v_gathered = pool.v.index_select(0, idx)
        # Flatten block dim into S → [H, N_blocks * block_size, D]
        N_blocks, H, BS, D = k_gathered.shape
        k_flat = k_gathered.permute(1, 0, 2, 3).reshape(H, N_blocks * BS, D)
        v_flat = v_gathered.permute(1, 0, 2, 3).reshape(H, N_blocks * BS, D)
        # Slice to true seq_len, add batch dim → [1, H, seq_len, D]
        return k_flat[:, :seq_len, :].unsqueeze(0), v_flat[:, :seq_len, :].unsqueeze(0)

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Write k_new, v_new (shape [1, H, S_new, D]) into pool blocks. Allocates blocks lazily."""
        pool = self.pools[layer_idx]
        block_size = pool.block_size
        seq_len = self.seq_lens[layer_idx]
        S_new = k_new.shape[2]
        bt = self.block_tables[layer_idx]

        # Make sure we have enough blocks for (seq_len + S_new) tokens.
        total_blocks_needed = (seq_len + S_new + block_size - 1) // block_size
        while len(bt) < total_blocks_needed:
            bt.append(pool.alloc())

        # Per-token write loop. Slow at prefill (M2.3 fuses this into the attention kernel).
        # k_new shape [1, H, S_new, D] → squeeze batch, transpose to [S_new, H, D]
        k_per_tok = k_new[0].transpose(0, 1)
        v_per_tok = v_new[0].transpose(0, 1)
        for t in range(S_new):
            global_pos = seq_len + t
            bid = bt[global_pos // block_size]
            slot = global_pos % block_size
            pool.k[bid, :, slot, :] = k_per_tok[t]
            pool.v[bid, :, slot, :] = v_per_tok[t]

        self.seq_lens[layer_idx] = seq_len + S_new
        self._evict(layer_idx)  # free blocks now fully out of the sliding window

    def free_all(self) -> None:
        """Release this session's references; blocks return to pool when refcount hits 0."""
        for layer_idx in range(self.num_layers):
            pool = self.pools[layer_idx]
            if pool is None:
                continue
            for bid in self.block_tables[layer_idx]:
                if bid >= 0:  # evicted slots (-1) were already released
                    pool.release(bid)
            self.block_tables[layer_idx] = []
            self.seq_lens[layer_idx] = 0
            self.evicted[layer_idx] = 0


class BatchedPagedKVCache:
    """Transient view over N per-row `PagedKVCache`s for a single batched decode forward.

    Presents the model's `get(layer)` / `append(layer)` interface but operates on all rows
    at once. `get` LEFT-pads each row's gathered K/V to the batch max length and stacks to
    `[N, H, Lmax, D]` (right-aligned, so the caller's bool mask is the same for every layer —
    see `CustomTorchBackend.decode_step_batched`). `append` scatters each row's new token
    back into its own blocks. Holds references only; build a fresh one per decode step.
    """

    def __init__(self, rows: list[PagedKVCache]):
        self.rows = rows

    @property
    def seq_len(self) -> int:
        return max((r.seq_len for r in self.rows), default=0)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        mats = [r.get(layer_idx) for r in self.rows]
        if mats[0] is None:  # KV-shared layer — all rows have no own K/V here
            return None
        lmax = max(k.shape[2] for k, _ in mats)
        ks, vs = [], []
        for k, v in mats:
            pad = lmax - k.shape[2]  # left-pad the seq dim → right-align real tokens
            if pad > 0:
                k = F.pad(k, (0, 0, pad, 0))
                v = F.pad(v, (0, 0, pad, 0))
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)  # [N, H, Lmax, D]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Decode scatter: write each row's single new token (k_new,v_new [N,H,1,D]) into its
        own block. ONE vectorized indexed write for the whole batch (vs a per-row loop +
        per-row launches). Block-boundary allocation stays in Python — rare, no tensor writes."""
        pool = self.pools_for(layer_idx)
        bs = pool.block_size
        block_ids, slots = [], []
        for row in self.rows:
            L = row.seq_lens[layer_idx]
            bt = row.block_tables[layer_idx]
            if L // bs >= len(bt):          # crossed into a new block this step
                bt.append(pool.alloc())
            block_ids.append(bt[L // bs])
            slots.append(L % bs)
            row.seq_lens[layer_idx] = L + 1
        dev = pool.k.device
        bidx = torch.tensor(block_ids, dtype=torch.long, device=dev)
        sidx = torch.tensor(slots, dtype=torch.long, device=dev)
        pool.k[bidx, :, sidx, :] = k_new[:, :, 0, :]   # [N,H,D] — rows own distinct (block,slot)
        pool.v[bidx, :, sidx, :] = v_new[:, :, 0, :]
        for row in self.rows:                          # free blocks now out of the sliding window
            row._evict(layer_idx)

    def pools_for(self, layer_idx: int):
        return self.rows[0].pools[layer_idx]

    def block_table_tensor(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row block table [N, max_blocks] (int32, 0-padded) + seq_lens [N] for the kernel.
        Built with one host→device copy each (one padded list) instead of per-row tensors."""
        device = self.pools_for(layer_idx).k.device
        bts = [r.block_tables[layer_idx] for r in self.rows]
        seq_lens = [r.seq_lens[layer_idx] for r in self.rows]
        max_blocks = max((len(b) for b in bts), default=1) or 1
        # Evicted slots (-1) map to 0 — the kernel's start_b skips them, so they're never read.
        padded = [[(x if x >= 0 else 0) for x in b] + [0] * (max_blocks - len(b)) for b in bts]
        bt = torch.tensor(padded, dtype=torch.int32, device=device)
        sl = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        return bt, sl


def make_pools_for_gemma(
    model, num_blocks_per_pool: int = 1024, block_size: int = 16,
    sliding_blocks: int | None = None,
) -> list[BlockPool | None]:
    """Build the BlockPool list matching a GemmaForCausalLM's heterogeneous layers.

    Shared layers get None (they don't own K/V). Full-attention pools are sized to
    `num_blocks_per_pool` (they grow with sequence length). Sliding pools are capped at the
    window, so they can be sized smaller via `sliding_blocks` — freeing GPU memory to enlarge
    the full pools (the binding constraint for long-context concurrency). Default: same size.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if sliding_blocks is None:
        sliding_blocks = num_blocks_per_pool
    pools: list[BlockPool | None] = []
    for layer in model.model.layers:
        attn = layer.self_attn
        if attn.is_kv_shared:
            pools.append(None)
        else:
            is_sliding = attn.sliding_window is not None
            pools.append(BlockPool(
                num_blocks=sliding_blocks if is_sliding else num_blocks_per_pool,
                block_size=block_size,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                dtype=dtype,
                device=device,
                window=attn.sliding_window,  # sliding layers evict out-of-window blocks; full layers don't
            ))
    return pools
