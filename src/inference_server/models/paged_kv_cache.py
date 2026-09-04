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

from collections import OrderedDict

import torch
import torch.nn.functional as F


class KVCacheExhausted(RuntimeError):
    """No KV block available even after reclaiming the prefix cache.

    A typed error so the scheduler can treat it as BACKPRESSURE (reject the request, 429) rather
    than as an engine failure. It is an expected operating condition under overload, not a bug.
    """


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
        # Called when the free list is empty, to reclaim blocks held only by a cache.
        # Without it, blocks parked in the PrefixCache are unreachable and alloc() raises
        # RuntimeError on the request path — a crash where backpressure belongs.
        self._reclaim = None

    def set_reclaimer(self, fn) -> None:
        """Register a callback that frees cache-held blocks on demand. Returns blocks freed."""
        self._reclaim = fn

    def alloc(self) -> int:
        if not self._free and self._reclaim is not None:
            self._reclaim(1)
        if not self._free:
            raise KVCacheExhausted(f"BlockPool exhausted ({self.num_blocks} blocks)")
        bid = self._free.pop()
        self._refcounts[bid] = 1
        return bid

    def acquire(self, block_idx: int) -> None:
        """Add a reference to a block already owned by something else (cache / other session)."""
        assert self._refcounts[block_idx] > 0, "acquiring a freed block"
        self._refcounts[block_idx] += 1

    def release(self, block_idx: int) -> bool:
        """Drop a reference; return to free list when the last reference drops.
        Returns True only if the block ACTUALLY became free."""
        self._refcounts[block_idx] -= 1
        if self._refcounts[block_idx] == 0:
            self._free.append(block_idx)
            return True
        return False

    @property
    def free_count(self) -> int:
        return len(self._free)


class PrefixCache:
    """Maps aligned token-prefixes → per-layer block ids. Shares blocks across sessions
    via refcount: storing acquires; eviction releases. Sharing happens only on full-block
    boundaries (partial blocks aren't safe to share — slots after seq_len aren't written).
    """

    def __init__(self, pools: list[BlockPool | None], max_entries: int = 1024,
                 max_block_fraction: float = 0.5):
        self.pools = pools
        self.block_size = next(p.block_size for p in pools if p is not None)
        # key = tuple of leading tokens aligned to block boundary
        # value = dict[layer_idx] -> list of block ids (one per full block)
        # OrderedDict = LRU order: least-recently-used first, hits move to the end.
        self.entries: "OrderedDict[tuple[int, ...], dict[int, list[int]]]" = OrderedDict()
        self.max_entries = max_entries
        # Block counts present in the cache, so we only build lookup keys for lengths that
        # actually exist (building every aligned prefix tuple is O(prompt^2) per lookup).
        self._lengths: dict[int, int] = {}
        # Cap the cache's share of the pool. Reclaim-on-demand alone is a safety net, not a
        # policy: without this the cache grows to 100% of the pool and every single alloc has
        # to evict first, leaving zero headroom for a burst of live requests.
        live = [p.num_blocks for p in pools if p is not None]
        self.max_blocks = int(max_block_fraction * min(live)) if live else 0
        self._blocks_held = 0          # per-pool block count held across all entries
        self.hits = 0
        self.lookups = 0
        self.evictions = 0
        for pool in pools:
            if pool is not None:
                pool.set_reclaimer(self.reclaim)

    def lookup(self, token_ids: list[int]) -> tuple[int, dict[int, list[int]]]:
        """Return (matched_tokens, per_layer_block_ids). Acquires blocks for the caller."""
        self.lookups += 1
        n_full_blocks = len(token_ids) // self.block_size
        # Walk back from the longest aligned prefix, skipping block counts the cache has never
        # seen — otherwise every lookup builds one tuple per block boundary, which is O(n^2)
        # in prompt length on the prefill hot path.
        for n in range(n_full_blocks, 0, -1):
            if n not in self._lengths:
                continue
            key = tuple(token_ids[: n * self.block_size])
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)          # LRU: this entry is now the newest
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
            self.entries.move_to_end(key)
            return  # already cached
        while (len(self.entries) >= self.max_entries
               or self._blocks_held + n_full_blocks > self.max_blocks):
            if self._evict_one() == 0:
                break
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
        self._blocks_held += n_full_blocks
        self._lengths[n_full_blocks] = self._lengths.get(n_full_blocks, 0) + 1

    def _evict_one(self) -> int:
        """Drop the least-recently-used entry, releasing its refs. Returns blocks released."""
        if not self.entries:
            return 0
        key, entry = self.entries.popitem(last=False)
        n = len(key) // self.block_size
        self._blocks_held -= n
        if self._lengths.get(n, 0) <= 1:
            self._lengths.pop(n, None)
        else:
            self._lengths[n] -= 1
        # Count blocks that actually returned to a free list, NOT release() calls. A cached
        # block that a live request is also holding frees nothing when the cache drops its ref;
        # counting the call made reclaim() report success having freed zero, so the caller
        # stopped evicting and alloc() raised anyway.
        released = 0
        for layer_idx, bids in entry.items():
            for bid in bids:
                if self.pools[layer_idx].release(bid):
                    released += 1
        self.evictions += 1
        return released

    def reclaim(self, min_blocks: int = 1) -> int:
        """Free cache-held blocks until at least `min_blocks` have been released.

        This is what BlockPool calls when its free list runs dry. Cached blocks are a cache,
        not a reservation: they must be givable-back on demand, or a warm cache slowly starves
        the pool and alloc() raises mid-request.
        """
        released = 0
        while released < min_blocks and self.entries:
            released += self._evict_one()   # may be 0 if those blocks are still live elsewhere
        return released

    def stats(self) -> dict:
        free = min((p.free_count for p in self.pools if p is not None), default=0)
        total = min((p.num_blocks for p in self.pools if p is not None), default=0)
        return {
            "entries": len(self.entries), "max_entries": self.max_entries,
            "lookups": self.lookups, "hits": self.hits,
            "hit_rate": round(self.hits / self.lookups, 4) if self.lookups else 0.0,
            "evictions": self.evictions,
            "blocks_held": self._blocks_held, "max_blocks": self.max_blocks,
            "pool_free_blocks": free, "pool_total_blocks": total,
            "pool_utilization": round(1 - free / total, 4) if total else 0.0,
        }


class _RadixNode:
    """One block-sized chunk of a cached prefix. Path from root = the aligned prefix."""

    __slots__ = ("chunk", "children", "blocks", "parent", "last_used")

    def __init__(self, chunk=(), parent=None):
        self.chunk = chunk
        self.parent = parent
        self.children: dict[tuple, "_RadixNode"] = {}
        self.blocks: dict[int, int] = {}      # layer_idx -> block id for THIS chunk
        self.last_used = 0


class RadixPrefixCache:
    """Block-granular radix trie over cached prefixes. Drop-in for PrefixCache.

    The dict version keyed whole aligned prefixes, so it could only match a prefix that had been
    STORED — i.e. a complete earlier prompt. Two requests sharing a long system prompt but
    differing after it matched nothing unless that exact boundary happened to be a stored key, and
    storing "A B C" then "A B D" duplicated the refs for A and B.

    A trie shares at EVERY block boundary: each node owns one block per layer, so any common
    leading blocks are reused automatically and stored once. Lookup is one dict probe per block
    instead of building an O(prompt^2) pile of prefix tuples.
    """

    def __init__(self, pools: list[BlockPool | None], max_entries: int = 1024,
                 max_block_fraction: float = 0.5):
        self.pools = pools
        self.block_size = next(p.block_size for p in pools if p is not None)
        self._live = [i for i, p in enumerate(pools) if p is not None]
        self.root = _RadixNode()
        live = [pools[i].num_blocks for i in self._live]
        # One node holds one block per layer, so for the trie the block watermark IS the node
        # cap and `max_entries` is redundant. They must NOT be combined: `max_entries` counts
        # whole PROMPTS for the dict implementation but single BLOCKS here, so taking the min
        # shrank the trie to 1024 blocks of a 4096 allowance and made it thrash (22589 evictions
        # for a 0.0015 hit rate under load).
        self.max_blocks = int(max_block_fraction * min(live)) if live else 0
        self.max_entries = max_entries      # kept for API parity; not a binding cap here
        self._nodes = 0
        self._clock = 0
        self.hits = 0
        self.lookups = 0
        self.evictions = 0
        for pool in pools:
            if pool is not None:
                pool.set_reclaimer(self.reclaim)

    # ---- internals ----

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _capacity(self) -> int:
        return self.max_blocks

    def _evict_lru_leaf(self) -> int:
        """Drop the least recently used LEAF (an interior node is still someone's path)."""
        best, stack = None, [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and not n.children:
                if best is None or n.last_used < best.last_used:
                    best = n
            stack.extend(n.children.values())
        if best is None:
            return 0
        freed = 0
        for layer_idx, bid in best.blocks.items():
            if self.pools[layer_idx].release(bid):
                freed += 1
        best.parent.children.pop(best.chunk, None)
        best.parent = None
        self._nodes -= 1
        self.evictions += 1
        return freed

    # ---- public API (same shape as PrefixCache) ----

    def lookup(self, token_ids: list[int]) -> tuple[int, dict[int, list[int]]]:
        self.lookups += 1
        node, matched = self.root, 0
        per_layer: dict[int, list[int]] = {i: [] for i in self._live}
        bs = self.block_size
        for i in range(len(token_ids) // bs):
            child = node.children.get(tuple(token_ids[i * bs:(i + 1) * bs]))
            if child is None:
                break
            node = child
            node.last_used = self._tick()
            for layer_idx, bid in node.blocks.items():
                per_layer[layer_idx].append(bid)
            matched += bs
        if matched == 0:
            return 0, {}
        self.hits += 1
        for layer_idx, bids in per_layer.items():
            for bid in bids:
                self.pools[layer_idx].acquire(bid)
        return matched, per_layer

    def store(self, token_ids: list[int], block_tables: list[list[int]]) -> None:
        bs = self.block_size
        n_chunks = len(token_ids) // bs
        if n_chunks == 0:
            return
        # A node is only usable if EVERY live layer has a real block for that chunk; a layer that
        # evicted its leading blocks (sliding window) caps how deep we can store.
        for layer_idx in self._live:
            bids = block_tables[layer_idx]
            usable = 0
            for b in bids[:n_chunks]:
                if b is None or b < 0:
                    break
                usable += 1
            n_chunks = min(n_chunks, usable)
        if n_chunks == 0:
            return

        node = self.root
        for i in range(n_chunks):
            key = tuple(token_ids[i * bs:(i + 1) * bs])
            child = node.children.get(key)
            if child is None:
                while self._nodes >= self._capacity():
                    if self._evict_lru_leaf() == 0:
                        break
                if self._nodes >= self._capacity():
                    return                       # cannot make room; stop growing the trie
                child = _RadixNode(chunk=key, parent=node)
                for layer_idx in self._live:
                    bid = block_tables[layer_idx][i]
                    child.blocks[layer_idx] = bid
                    self.pools[layer_idx].acquire(bid)
                node.children[key] = child
                self._nodes += 1
            node = child
            node.last_used = self._tick()

    def reclaim(self, min_blocks: int = 1) -> int:
        freed = 0
        while freed < min_blocks and self._nodes > 0:
            got = self._evict_lru_leaf()
            if got == 0 and self._nodes == 0:
                break
            freed += got
        return freed

    def stats(self) -> dict:
        free = min((self.pools[i].free_count for i in self._live), default=0)
        total = min((self.pools[i].num_blocks for i in self._live), default=0)
        return {
            "impl": "radix",
            "entries": self._nodes, "max_entries": self.max_entries,
            "lookups": self.lookups, "hits": self.hits,
            "hit_rate": round(self.hits / self.lookups, 4) if self.lookups else 0.0,
            "evictions": self.evictions,
            "blocks_held": self._nodes, "max_blocks": self._capacity(),
            "pool_free_blocks": free, "pool_total_blocks": total,
            "pool_utilization": round(1 - free / total, 4) if total else 0.0,
        }


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

        # Vectorized scatter (was a per-token Python loop — O(S_new) tiny GPU writes, brutal when
        # distributing a batched prefill). k_new [1, H, S_new, D] → [S_new, H, D]; positions
        # seq_len..seq_len+S_new-1 map to (block, slot) and write in one indexed assignment.
        k_per_tok = k_new[0].transpose(0, 1).contiguous()
        v_per_tok = v_new[0].transpose(0, 1).contiguous()
        pos = seq_len + torch.arange(S_new, device=k_new.device)
        bids = torch.tensor(bt, device=k_new.device)[pos // block_size]
        slots = pos % block_size
        pool.k[bids, :, slots, :] = k_per_tok
        pool.v[bids, :, slots, :] = v_per_tok

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


class BatchedDecodeState:
    """Persistent batched-decode state — the de-Pythoned replacement for rebuilding
    `BatchedPagedKVCache` every step. Block tables and seq_lens live as GPU tensors, updated
    incrementally on admit (`add_row`) / decode / evict, so the per-step hot path is pure GPU
    ops (no Python list→tensor conversion, which profiling showed is ~95% of decode latency).

    Owns its blocks directly (ingests a prefilled `PagedKVCache` via `add_row`, then frees on
    `remove_row`). Drop-in for the model's paged_ctx interface: `append()` writes the new token
    (scatter, no advance), `block_table_tensor()` returns (block table, effective seq_lens) for
    the kernel. `prepare_step()` allocs boundary-crossing blocks once per step (before the
    forward); `advance()` bumps seq_lens + evicts (after the forward).
    """

    def __init__(self, pools: list[BlockPool | None], device: torch.device):
        self.pools = pools
        self.device = device
        self.bs = next(p.block_size for p in pools if p is not None)
        self.seq_lens = torch.zeros(0, dtype=torch.long, device=device)   # [R] logical len (pre-step)
        self.n_alloc = torch.zeros(0, dtype=torch.long, device=device)    # [R] logical blocks alloc'd
        self.block_tables: list[torch.Tensor | None] = [None] * len(pools)  # per layer [R, cols] (-1 = evicted)
        # Sliding layers all share one window + seq_lens, so they evict in lockstep → one
        # shared evicted-count tensor [R] (not per-layer), enabling a sync-free fast path.
        self._sliding = [L for L, p in enumerate(pools) if p is not None and p.window is not None]
        self._window = next((p.window for p in pools if p is not None and p.window is not None), None)
        self.evicted = torch.zeros(0, dtype=torch.long, device=device)    # [R] evicted leading blocks

    @property
    def n_rows(self) -> int:
        return self.seq_lens.shape[0]

    @staticmethod
    def _grow_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
        if t.shape[1] >= cols:
            return t
        pad = torch.zeros(t.shape[0], cols - t.shape[1], dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=1)

    def add_row(self, pkv: "PagedKVCache") -> None:
        """Adopt a prefilled PagedKVCache's blocks as a new batch row (no per-step cost)."""
        self.seq_lens = torch.cat([self.seq_lens, torch.tensor([pkv.seq_len], device=self.device)])
        n0 = max((len(pkv.block_tables[L]) for L, p in enumerate(self.pools) if p is not None), default=0)
        self.n_alloc = torch.cat([self.n_alloc, torch.tensor([n0], device=self.device)])
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            row = torch.tensor(pkv.block_tables[L], dtype=torch.long, device=self.device)
            cur = self.block_tables[L]
            if cur is None:
                self.block_tables[L] = row.unsqueeze(0)
            else:
                cols = max(cur.shape[1], row.shape[0])
                cur = self._grow_cols(cur, cols)
                row = torch.cat([row, torch.zeros(cols - row.shape[0], dtype=torch.long, device=self.device)])
                self.block_tables[L] = torch.cat([cur, row.unsqueeze(0)], dim=0)
        ev0 = pkv.evicted[self._sliding[0]] if self._sliding else 0  # sliding layers share the count
        self.evicted = torch.cat([self.evicted, torch.tensor([ev0], device=self.device)])

    def remove_row(self, idx: int) -> None:
        """Free row `idx`'s blocks and drop it from the batch."""
        keep = [i for i in range(self.n_rows) if i != idx]
        keep_t = torch.tensor(keep, dtype=torch.long, device=self.device)
        n_alloc_idx = int(self.n_alloc[idx].item())
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            row_blocks = self.block_tables[L][idx, :n_alloc_idx].tolist()
            for bid in row_blocks:
                if bid >= 0:  # -1 = already-evicted (freed); padding never reaches here (< n_alloc)
                    pool.release(bid)
            self.block_tables[L] = self.block_tables[L][keep_t] if keep else None
        self.seq_lens = self.seq_lens[keep_t]
        self.n_alloc = self.n_alloc[keep_t]
        self.evicted = self.evicted[keep_t]

    def prepare_step(self) -> None:
        """Alloc a block for every boundary-crossing row in every layer (once per step)."""
        if self.n_rows == 0:
            return
        lb = self.seq_lens // self.bs                  # logical block of the new token, per row
        crossing = (lb >= self.n_alloc).nonzero().flatten().tolist()
        if not crossing:
            return
        need_cols = int(lb.max().item()) + 1
        for L, pool in enumerate(self.pools):
            if pool is None:
                continue
            self.block_tables[L] = self._grow_cols(self.block_tables[L], need_cols)
            for r in crossing:
                self.block_tables[L][r, int(lb[r].item())] = pool.alloc()
        for r in crossing:
            self.n_alloc[r] += 1

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Scatter the new token (k_new,v_new [R,H,1,D]) into each row's block. Pure GPU —
        block id + slot computed from seq_lens (prepare_step already alloc'd any new block)."""
        pool = self.pools[layer_idx]
        rows = torch.arange(self.n_rows, device=self.device)
        lb = self.seq_lens // self.bs
        slot = self.seq_lens % self.bs
        bid = self.block_tables[layer_idx][rows, lb]   # [R]
        pool.k[bid, :, slot, :] = k_new[:, :, 0, :]
        pool.v[bid, :, slot, :] = v_new[:, :, 0, :]

    def block_table_tensor(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Kernel inputs: persistent block table (int32, -1→0) + effective seq_lens (incl. the
        new token just written). No rebuild — just dtype/sign cleanup + the +1."""
        bt = self.block_tables[layer_idx].to(torch.int32).clamp_min(0)
        return bt, (self.seq_lens + 1).to(torch.int32)

    def advance(self) -> None:
        """Bump seq_lens by the new token, then evict out-of-window blocks. Sync-free fast path:
        one `.any()` to check if any row crossed the window; the per-row Python (with its
        GPU→CPU `.item()` syncs) runs only when a block is actually evicted (~1 per 16 steps)."""
        self.seq_lens = self.seq_lens + 1
        if self._window is None or self.n_rows == 0:
            return
        keep_from = ((self.seq_lens - self._window).clamp_min(0)) // self.bs   # [R], same for all sliding layers
        need = keep_from > self.evicted
        if not bool(need.any()):   # one sync; usually nothing to evict
            return
        for r in need.nonzero().flatten().tolist():
            for i in range(int(self.evicted[r].item()), int(keep_from[r].item())):
                for L in self._sliding:
                    bid = int(self.block_tables[L][r, i].item())
                    if bid >= 0:
                        self.pools[L].release(bid)
                        self.block_tables[L][r, i] = -1
            self.evicted[r] = keep_from[r]


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
