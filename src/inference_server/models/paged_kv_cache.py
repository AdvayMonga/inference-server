"""Block-paged KV cache for the custom Qwen3-MoE forward (M2.2).

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

import heapq
from collections import OrderedDict

import torch
import torch.nn.functional as F

from inference_server.models import fused


class _PinnedRing:
    """Pinned staging buffers for host -> device copies, reused in rotation: pinning a fresh buffer per copy
    (cudaHostAlloc) was 40% of the scheduler thread at 1,000 decode rows (py-spy, H200 2026-10-20). A slot is reused
    only once the copy that last read it has completed (its event)."""

    SLOTS, ELEMS = 16, 1 << 20                                  # 16 x 8 MB

    def __init__(self):
        self.bufs = [torch.empty(self.ELEMS, dtype=torch.int64, pin_memory=True) for _ in range(self.SLOTS)]
        self.events: list = [None] * self.SLOTS
        self.i = 0

    def copy(self, t: torch.Tensor, device: torch.device) -> torch.Tensor:
        i = self.i
        self.i = (i + 1) % self.SLOTS
        if self.events[i] is not None:
            self.events[i].synchronize()
        n = t.numel()
        src = self.bufs[i].view(t.dtype)[:n]
        src.copy_(t.reshape(-1))
        out = src.to(device, non_blocking=True).view(t.shape)
        ev = torch.cuda.Event()
        ev.record()
        self.events[i] = ev
        return out


_ring: _PinnedRing | None = None


def to_device(values, device: torch.device, dtype=torch.long) -> torch.Tensor:
    """A host list as a device tensor. On CUDA through pinned memory with a non-blocking copy: a pageable copy waits for
    every kernel queued on the stream, i.e. for the decode step running asynchronously (2026-10-19: such copies in row
    removal made each evict phase wait ~12 ms under long prompts). The pinned memory is a ring of reused buffers."""
    global _ring
    t = torch.tensor(values, dtype=dtype)
    if device.type != "cuda":
        return t.to(device)
    if t.numel() * t.element_size() > _PinnedRing.ELEMS * 8 or t.numel() == 0:
        return t.pin_memory().to(device, non_blocking=True)
    if _ring is None:
        _ring = _PinnedRing()
    return _ring.copy(t, device)


def write_token(pool, bt: torch.Tensor, rows: torch.Tensor, lb: torch.Tensor, slot: torch.Tensor,
                k_new: torch.Tensor, v_new: torch.Tensor) -> None:
    """One decode token per row into the pool: pool.k[bt[r, lb[r]], :, slot[r], :] = k_new[r, :, 0, :] (and V)."""
    if fused._on(k_new, v_new, pool.k):
        fused.kv_store(pool.k, pool.v, bt, lb, slot, k_new, v_new)
        return
    bid = bt[rows, lb]
    pool.k[bid, :, slot, :] = k_new[:, :, 0, :].to(pool.k.dtype)
    pool.v[bid, :, slot, :] = v_new[:, :, 0, :].to(pool.v.dtype)


class KVCacheExhausted(RuntimeError):
    """No KV block available even after reclaiming the prefix cache.

    A typed error so the scheduler can treat it as BACKPRESSURE (reject the request, 429) rather
    than as an engine failure. It is an expected operating condition under overload, not a bug.
    """


class BlockAllocator:
    """Block ids, refcounts and the free list, shared by every layer's pool. A request's block id is the same in all
    of its layers' pools, so one list of ids and one refcount per block serve every layer. (One list per layer, with
    48x the acquire, release and allocation calls, was ~2.4 ms of host time per 4k-token request, 2026-10-20.)"""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
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

    def alloc_many(self, n: int) -> list[int]:
        """`n` blocks, all or none (a pool that runs dry leaves nothing half-taken)."""
        got: list[int] = []
        try:
            for _ in range(n):
                got.append(self.alloc())
        except KVCacheExhausted:
            self.release_all(got)
            raise
        return got

    def acquire(self, block_idx: int) -> None:
        """Add a reference to a block already owned by something else (cache / other session)."""
        assert self._refcounts[block_idx] > 0, "acquiring a freed block"
        self._refcounts[block_idx] += 1

    def acquire_all(self, bids) -> None:
        rc = self._refcounts
        for bid in bids:
            assert rc[bid] > 0, "acquiring a freed block"
            rc[bid] += 1

    def release(self, block_idx: int) -> bool:
        """Drop a reference; return to free list when the last reference drops.
        Returns True only if the block ACTUALLY became free."""
        self._refcounts[block_idx] -= 1
        if self._refcounts[block_idx] == 0:
            self._free.append(block_idx)
            return True
        return False

    def release_all(self, bids) -> int:
        """Drop one reference on each; returns how many blocks actually became free."""
        rc, free, freed = self._refcounts, self._free, 0
        for bid in bids:
            rc[bid] -= 1
            if rc[bid] == 0:
                free.append(bid)
                freed += 1
        return freed

    @property
    def free_count(self) -> int:
        return len(self._free)


class BlockPool:
    """One layer's K/V storage. The block ids come from the group's shared `BlockAllocator` (block `b` of this pool
    and of every other layer's pool hold the same request's tokens); the allocation calls here delegate to it."""

    def __init__(
        self, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
        dtype: torch.dtype, device: torch.device, allocator: BlockAllocator | None = None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.k = torch.zeros(num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.allocator = allocator if allocator is not None else BlockAllocator(num_blocks)
        assert self.allocator.num_blocks == num_blocks, "every pool of a group has the allocator's block count"

    def set_reclaimer(self, fn) -> None:
        self.allocator.set_reclaimer(fn)

    def alloc(self) -> int:
        return self.allocator.alloc()

    def acquire(self, block_idx: int) -> None:
        self.allocator.acquire(block_idx)

    def release(self, block_idx: int) -> bool:
        return self.allocator.release(block_idx)

    @property
    def free_count(self) -> int:
        return self.allocator.free_count

    @property
    def _free(self) -> list[int]:
        return self.allocator._free

    @property
    def _refcounts(self) -> list[int]:
        return self.allocator._refcounts


def pool_group(n_layers: int, num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int,
               dtype: torch.dtype, device: torch.device) -> list[BlockPool]:
    """`n_layers` pools over one shared allocator (what `make_pools` builds for a model)."""
    alloc = BlockAllocator(num_blocks)
    return [BlockPool(num_blocks, block_size, num_kv_heads, head_dim, dtype, device, alloc) for _ in range(n_layers)]


def _allocator(pools) -> BlockAllocator:
    """The allocator every live pool shares (a request's blocks are one list for all of its layers)."""
    live = [p for p in pools if p is not None]
    alloc = live[0].allocator
    assert all(p.allocator is alloc for p in live), "the live pools must share one BlockAllocator (pool_group / make_pools)"
    return alloc


def _tightest_pool(pools) -> tuple[int, int]:
    """(free, total) blocks of the pools' shared allocator (every layer fills at the same rate)."""
    live = [p for p in pools if p is not None and p.num_blocks]
    if not live:
        return 0, 0
    alloc = _allocator(live)
    return alloc.free_count, alloc.num_blocks


class PrefixCache:
    """Maps aligned token-prefixes → block ids (one list, shared by every layer). Shares blocks across sessions
    via refcount: storing acquires; eviction releases. Sharing happens only on full-block
    boundaries (partial blocks aren't safe to share — slots after seq_len aren't written).
    """

    def __init__(self, pools: list[BlockPool | None], max_entries: int = 1024,
                 max_block_fraction: float = 0.5):
        self.pools = pools
        self.block_size = next(p.block_size for p in pools if p is not None)
        self.alloc = _allocator(pools)
        # key = tuple of leading tokens aligned to block boundary; value = block ids (one per full block)
        # OrderedDict = LRU order: least-recently-used first, hits move to the end.
        self.entries: "OrderedDict[tuple[int, ...], list[int]]" = OrderedDict()
        self.max_entries = max_entries
        # Block counts present in the cache, so we only build lookup keys for lengths that
        # actually exist (building every aligned prefix tuple is O(prompt^2) per lookup).
        self._lengths: dict[int, int] = {}
        # Cap the cache's share of the pool. Reclaim-on-demand alone is a safety net, not a
        # policy: without this the cache grows to 100% of the pool and every single alloc has
        # to evict first, leaving zero headroom for a burst of live requests.
        self.max_blocks = int(max_block_fraction * self.alloc.num_blocks)
        self._blocks_held = 0          # blocks held across all entries
        self.hits = 0
        self.lookups = 0
        self.evictions = 0
        self.alloc.set_reclaimer(self.reclaim)

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Return (matched_tokens, block_ids). Acquires the blocks for the caller."""
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
                bids = self.entries[key]
                self.alloc.acquire_all(bids)           # the caller becomes an additional owner
                return n * self.block_size, list(bids)
        return 0, []

    def store(self, token_ids: list[int], blocks: list[int]) -> None:
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
        full_bids = list(blocks[:n_full_blocks])        # only the first n_full_blocks are guaranteed full
        self.alloc.acquire_all(full_bids)                # the cache's own ref: the blocks survive the session
        self.entries[key] = full_bids
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
        released = self.alloc.release_all(entry)
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
        free, total = _tightest_pool(self.pools)
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

    __slots__ = ("chunk", "children", "block", "parent", "last_used")

    def __init__(self, chunk=(), parent=None, block: int = -1):
        self.chunk = chunk
        self.parent = parent
        self.children: dict[tuple, "_RadixNode"] = {}
        self.block = block                    # this chunk's block id (the same in every layer's pool)
        self.last_used = 0


class RadixPrefixCache:
    """Block-granular radix trie over cached prefixes. Drop-in for PrefixCache.

    The dict version keyed whole aligned prefixes, so it could only match a prefix that had been
    STORED — i.e. a complete earlier prompt. Two requests sharing a long system prompt but
    differing after it matched nothing unless that exact boundary happened to be a stored key, and
    storing "A B C" then "A B D" duplicated the refs for A and B.

    A trie shares at EVERY block boundary: each node owns one block (the same id in every layer's pool), so any
    common leading blocks are reused automatically and stored once. Lookup is one dict probe per block
    instead of building an O(prompt^2) pile of prefix tuples.
    """

    def __init__(self, pools: list[BlockPool | None], max_entries: int = 1024,
                 max_block_fraction: float = 0.5):
        self.pools = pools
        self.block_size = next(p.block_size for p in pools if p is not None)
        self._live = [i for i, p in enumerate(pools) if p is not None]
        self.alloc = _allocator(pools)
        self.root = _RadixNode()
        # One node holds one block, so for the trie the block watermark IS the node
        # cap and `max_entries` is redundant. They must NOT be combined: `max_entries` counts
        # whole PROMPTS for the dict implementation but single BLOCKS here, so taking the min
        # shrank the trie to 1024 blocks of a 4096 allowance and made it thrash (22589 evictions
        # for a 0.0015 hit rate under load).
        self.max_blocks = int(max_block_fraction * self.alloc.num_blocks)
        self.max_entries = max_entries      # kept for API parity; not a binding cap here
        self._nodes = 0
        self._clock = 0
        # Leaves by last use, as a lazy min-heap: an entry is stale once its node gained a child, left the trie or was
        # used again, and is skipped when popped. Evicting the LRU leaf is then O(log n), not a walk of the whole trie
        # once per block (which made long-context prefill quadratic under memory pressure).
        self._leaves: list[tuple[int, int, _RadixNode]] = []
        self._seq = 0
        self.hits = 0
        self.lookups = 0
        self.evictions = 0
        self.alloc.set_reclaimer(self.reclaim)

    # ---- internals ----

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _capacity(self) -> int:
        return self.max_blocks

    def _push_leaf(self, node: "_RadixNode") -> None:
        if node is not self.root and not node.children:
            self._seq += 1
            heapq.heappush(self._leaves, (node.last_used, self._seq, node))
            if len(self._leaves) > 4 * self._nodes + 64:          # bound the stale entries: rebuild from the trie
                self._rebuild_leaves()

    def _rebuild_leaves(self) -> None:
        leaves, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and not n.children:
                self._seq += 1
                leaves.append((n.last_used, self._seq, n))
            stack.extend(n.children.values())
        heapq.heapify(leaves)
        self._leaves = leaves

    def _evict_lru_leaf(self) -> int:
        """Drop the least recently used LEAF (an interior node is still someone's path)."""
        best = None
        while self._leaves:
            used, _, n = heapq.heappop(self._leaves)
            if n.parent is not None and not n.children and n.last_used == used:
                best = n
                break
        if best is None:
            return 0
        freed = 1 if self.alloc.release(best.block) else 0
        parent = best.parent
        parent.children.pop(best.chunk, None)
        best.parent = None
        self._push_leaf(parent)                  # its last child gone, the parent is now a leaf
        self._nodes -= 1
        self.evictions += 1
        return freed

    # ---- public API (same shape as PrefixCache) ----

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """(matched tokens, their block ids); the blocks are acquired for the caller."""
        self.lookups += 1
        node, blocks = self.root, []
        bs = self.block_size
        tick = self._tick()
        for i in range(len(token_ids) // bs):
            child = node.children.get(tuple(token_ids[i * bs:(i + 1) * bs]))
            if child is None:
                break
            node = child
            node.last_used = tick
            blocks.append(node.block)
        if not blocks:
            return 0, []
        self._push_leaf(node)                        # only the deepest node can be a leaf
        self.hits += 1
        self.alloc.acquire_all(blocks)
        return len(blocks) * bs, blocks

    def store(self, token_ids: list[int], blocks: list[int]) -> None:
        bs = self.block_size
        n_chunks = len(token_ids) // bs
        if n_chunks == 0:
            return

        node = self.root
        tick = self._tick()
        for i in range(n_chunks):
            key = tuple(token_ids[i * bs:(i + 1) * bs])
            child = node.children.get(key)
            if child is None:
                while self._nodes >= self._capacity():
                    if self._evict_lru_leaf() == 0:
                        break
                if self._nodes >= self._capacity():
                    self._push_leaf(node)        # cannot make room; stop growing the trie (the path's end stays evictable)
                    return
                child = _RadixNode(chunk=key, parent=node, block=blocks[i])
                self.alloc.acquire(blocks[i])
                node.children[key] = child
                self._nodes += 1
            node = child
            node.last_used = tick
        self._push_leaf(node)

    def reclaim(self, min_blocks: int = 1) -> int:
        freed = 0
        while freed < min_blocks and self._nodes > 0:
            got = self._evict_lru_leaf()
            if got == 0 and self._nodes == 0:
                break
            freed += got
        return freed

    def stats(self) -> dict:
        free, total = _tightest_pool([self.pools[i] for i in self._live])
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


class _LayerTables:
    """`PagedKVCache.block_tables`: the one block list seen per layer (None for a layer without K/V)."""

    __slots__ = ("_c",)

    def __init__(self, cache: "PagedKVCache"):
        self._c = cache

    def __getitem__(self, layer_idx: int):
        return self._c.blocks if self._c.pools[layer_idx] is not None else None

    def __len__(self) -> int:
        return self._c.num_layers

    def __iter__(self):
        return (self[L] for L in range(self._c.num_layers))


class PagedKVCache:
    """Per-session block table: one list of block ids, the same for every layer's pool (every layer stores a
    token's K/V in the same block id). Shares pools with other sessions.

    Construct with `shared_prefix=[bid, ...]` and `shared_prefix_tokens` to seed the cache from a prefix-cache hit.
    The caller is responsible for skipping forward computation on the prefix-matched portion of the prompt.
    """

    def __init__(
        self, pools: list[BlockPool | None],
        shared_prefix: list[int] | None = None,
        shared_prefix_tokens: int = 0,
    ):
        self.pools = pools
        self.num_layers = len(pools)
        self.alloc = _allocator(pools)
        self.blocks: list[int] = list(shared_prefix) if shared_prefix else []
        # Per-layer lengths: mid-forward, the layers already appended to are one token ahead of the rest.
        self.seq_lens: list[int] = [shared_prefix_tokens if shared_prefix and p is not None else 0 for p in pools]

    @property
    def block_tables(self) -> _LayerTables:
        """Per-layer view of the one block list (reads and appends go through)."""
        return _LayerTables(self)

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
        # Gather all blocks → [N_blocks, H, block_size, D].
        idx = torch.tensor(self.blocks, device=pool.k.device, dtype=torch.long)
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
        bt = self.blocks

        # Enough blocks for (seq_len + S_new) tokens: the first layer to reach a new block allocates it for all.
        total_blocks_needed = (seq_len + S_new + block_size - 1) // block_size
        while len(bt) < total_blocks_needed:
            bt.append(self.alloc.alloc())

        # Vectorized scatter (was a per-token Python loop — O(S_new) tiny GPU writes, brutal when
        # distributing a batched prefill). k_new [1, H, S_new, D] → [S_new, H, D]; positions
        # seq_len..seq_len+S_new-1 map to (block, slot) and write in one indexed assignment.
        k_per_tok = k_new[0].transpose(0, 1).contiguous()
        v_per_tok = v_new[0].transpose(0, 1).contiguous()
        pos = seq_len + torch.arange(S_new, device=k_new.device)
        bids = torch.tensor(bt, device=k_new.device)[pos // block_size]
        slots = pos % block_size
        pool.k[bids, :, slots, :] = k_per_tok.to(pool.k.dtype)
        pool.v[bids, :, slots, :] = v_per_tok.to(pool.v.dtype)

        self.seq_lens[layer_idx] = seq_len + S_new

    def free_all(self) -> None:
        """Release this session's references; blocks return to the pool when their refcount hits 0."""
        self.alloc.release_all(self.blocks)
        self.blocks = []
        self.seq_lens = [0] * self.num_layers


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
            bt = row.blocks
            if L // bs >= len(bt):          # crossed into a new block this step (the first layer allocates for all)
                bt.append(row.alloc.alloc())
            block_ids.append(bt[L // bs])
            slots.append(L % bs)
            row.seq_lens[layer_idx] = L + 1
        dev = pool.k.device
        bidx = torch.tensor(block_ids, dtype=torch.long, device=dev)
        sidx = torch.tensor(slots, dtype=torch.long, device=dev)
        pool.k[bidx, :, sidx, :] = k_new[:, :, 0, :]   # [N,H,D] — rows own distinct (block,slot)
        pool.v[bidx, :, sidx, :] = v_new[:, :, 0, :]

    def pools_for(self, layer_idx: int):
        return self.rows[0].pools[layer_idx]

    def block_table_tensor(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row block table [N, max_blocks] (int32, 0-padded) + seq_lens [N] for the kernel.
        Built with one host→device copy each (one padded list) instead of per-row tensors."""
        device = self.pools_for(layer_idx).k.device
        bts = [r.blocks for r in self.rows]
        seq_lens = [r.seq_lens[layer_idx] for r in self.rows]
        max_blocks = max((len(b) for b in bts), default=1) or 1
        padded = [b + [0] * (max_blocks - len(b)) for b in bts]
        bt = torch.tensor(padded, dtype=torch.int32, device=device)
        sl = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        return bt, sl


class BatchedDecodeState:
    """Persistent batched-decode state — the de-Pythoned replacement for rebuilding
    `BatchedPagedKVCache` every step. Block tables and seq_lens live as GPU tensors, updated
    incrementally on admit (`add_row`) / decode, so the per-step hot path is pure GPU
    ops (no Python list→tensor conversion, which profiling showed is ~95% of decode latency).

    Owns its blocks directly (ingests a prefilled `PagedKVCache` via `add_row`, then frees on
    `remove_row`). Drop-in for the model's paged_ctx interface: `append()` writes the new token
    (scatter, no advance), `block_table_tensor()` returns (block table, effective seq_lens) for
    the kernel. `prepare_step()` allocs boundary-crossing blocks once per step (before the
    forward); `advance()` bumps seq_lens (after the forward).
    """

    def __init__(self, pools: list[BlockPool | None], device: torch.device):
        self.pools = pools
        self.device = device
        self.bs = next(p.block_size for p in pools if p is not None)
        self.alloc = _allocator(pools)
        self.seq_lens = torch.zeros(0, dtype=torch.long, device=device)   # [R] logical len (pre-step)
        self.n_alloc = torch.zeros(0, dtype=torch.long, device=device)    # [R] logical blocks alloc'd
        # One table [R, cols] for every layer that owns K/V: a block id is the same in every layer's pool. (A table
        # per layer cost a launch or a sync per layer on every join, leave and allocation; stacked [layers, R, cols]
        # they were one op each but 48x the bytes to build, grow, compact and copy into the graphs' buffers.)
        self._live = [L for L, p in enumerate(pools) if p is not None]
        self._li = {L: i for i, L in enumerate(self._live)}
        self.bt: torch.Tensor | None = None
        # Host mirrors of what join, leave and allocation need, so none of them waits on the GPU (an async decode
        # step overlaps host work with the running forward; a .item() here would serialize them again).
        self._host_len: list[int] = []                   # per row: logical length (pre-step)
        self._host_alloc: list[int] = []                 # per row: logical blocks allocated
        self._host_blocks: list[list[int]] = []          # per row: block ids

    @property
    def block_tables(self) -> list:
        """Per-layer view: the one table [R, cols] for every layer with K/V (None otherwise); writes go through."""
        return [self.bt if p is not None else None for p in self.pools]

    @property
    def n_rows(self) -> int:
        return self.seq_lens.shape[0]

    @staticmethod
    def _grow_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
        if t.shape[-1] >= cols:
            return t
        pad = torch.zeros(*t.shape[:-1], cols - t.shape[-1], dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=-1)

    def add_row(self, pkv: "PagedKVCache") -> None:
        """Adopt a prefilled PagedKVCache's blocks as a new batch row (no per-step cost)."""
        self.seq_lens = torch.cat([self.seq_lens, to_device([pkv.seq_len], self.device)])
        blocks = list(pkv.blocks)
        n0 = len(blocks)
        self.n_alloc = torch.cat([self.n_alloc, to_device([n0], self.device)])
        self._host_len.append(pkv.seq_len)
        self._host_alloc.append(n0)
        self._host_blocks.append(blocks)
        cols = max(n0, self.bt.shape[1] if self.bt is not None else 0)
        row = to_device([blocks + [0] * (cols - n0)], self.device)                   # [1, cols]
        self.bt = row if self.bt is None else torch.cat([self._grow_cols(self.bt, cols), row], dim=0)

    def remove_row(self, idx: int) -> None:
        """Free row `idx`'s blocks and drop it from the batch."""
        self.remove_rows([idx])

    def remove_rows(self, idxs) -> None:
        """Free these rows' blocks and drop them from the batch: one compaction for all of them, no stream sync."""
        drop = set(idxs)
        keep = [i for i in range(self.n_rows) if i not in drop]
        for idx in sorted(drop, reverse=True):
            self.alloc.release_all(self._host_blocks.pop(idx))          # host mirror: no D2H
            self._host_len.pop(idx)
            self._host_alloc.pop(idx)
        if not keep:
            self.bt, self.seq_lens, self.n_alloc = None, self.seq_lens[:0], self.n_alloc[:0]
            return
        keep_t = to_device(keep, self.device)
        self.bt = self.bt[keep_t]
        self.seq_lens = self.seq_lens[keep_t]
        self.n_alloc = self.n_alloc[keep_t]

    def prepare_step(self) -> None:
        """Alloc a block for every boundary-crossing row in every layer (once per step)."""
        if self.n_rows == 0:
            return
        bs = self.bs                                   # from the host mirrors: no sync before the step
        crossing = [r for r in range(self.n_rows) if self._host_len[r] // bs >= self._host_alloc[r]]
        if not crossing:
            return
        lbs = [self._host_len[r] // bs for r in crossing]
        self.bt = self._grow_cols(self.bt, max(lbs) + 1)
        ids = self.alloc.alloc_many(len(crossing))                # all or nothing (KVCacheExhausted leaves no block taken)
        for r, bid in zip(crossing, ids):
            self._host_blocks[r].append(bid)
            self._host_alloc[r] += 1
        new, rows_t, lbs_t = (to_device(v, self.device) for v in (ids, crossing, lbs))
        self.bt[rows_t, lbs_t] = new
        self.n_alloc[rows_t] += 1

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Scatter the new token (k_new,v_new [R,H,1,D]) into each row's block. Pure GPU —
        block id + slot computed from seq_lens (prepare_step already alloc'd any new block)."""
        if layer_idx == self._live[0]:
            self._begin_step()
        write_token(self.pools[layer_idx], self.bt, self._rows, self._lb, self._slot, k_new, v_new)

    def _begin_step(self) -> None:
        """What every layer's append and block_table_tensor share, built once at the first layer of a step (as
        per-layer ops they were ~10 launches a layer, ~480 a step)."""
        self._rows = torch.arange(self.n_rows, device=self.device)
        self._lb = self.seq_lens // self.bs
        self._slot = self.seq_lens % self.bs
        self._bt32 = self.bt.to(torch.int32).clamp_min_(0)
        self._sl32 = (self.seq_lens + 1).to(torch.int32)

    def block_table_tensor(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Kernel inputs: persistent block table (int32) + effective seq_lens (incl. the
        new token just written), both from this step's cache."""
        return self._bt32, self._sl32

    def advance(self) -> None:
        """Bump seq_lens by the new token (GPU tensor and host mirror; no sync)."""
        self.seq_lens = self.seq_lens + 1
        for r in range(len(self._host_len)):
            self._host_len[r] += 1


def blocks_for_budget(model, budget_bytes: int, block_size: int, kv_dtype: torch.dtype | None = None) -> int:
    """Blocks per pool that fit `budget_bytes` across every layer owning K/V (K and V, in `kv_dtype`: default the
    model's)."""
    elem = torch.empty((), dtype=kv_dtype or next(model.parameters()).dtype).element_size()
    per_block = sum(2 * layer.self_attn.num_kv_heads * block_size * layer.self_attn.head_dim * elem
                    for layer in model.model.layers if not layer.self_attn.is_kv_shared)
    return max(1, budget_bytes // per_block)


def make_pools(model, num_blocks_per_pool: int = 1024, block_size: int = 16,
               kv_dtype: torch.dtype | None = None) -> list[BlockPool | None]:
    """Build the BlockPool list matching a Qwen3MoeForCausalLM's layers: one pool of
    `num_blocks_per_pool` per layer that owns K/V, None for a layer that does not. `kv_dtype`: the
    pools' element type (default the model's; float8_e4m3fn halves the bytes per token, unit scale)."""
    device = next(model.parameters()).device
    dtype = kv_dtype or next(model.parameters()).dtype
    alloc = BlockAllocator(num_blocks_per_pool)             # one id space for every layer (a request = one block list)
    pools: list[BlockPool | None] = []
    for layer in model.model.layers:
        attn = layer.self_attn
        if attn.is_kv_shared:
            pools.append(None)
        else:
            pools.append(BlockPool(
                num_blocks=num_blocks_per_pool,
                block_size=block_size,
                num_kv_heads=attn.num_kv_heads,
                head_dim=attn.head_dim,
                dtype=dtype,
                device=device,
                allocator=alloc,
            ))
    return pools
