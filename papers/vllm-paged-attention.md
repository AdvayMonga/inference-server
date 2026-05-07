# Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

**Link:** https://arxiv.org/abs/2309.06180
**Venue:** SOSP 2023
**Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (UC Berkeley)
**Status:** read
**Relevant phase:** Phase 5 (KV cache integration) + Phase 6 (continuous batching, vLLM head-to-head)

## Core idea

Naive LLM serving wastes KV cache memory three ways: internal fragmentation (over-reserving for `max_tokens` that never generate), reservation (slots held for future tokens of an active sequence), and external fragmentation (gaps between sequences with different lifetimes). vLLM borrows the operating-system virtual-memory model: split the KV cache into fixed-size **blocks** (e.g., 16 tokens of K and V per block), give each sequence a **block table** mapping logical token positions to physical block IDs, and allocate blocks on demand. Blocks need not be contiguous, can be freed independently, and can be **shared** across sequences when prefixes match. A custom **PagedAttention** kernel reads K and V through the block-table indirection so attention runs directly on non-contiguous memory.

## Why it matters

- This is the architectural foundation Phase 5 is building toward. The planned `BlockManager`, `RadixTree`, and `CacheManager` are direct analogs of vLLM's block allocator, prefix-sharing index, and `BlockSpaceManager`.
- vLLM is the explicit benchmarking target for this project. Understanding the paper deeply is what justifies specific throughput and memory-utilization expectations rather than treating vLLM as an opaque baseline.
- Block-paging is what makes continuous batching practical at scale: without it, packing variable-length sequences into one batch wastes too much memory to be worth the compute gain.

## Key mechanism

1. **Paged KV cache.** Cache memory is a pool of fixed-size blocks. Each sequence holds a block table — an array of physical block IDs in logical-position order. New blocks are allocated only when the sequence's last block fills, eliminating internal fragmentation beyond at most one partially-filled block per sequence.
2. **PagedAttention kernel.** A custom CUDA kernel takes the block table plus the global block pool and computes attention by gathering through the indirection. Naive PyTorch indexing for this gather would dominate runtime; the kernel fuses the gather into the attention computation. This kernel is the part that does not exist as an off-the-shelf PyTorch primitive.
3. **Copy-on-write block sharing.** Multiple sequences can point to the same physical block — e.g., shared system prompts, beam-search siblings, parallel sampling from the same prompt. Each block carries a reference count. When a sequence needs to write into a shared block, the block is copied first and the writing sequence's block table is updated.
4. **Iteration-level scheduling (continuous batching).** At every generation step, the scheduler can: admit waiting sequences if memory allows, evict (swap-out or recompute) lower-priority sequences when memory is tight, and resume previously evicted sequences. Because admission and eviction operate on block IDs rather than tensor copies, these decisions are cheap.
5. **Swap vs recompute.** When the cache is full and a new sequence must run, vLLM either swaps blocks to CPU memory or drops them and rebuilds via prefill on resume. Swap wins for long prefixes when PCIe bandwidth is high; recompute wins for short prefixes and is simpler.
6. **Memory accounting.** Total concurrent capacity is `total_blocks * block_size / avg_seq_len`, not `max_concurrent * max_seq_len`. This is the single biggest reason vLLM serves more concurrent users on the same hardware than naive serving stacks.

## Implementation sketch

```
0. Block manager, radix tree, cache manager (Phase 5 stages 1-3)
   - Bookkeeping data structures only; not yet integrated with generation
   verify: unit tests already in place

1. Wire block-paged KV into the generation loop (Phase 5 stage 7)
   - Replace HF's contiguous KV with the block pool
   - Pure-PyTorch "gather K/V from block table → attend" path for correctness
   - On MPS this will be slower than baseline; that is acceptable
   verify: end-to-end outputs match baseline within numerical tolerance

2. Reference counting + copy-on-write
   - Hook into the radix tree on prefix-match: increment refcount on shared blocks
   - On a write to a refcount > 1 block, copy and update the writing sequence's block table
   verify: two sessions with a shared system prompt actually share blocks; mutation correctly forks

3. Continuous batching scheduler (Phase 6)
   - Iteration-level admit / preempt / resume
   - Backpressure policy: queue when full, choose swap-vs-recompute on preemption
   verify: load test with mixed-length requests does not OOM; batch utilization stays high

4. (CUDA phase) Real PagedAttention kernel
   - Triton port, or integrate FlashInfer / xformers paged variant
   - Drops in behind the existing block-table abstraction; no scheduler changes required
   verify: per-step attention latency competitive with vLLM on equivalent hardware

5. Head-to-head vs vLLM (Phase 11)
   - Same model, same hardware, same workload mix
   - Metrics: throughput (req/s, tok/s), p50/p95/p99 TTFT and TPOT under concurrency,
     KV cache utilization, memory footprint, saturation point
   verify: numbers documented in BENCHMARKS.md alongside vLLM's
```

## Notes on integration with other engine components

- **Eviction policies (LRU / H2O):** operate at the block level inside the block pool. The block manager is the right home for the eviction interface; no changes to scheduler or attention kernel required.
- **Loki sparse attention:** reads K and V through the same block-table indirection. A paged backend is strictly more flexible for Loki than a contiguous one because Loki's gather is already indirection-based.
- **Modal deployment:** the block pool is in-process and per-container, consistent with the constraint in CLAUDE.md. Multi-replica scaling would require sticky session routing on `session_id` rather than any change to the block layer.
