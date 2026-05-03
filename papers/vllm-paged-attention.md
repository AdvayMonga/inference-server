# Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)

**Link:** https://arxiv.org/abs/2309.06180
**Venue:** SOSP 2023
**Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (UC Berkeley)
**Status:** understood, not yet implemented
**Relevant phase:** Phase 5 (KV cache) + Phase 6 (continuous batching, vLLM comparison)

## Core idea (in your words)

KV cache memory in naive serving is wasted three ways: internal fragmentation (over-reserving for max_tokens that never generate), reservation (slots held for future tokens), and external fragmentation (gaps between sequences). vLLM borrows the OS virtual-memory playbook: split the KV cache into fixed-size **blocks**, and let each sequence carry a **block table** mapping logical positions → physical block IDs. Blocks are allocated on demand, never contiguous, and can be **shared** across sequences when their prefixes match (parallel sampling, beam search, shared system prompts). The custom **PagedAttention** kernel reads K/V through this indirection so attention works directly on non-contiguous memory.

## Why it matters for this project

- This is the architectural foundation our Phase 5 work is moving toward — our planned `BlockManager`, `RadixTree`, and `CacheManager` mirror vLLM's block allocator, prefix-sharing mechanism, and `BlockSpaceManager`. Currently scaffolding only; nothing wired into generation yet.
- vLLM is our explicit benchmarking target — understanding the paper deeply tells us *why* we expect specific throughput/memory wins, not just that vLLM is faster.
- The block + indirection design is what makes continuous batching (Phase 6) practical at all: without paging, packing variable-length sequences into one batch wastes too much memory to be worth it.

## Key mechanism

1. **Paged KV cache.** Cache is a pool of fixed-size blocks (e.g., 16 tokens worth of K and V). A sequence's KV state is a **list of block IDs** (the block table), not a contiguous tensor. New blocks allocated only when the current last block fills.
2. **PagedAttention kernel.** Custom CUDA kernel that takes the block table + the global block pool and computes attention over the logical sequence by gathering through the indirection. This is the part that requires a kernel — naive PyTorch indexing kills throughput.
3. **Copy-on-write block sharing.** Multiple sequences can point to the same physical block (e.g., shared system prompt, beam search siblings). Reference counts on blocks. When a sequence needs to *write* into a shared block, it gets copied first.
4. **Continuous batching at the scheduler level.** Iteration-level scheduling: every step, the scheduler can admit new sequences, evict (swap-out / recompute) ones that no longer fit, and resume swapped sequences. Block paging is what makes admission/eviction cheap (just move block IDs around, no big tensor copies).
5. **Swapping vs recomputation.** When KV cache is full and a new sequence needs to run, vLLM either swaps blocks to CPU or drops them entirely and recomputes via prefill. Choice depends on block size and PCIe bandwidth.

## Open questions / things to figure out

- Our `BlockManager` allocates blocks but our attention path still uses HF's contiguous-cache attention. Implementing a real PagedAttention kernel on MPS is impractical (no Triton). Plan: keep the block bookkeeping correct now, defer the actual paged-attention kernel to the CUDA phase and use a "gather then attend" PyTorch path in the meantime for correctness.
- Block size tradeoff: smaller blocks reduce internal fragmentation but increase block table overhead and kernel launch cost. Paper uses 16. Is that right for our workloads / our model?
- Copy-on-write logic for prefix sharing — do we implement this now (paired with the radix tree) or wait until we actually have multi-user workloads that exercise it?
- Swap vs recompute policy — only matters once we have backpressure (Phase 6). Recompute is simpler and usually wins for short prefixes.
- How does paged attention interact with H2O / Loki? Both reorder which tokens are attended to; paged attention is about *where they live in memory*. Should compose cleanly but worth thinking through.

## Implementation sketch (alignment with our roadmap)

```
Pending — directly mapped to paper:
0. Block manager, radix tree, cache manager (Phase 5 stages 1-3)
   - Bookkeeping only; not yet integrated with generation
1. Wire block-paged KV into the generation loop (Phase 5 stage 7)
   - Replace HF's contiguous KV with our block pool
   - PyTorch "gather K/V from block table → attend" path (correctness, not speed)
   verify: equivalent outputs to baseline within tolerance

2. Reference counting + copy-on-write for shared blocks
   verify: two sessions with shared system prompt actually share blocks; mutation forks correctly

3. Continuous batching scheduler (Phase 6)
   - Iteration-level admit / preempt / resume
   - Swap-to-CPU OR recompute on backpressure
   verify: load test with mixed-length requests doesn't OOM, batch utilization stays high

4. (CUDA phase) Real PagedAttention kernel — Triton port or use FlashInfer / xformers paged variant
   verify: per-step attention latency competitive with vLLM on same hardware

5. Head-to-head vs vLLM (Phase 11)
   - Same model, same hardware, same workload mix
   - Throughput, p95 TTFT/TPOT under concurrency, KV cache utilization, saturation
```
