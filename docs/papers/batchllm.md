# BatchLLM: Optimizing Large Batched LLM Inference

**Link:** https://arxiv.org/abs/2412.03594
**Venue:** MLSys 2026
**Authors:** Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, Gang Peng
**Status:** not read
**Relevant phase:** Phase 5 (prefix sharing) + Phase 6 (continuous batching, scheduler)

## Core idea

Existing engines like vLLM and SGLang share KV prefixes opportunistically — typically via an LRU-managed prefix tree that catches reuse only when requests happen to arrive close in time. Under bursty multi-request workloads with many shared prefixes (system prompts, few-shot demonstrations, RAG templates), this opportunistic approach prematurely evicts prefixes that will be reused shortly and misses sharing opportunities entirely. BatchLLM treats prefix sharing as a **global scheduling problem**: it identifies common prefixes across the entire pending request set up front, groups requests by prefix, and reorders execution so requests that share a prefix run together. It also schedules requests with a higher decode-to-prefill token ratio first to keep the device fed with decode work while prefill chunks are interleaved.

## Why it matters

- Directly relevant to the project's radix-tree prefix sharing. The radix tree as designed catches prefix matches when they exist in the cache *at request arrival time*; BatchLLM's contribution is to *change the order of arrivals into the engine* so matches happen more often.
- The reported 1.3×–10.8× win over vLLM and SGLang on representative workloads is large enough to take seriously even after accounting for cherry-picked batches. The high end of that range is on workloads with heavy prefix overlap, which is exactly the multi-tenant / RAG / agent regime.
- The "decode-heavy first" scheduling rule is a concrete heuristic that fits cleanly into the planned `SchedulerInterface` — it is a policy, not an architectural change.

## Key mechanism

1. **Global prefix identification.** Across the current pending request set, common prefixes are extracted (the abstract describes this as global rather than per-arrival). Mechanically, this corresponds to building a prefix tree over the *queue* of pending requests, not just the cache, so the scheduler can see all sharing opportunities before dispatch.
2. **Prefix-grouped scheduling.** Requests sharing a prefix are scheduled together so the prefix's KV is computed once and reused by all members of the group while still resident.
3. **Decode-ratio prioritization.** Requests with a higher fraction of decode tokens (relative to remaining prefill chunks) are scheduled earlier. This keeps the in-flight batch dominated by decode work and lets newly-arrived prefill chunks interleave without bubbles.
4. **Memory-centric token batching.** Token-batch sizes are sized to fit the available KV/activation memory rather than a fixed shape, increasing GPU utilization on workloads with variable per-request memory demand.
5. **Anti-premature-eviction.** Because the scheduler knows which pending requests will reuse a prefix, it can pin or refcount that prefix's blocks so an LRU policy does not evict them out from under upcoming requests.

## Implementation sketch

```
1. Pending-queue prefix index
   - Build a radix tree over the queue (separate from the cache's radix tree, or unified)
   - Maintain it incrementally as requests arrive and depart
   verify: queries return the set of pending requests sharing a given prefix

2. Prefix-grouped scheduler policy
   - New SchedulerInterface implementation: GroupByPrefixScheduler
   - At each dispatch tick, prefer requests whose prefix is already resident or whose
     prefix has the largest pending-group size
   verify: synthetic workload with N requests sharing one prefix executes the prefix
           exactly once; baseline runs prefill N times

3. Decode-ratio prioritization
   - Per-request, track remaining_decode_tokens / remaining_prefill_chunks
   - Sort the dispatch candidate set by this ratio descending before applying
     fairness / priority-class constraints
   verify: under a mixed-length workload, average decode tokens per step increases
           vs FIFO

4. Pinning to prevent premature eviction
   - Refcount prefix blocks by the number of pending requests that will use them
   - Eviction policy must respect refcount > 0
   verify: under cache-pressure load, prefixes with pending consumers are not evicted

5. Benchmark
   - RAG-style workload (heavy prefix overlap) and agent-style workload (shared
     system prompt, varied user turns)
   - Compare engine throughput vs the engine's own baseline (no global prefix scheduling)
   verify: throughput improvement on prefix-heavy workloads, no regression on prefix-light ones
```

## Notes on integration with other engine components

- **Radix tree:** the queue-side prefix index can either be a second radix tree over pending requests or an extension of the cache-side tree with a "pending consumers" annotation per node. The latter is cleaner because eviction already needs to read it.
- **FairScheduler:** prefix grouping and per-session fairness can conflict — grouping all requests sharing a prefix together can starve a session whose requests are not in any group. Resolution lives at the policy layer: apply fairness as the outer constraint, prefix grouping as the tie-breaker.
- **Continuous batching:** decode-ratio prioritization is per-step and slots into the iteration-level admit decision without architectural change.
