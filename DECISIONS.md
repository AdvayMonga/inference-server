# Decisions Log

> **Generated file — do not edit.** Source of truth is `knowledge/*.json`.
> Regenerate with `python -m inference_server.research.loop index`.

68 entries. Tags: `kv`(29), `benchmark`(27), `kernel`(26), `prefill`(25), `cache`(23), `decode`(21), `modal`(21), `graph`(18), `scheduler`(16), `compile`(13), `memory`(13), `numerics`(10), `loop`(10), `backpressure`(8), `triton`(4), `roofline`(4), `quantization`(3), `attention`(3), `metrics`(2), `observability`(2), `attribution`(2), `batching`(2), `slo`(2), `benchmarking`(2), `tpot`(2), `torch-compile`(1), `flash-attention`(1), `wave-planning`(1), `capture-order`(1), `corrected`(1), `gates`(1), `resolved-noise`(1), `knowledge-base`(1), `refined`(1), `kv-cache`(1), `bug`(1), `throughput`(1), `measurement-gap`(1), `admission`(1), `config`(1), `variance`(1), `harness`(1), `validity`(1), `scheduling`(1)

Grep by tag or title rather than reading top-to-bottom.

## Open (19)

Live — being worked, or waiting on a trigger.

### [2026-09-06] pool_utilization reports a constant 0.5002 across every load level — likely not live
*tags: `metrics`, `kv-cache`, `observability`, `bug`, `attribution`* · `kb-20260906-22d4a185`

Across rates 8, 16, 32, 64 and 128 req/s — a 16x range in offered load, with kv_admit_blocked climbing 0 -> 1176 over the same span — pool_utilization reads exactly 0.5002 in all five panels. A KV pool genuinely under increasing pressure should not hold four decimal places constant.

Computed at paged_kv_cache.py:215 and :380 as round(1 - free/total, 4). Either the sampled pool is not the one the scheduler allocates from, or the value is read at a point where it has been reset. Cache hit_rate on the same panels moves sensibly (0.013 -> 0.0034 as load rises), so the cache stats path as a whole is not dead.

IMPACT: pool_utilization is one of the inputs attribute.py uses to decide whether a run is KV-pressure-bound, so a stuck value can misdirect a whole iteration. Treat any KV-pressure conclusion drawn from it as unproven until this is resolved.

**Revisit when:** a run where pool_utilization differs across rates would close this; check whether BlockPool.stats() is called on the pool the scheduler uses

**Evidence:** run-20260906-a0bde214, run-20260906-8c2a0890, run-20260906-701dfded, run-20260906-1862aada, run-20260906-c01ca7c6

### [2026-09-06] max_queue_wait_s=30s makes admission shedding useless — overload becomes 30s latency
*tags: `backpressure`, `scheduler`, `slo`, `admission`, `config`* · `kb-20260906-7efc7fcd`

The admission deadline is 30.0s while the TTFT budget is 200ms — 150x apart. Under overload requests therefore sit in the queue for the full deadline before being shed: TTFT p50 measured 9.5s / 20.7s / 29.7s / 29.9s at rates 16/32/64/128, i.e. equal to the test window. Shedding does fire (expired: 0/0/200/1220/3779, exactly equal to rejected, so every rejection is a deadline expiry rather than a full queue) but only after the user has already waited 30 seconds to be told no.

The mechanism is correct; the threshold defeats its purpose. LOOP.md frames backpressure as turning overload into 'bounded latency plus refusals' — a 30s bound is not a bound any caller wants. A deadline should be a small multiple of the TTFT SLO.

NOT yet established: whether tightening it improves throughput or only latency. Shedding earlier frees queue memory and scheduler work, but capacity is currently limited by wave size, not queueing — see the K=1 entry.

**Revisit when:** an A/B of max_queue_wait_s at 1-2s vs 30s under rate>=32 would settle it

**Evidence:** run-20260906-a0bde214, run-20260906-8c2a0890, run-20260906-701dfded, run-20260906-1862aada, run-20260906-c01ca7c6

### [2026-09-06] Throughput plateaus at ~23 req/s, but batch occupancy was never measured — cause unresolved
*tags: `scheduler`, `throughput`, `observability`, `batching`, `measurement-gap`* · `kb-20260906-6000f7f5`

CORRECTED. This entry first concluded 'the batch never fills, and that is the bottleneck'. That conclusion is NOT supported by the data and has been withdrawn.

WHAT IS MEASURED (open-loop sweep at 1a1c141, rates 8-128 req/s, single replicate): throughput plateaus at 163.9 -> 612 -> 620 -> 782 -> 665 tok/s for rates 8/16/32/64/128. Completed requests flatten at ~700-740 per 30s window = ~23-25 req/s capacity regardless of offered load. At rate 128, 2985 requests were queued. Offered load beyond ~16 req/s buys queue depth and latency, not work. The plateau is real.

WHY THE K=1 READING WAS WRONG: 49% of prefill waves being K=1 at rate 128 was read as a starved batch. It is equally consistent with a FULL batch: once max_batch_size is reached, each completing request frees exactly one slot, so the next admission wave is K=1 by construction. At steady state on a full batch, K=1 waves are CORRECT behaviour, not a symptom. wave_sizes measures prefill wave width; it says nothing about batch occupancy.

WHY IT CANNOT BE SETTLED FROM THESE PANELS: the panel has no measure of batch occupancy. `active_size` is len(self._active) sampled when stats() is called — after the window ends and the queue has drained — so it reads 0 in every panel. There is a high-water mark for PENDING but none for ACTIVE. Worse, the panel's `concurrency_observed` is populated from `pending_high_water` (bench_serving_modal.py:273), i.e. queue depth, not concurrency — the exact field LOOP.md lists as catching 'claiming a concurrency the run never reached'.

So the engine's central question for a batching server — how full was the batch — is not answerable from any run recorded so far. Fix the instrument before drawing the conclusion.

**Revisit when:** a mean/high-water active_size in the panel would settle whether the batch fills; if occupancy is high and throughput is still ~780 tok/s, the gap is per-step overhead; if occupancy is low with thousands queued, admission or wave planning is the cause

**Evidence:** run-20260906-a0bde214, run-20260906-8c2a0890, run-20260906-701dfded, run-20260906-1862aada, run-20260906-c01ca7c6

### [2026-09-06] The TPOT SLO (p95 < 50ms) is unreachable, so throughput-within-SLO is always None
*tags: `slo`, `metrics`, `benchmarking`, `tpot`, `loop`* · `kb-20260906-9454d8a1`

All 24 panels across 4 sweeps at 5681560 report tok_s_within_slo=None and saturation_knee_rate=1.0 — i.e. the budget is broken at the LOWEST rate measured, with essentially one user. The binding constraint is TPOT, not TTFT, and not load: TPOT p50 is 113ms at rate 1 on a slow draw and 45.7ms on the fast draw, against a 50ms budget. Even the fast draw fails at p95 (52.1ms).

50ms/token = 20 tok/s per user, which is roughly the best a fast A100 draw delivers for E4B here. The SLO is therefore set at best-case machine capability with zero headroom, so the project's primary metric — throughput at a fixed tail-latency budget — is undefined on every run and cannot discriminate between engine versions.

This is a MEASUREMENT problem before it is a performance problem. Either the budget is recalibrated to something the hardware can hold, or the headline metric stays None forever and the loop keeps judging secondary metrics as it has for ten iterations.

**Revisit when:** decode step latency dropping below ~40ms p95 would make the current budget live; a decision to recalibrate slo_tpot_ms would close this

**Evidence:** run-20260906-15a33a53, run-20260906-b2d3a54f, run-20260906-47218cd4, run-20260906-5c565305, run-20260906-a494262c, run-20260906-eb44c245, run-20260906-c0f72f5a, run-20260906-52fd6e63, run-20260906-3d0d3821, run-20260906-46c24518, run-20260906-dddc76e3, run-20260906-b8d219ff, run-20260906-c5d86987, run-20260906-af161f76, run-20260906-ef337ea0, run-20260906-f00be70c, run-20260906-23dea7a8, run-20260906-274a67a2, run-20260906-c7bcd8e0, run-20260906-ce942b4d, run-20260906-fead7bcd, run-20260906-6e3a6297, run-20260906-91d88a72, run-20260906-f434514f

### [2026-09-05] The prefill attention lever is head_dim 512 (full-attention layers), not 256
*tags: `prefill`, `kernel`, `attention`, `triton`, `roofline`* · `kb-20260905-b170a1ac`

Sliding layers cap attention at `window` keys (512), so their attention cost is bounded regardless of prompt length. Full-attention layers (head_dim **512**, no window) attend to the whole prompt, so on an 824-token prompt they dominate the 63% of prefill time that attention takes.

Tiling currently wins 1.57x at D=256 and LOSES 0.53x at D=512, because the [BLOCK_M, D] accumulator must shrink to BLOCK_M=16 and spills. So the tiled kernel is fast exactly where the work is bounded, and slow exactly where the work is.

**Next lever:** make tiling work at D=512 — chunk the head dimension so the accumulator fits (process D in two halves of 256, or keep acc in shared memory), then re-A/B. Until then the isolated 1.57x is not reachable end-to-end.

**Revisit when:** a D=512 tiled variant beats the untiled kernel in isolation

**Evidence:** exp-20260905-26c0e0f5, scripts/gpu_tests/test_tiled_prefill_modal.py

### [2026-09-05] Prefill is 4.4x off its arithmetic ceiling (tier-1 sizing; see refinement for where it goes)
*tags: `benchmark`, `kernel`, `prefill`, `refined`, `roofline`* · `kb-20260905-dcb78725`

At the observed workload (lognormal 5.48/0.75: p50 240 tok, p95 824 tok) on A100/E4B:

| prompt | attention | GEMM | total | floor @50% peak |
|---|---|---|---|---|
| p50 240 | 0.01 TFLOP | 2.20 | 2.21 | 14.2 ms |
| p95 824 | 0.12 TFLOP | 7.56 | 7.68 | 49.2 ms |

Measured prefill p95 is **218 ms vs a 49 ms floor = 4.4x**. Attention is only **1.5%** of the FLOPs, so the long-standing assumption that the prefill tail is irreducible O(S^2) attention is false — the dense GEMM dominates and the gap is overhead.

Sensitivity: at a more realistic 25% of peak for this stack the floor is ~98 ms and the gap ~2.2x. Still a real lever either way.

**This is where prefill work should go.** Not attention kernels, not scheduling (ruled out separately: queue p95 21 ms vs prefill p95 203 ms), not bucket boundaries (falsified below).

**REFINED by tier 3 (iteration 2).** The FLOP arithmetic above is correct but the conclusion drawn from it was not. Profiling one graphed prefill by op category:

| tokens | wall | attention | GEMM | norm/rope | kv_scatter |
|---|---|---|---|---|---|
| 240 | 41.4 ms | 12.7 (31%) | 12.9 (31%) | 7.3 | 5.2 |
| 512 | 95.0 ms | 51.3 (55%) | 22.6 (24%) | 10.1 | 6.8 |
| 824 | 178.0 ms | **112.6 (63.5%)** | 35.1 (20%) | 16.1 | 9.6 |

**Attention IS the dominant cost** — while being 1.5% of the FLOPs. So the original claim ('the tail is attention') was right about WHERE and wrong about WHY; my correction was right about the FLOPs and wrong to conclude attention was not the cost. Both were wrong. The kernel is simply extremely inefficient, and that makes it reclaimable, not irreducible.

**Revisit when:** a tier-3 probe attributes the 4.4x to specific ops

**Evidence:** scripts/probes/probe_prefill_ceiling.py, runs/ (rate 2, group=first-real-panel)

### [2026-09-05] The prefill attention kernel has no data reuse — 0.34% of peak, 63% of the time
*tags: `prefill`, `kernel`, `attention`, `triton`, `roofline`* · `kb-20260905-b9bc66c6`

Measured on A100/E4B, one graphed 824-token prefill: attention takes **112.6 ms of 178 ms (63.5%)** while doing **0.12 of 7.68 TFLOP (1.5%)** of the work. That is **1.07 TFLOPS, or 0.34% of A100 peak** — roughly 100x off what the arithmetic allows.

Time also scales superlinearly while FLOPs barely move: 12.7 -> 51.3 -> 112.6 ms for 240 -> 512 -> 824 tokens.

**Cause (from the kernel's shape, to be confirmed):** `_paged_prefill_kernel` launches one program per (sequence, head, query token) and each program walks its own row's KV blocks serially. Nothing is shared between query tokens, so K/V blocks are re-read from HBM once per query — O(S^2) memory traffic for O(S^2) *arithmetic that is tiny*. This is exactly the problem FlashAttention-style tiling solves: block the queries, load each K/V tile once, reuse it across the whole tile.

**This is now the single biggest lever in the engine.** TTFT p95 is the only thing breaking the SLO, prefill is 81% of that p95, and attention is 63% of prefill. Nothing else measured comes close.

Not to be confused with the DECODE attention kernel, which was already given split-K (7.09x at n=1) — decode has one query per row, so it has no reuse to exploit. Prefill has S queries and currently exploits none of it.

**Revisit when:** a tiled prefill attention kernel is written and A/B'd against the current one

**Evidence:** scripts/probes/probe_prefill_breakdown_modal.py, iter2-prefill-breakdown

### [2026-09-05] LOOP: the rehash detector has false positives AND false negatives
*tags: `loop`, `knowledge-base`* · `kb-20260905-521365f0`

First real iteration, step 3. `already_rejected()` matches on keyword overlap and got both directions wrong:

- **False positive:** flagged the roofline hypothesis as a rehash of 'length-grouped prefill waves' on shared words. That hypothesis was the one that found the 4.4x gap.
- **False negative:** did NOT flag 'chunk long prefills across ticks', which is a scheduling lever that `kb-20260903-001` already rules out (queue p95 21ms vs prefill p95 203ms).

A detector that flags good ideas and misses bad ones is worse than none — it teaches you to ignore the flag. It should surface *related* entries across all statuses for reading, ranked by relevance, and stop claiming a verdict.

**Revisit when:** it fires wrongly again on a real iteration

### [2026-06-11] Custom prefill mirrors two pre-existing edge bugs (M2.4)
*tags: `benchmark`, `cache`, `decode`, `kv`, `prefill`* · `kb-20260611-039`

Context: to keep M2.4 a pure plumbing change, `CustomTorchBackend.prefill`/`prefill_chunk` mirror `generate()`/`TorchBackend` behavior exactly rather than fixing two latent edges. (1) Full-cache-hit (`matched == len(prompt)`, only possible when the prompt is an exact multiple of `block_size` and fully cached) resets `matched = len-1` and re-forwards the last token at position `len` (should be `len-1`) — double-counts one position. Rare. (2) The first token after prefill is sampled greedily (the `prefill` ABC method takes no `sampling` arg), ignoring `req.sampling`; only decode tokens honor temperature/top-k/top-p. Both are identical to the existing `TorchBackend` path. **Trigger:** if either bites a correctness test or a sampling-quality benchmark. Fix for (2) would thread `sampling` through `prefill` for both backends (interface change).

### [2026-06-11] Custom backend skips scheduler-visible KV backpressure (M2.4)
*tags: `backpressure`, `cache`, `kv`, `memory`, `scheduler`* · `kb-20260611-038`

Context: the custom backend caches via its own `PrefixCache` + paged pools, not the HF-format `CacheManager`. So `set_cache_adapter` is a no-op (`cache_adapter` stays `None`), and the scheduler's cache-pool admission gate (`blocks_needed` vs `free_blocks`) + `release()` are skipped for this backend — only the active-KV-token gate and queue-size limit apply. Pool exhaustion surfaces as a caught `BlockPool` `RuntimeError` in `_admit_pending` → request rejected (degrades, doesn't crash). Also: `/cache/stats` reports the unused `CacheManager`, not the real `PrefixCache` hits/lookups; and `_fail_all` (scheduler-crash path) nulls `_batched_kv` without `free_all()` → blocks leak only on a scheduler crash. **Trigger:** if the CUDA load-test shows KV-pressure churn or we want graceful backpressure for `custom-*`. Fix = expose pool free-count/`blocks_needed`/`total_blocks` from the backend (a thin `cache_adapter`-shaped view over the pools) so the existing gates light up; surface `PrefixCache` stats on `/cache/stats`; free pools in `_fail_all`.

### [2026-05-14] Batch utilization % metric

Derivable from `active_size / max_active`. Add when dashboard needs the rolled-up number.

### [2026-05-14] `COMPILE_MODEL` default off
*tags: `benchmark`, `compile`* · `kb-20260514-026`

MPS `torch.compile` is shaky; continuous-batching's dynamic batch shape can trigger recompile storms. Flip on after CUDA deploy and benchmark.

### [2026-05-14] Eviction quality benchmarks not run
*tags: `benchmark`* · `kb-20260514-021`

`eviction_benchmark.py` measures throughput/latency/hit-rate, not output quality over N turns. **Trigger:** picking a default policy for production, or H2O wiring lands.

### [2026-05-14] Per-row `last_cache_hit_tokens` overwritten in `generate_batch`
*tags: `cache`* · `kb-20260514-020`

Only last row's value survives. Not user-facing (no endpoint routes through `generate_batch`). **Trigger:** any metric starts depending on per-row hit accounting there.

### [2026-05-14] H2O `record_attention()` never called
*tags: `benchmark`, `kernel`, `kv`* · `kb-20260514-018`

H2OPolicy stores scores but no backend code extracts attention weights → H2O behaves like degenerate-LRU. Fix needs `output_attentions=True` + per-block aggregation through HF forward path. Defer until H2O is load-bearing for a benchmark.

### [2026-09-02] Batched prefill KV != single-row prefill KV — INVESTIGATED, BENIGN
*tags: `kv`, `modal`, `numerics`, `prefill`, `scheduler`* · `kb-20260902-007`

**Symptom.** `prefill_batch([p])` and `prefill_batch([p0..pK])` write measurably different KV for
the same `p`: relative error 0.25-1.4, far beyond bf16 rounding. Alarming because
`PREFILL_MODE=batched` is what the serving config runs, so a real bug here would invalidate every
serving number.

**Not a regression.** `scripts/gpu_tests/test_prefill_ctx_scatter_modal.py` produces byte-identical numbers
on main and on the vectorized-scatter branch.

**Localized** (`scripts/probes/diag_batched_prefill_kv_modal.py`): divergence appears ONLY in deep layers
(7-14, never 0-6) at 1-7 isolated token positions out of 128-188. A scatter/indexing bug would hit
layer 0 and contiguous ranges.

**Decisive test** (`scripts/probes/diag_prefill_crossrow_modal.py`): run the same wave twice, same K and
same Smax, holding the probe row identical and changing only its NEIGHBOURS' content. Probe-row KV
came back **bit-identical, max abs diff 0.000000, first token unchanged**. No cross-row dependence.

**Conclusion:** batching changes GEMM shapes, which changes rounding below threshold at layer 0;
RMSNorm / GeGLU / softmax amplify it over 15 layers until isolated positions blow up in relative
terms. Same family as the batch-invariance entry below. No action; do NOT gate tests on
"batched KV == single-row KV" (the existing K=3 unit test already hedges this with `matches >= 2`).

### [2026-09-02] The serving harness is a prefix-cache-HIT benchmark
*tags: `benchmark`, `cache`, `graph`, `kv`, `modal`, `prefill`* · `kb-20260902-005`

`bench_serving_modal.py` draws from a pool of 64 prompts (`POOL_SIZE = 64`) against a PrefixCache
with no eviction. After warmup nearly every request is a cache HIT with a tiny suffix. Consequences
we measured directly:
- Extending prefill graph buckets to 1024 for long prompts is **neutral** here (255.5 vs 256.2
  tok/s at rate 2, TTFT p95 239 vs 238, at 164 requests) — the suffixes are too short to reach
  the new buckets. It should matter on real traffic, which misses far more often.
- Conversely the prefix-hit graph work is *over*-rewarded here relative to a realistic miss rate.
**Before trusting any prefill number as representative, raise POOL_SIZE or add cache eviction.**

### [2026-09-02] Benchmarks must state their cache-hit regime
*tags: `benchmark`, `cache`, `graph`, `kv`, `modal`, `prefill`* · `kb-20260902-004`

`bench_serving_modal.py` at the default `POOL_SIZE=64` is a prefix-cache-HIT benchmark: after
warmup nearly every request hits with a tiny suffix. That flatters prefill work (the prefix-hit
graph especially) and completely hides miss behaviour, which is where the KV bugs above lived.
`BENCH_POOL_SIZE` now raises it; `bench_stress_modal.py` covers misses and overload. **Quote the
regime alongside any prefill number.**

### [2026-09-03] The two graph ladders are tuned in OPPOSITE directions — deliberate
*tags: `cache`, `compile`, `decode`, `graph`, `modal`, `prefill`* · `kb-20260903-002`

Decode graphs go through `torch.compile`, so each bucket costs a fresh compile (~140s, linear in
bucket count: 2 buckets 285s, 5 buckets 574s). A persistent Inductor cache on a Modal volume
recovers only ~19% (705s -> 574s) because most of the cost is Dynamo tracing, not codegen. So the
DECODE ladder is coarse when compiling (powers of four) and fine otherwise. Prefill graphs do NOT
go through compile — all eight capture in 6.9s — so the PREFILL ladder is fine-grained, because
there a graph processes its whole bucket and padding is pure waste. Do not "unify" them.

## Deferred (9)

Deliberately not doing this yet; each carries the trigger that would change that.

### [2026-05-18] vLLM head-to-head — use Modal `stopwatch` / `guidellm`
*tags: `benchmark`, `modal`* · `kb-20260518-042`

Final benchmark for the vLLM head-to-head will run through Modal's `stopwatch` (wraps Red Hat's `guidellm`) so numbers slot directly into the LLM Almanac table — apples-to-apples by construction, no need to defend a homegrown harness. `scripts/bench/load_test.py` stays as the dev-time smoke test; not replaced. Cost: `guidellm` speaks the OpenAI API, so a narrow `/v1/completions` shim on our server (~100 lines, engine-only — no auth/registry/etc.) will be needed to drive it. Shim is a benchmark-compatibility surface, not a public API contract. **Trigger:** Phase 11 — after Modal deploy + load-test sweep on CUDA. Revisit shim yes/no then.

---

### [2026-06-11] Prefer prefill/decode disaggregation over mixed-batch prefill
*tags: `decode`, `graph`, `kernel`, `kv`, `memory`, `prefill`* · `kb-20260611-040`

Context: weighing mixed-batch chunked prefill (V-B) vs other work. Mixed-batch exists mainly to stop a long prompt's prefill from stalling ongoing decodes (HOL blocking) when both share one GPU. The modern production direction is **P/D disaggregation**: run prefill on one GPU pool and decode on another, ship the KV blocks between them. Prefill is compute-bound, decode is memory-bandwidth-bound — separating them lets each be sized/optimized independently and removes the interference that mixed-batch works around. **Decision:** keep mixed-batch prefill (V-B) deferred; if/when prefill interference becomes the bottleneck, prefer disaggregation. V-B's varlen kernel is also the single biggest remaining lift (our Triton kernel is decode-only). **Trigger:** after CUDA graphs + a mixed/long-workload sweep, if TTFT-under-load is the gap. Disaggregation is a bigger architectural change (KV transfer, two pools, routing) — Phase 7+ territory; revisit with the platform layer.

### [2026-05-14] Per-session KV quotas
*tags: `kv`* · `kb-20260514-027`

Out of scope until engine done. `session_id` scoping already in place → additive when needed.

### [2026-05-14] Load-test real sweep
*tags: `modal`* · `kb-20260514-025`

Sweep + plot scripts shipped. MPS ~50 tok/s on E2B → saturation knee dominated by compute, not engine work. **Trigger:** Modal/CUDA deployment live.

### [2026-05-14] Preemption
*tags: `backpressure`, `kv`, `prefill`* · `kb-20260514-024`

Today: hard-reject only for impossible requests; soft-hold (HOL-wait) is behaviorally equivalent to preemption for "fits eventually". Preemption only matters when strict priority/SLAs need to bump active rows *now*. Design sketch when revisited: recompute-preempt (re-prefill on resume, right for short prompts) or swap-preempt (CPU swap pool, right for long prompts); victim selection must coordinate with FairPolicy; reservation accounting must release/re-add `(prompt_len + max_tokens)` from `_active_kv_reserved`. **Triggers:** strict priority semantics, per-tenant SLAs, or tail-latency-vs-vLLM gap traced to wait-or-reject.

### [2026-05-14] Chunked Prefill — Version B (mixed-batch, vLLM-style)
*tags: `benchmark`, `decode`, `kernel`, `kv`, `prefill`* · `kb-20260514-023`

Single packed forward combining decode + one prefill chunk via varlen attention. **Blocker:** HF `AutoModelForCausalLM.forward` requires rectangular inputs and treats every token as new KV — needs FlashAttention-varlen (CUDA-only) or a custom per-layer attention path. This is why vLLM is its own runtime. **Sequencing:** after CUDA swap, as part of vLLM head-to-head. V-A's `_prefilling` state + `prefill_chunk` primitive is what V-B extends. **Trigger:** CUDA live AND V-A shipped AND starting vLLM benchmark.

### [2026-05-14] MLX backend has no cache integration
*tags: `cache`, `kv`* · `kb-20260514-019`

`mlx_lm.stream_generate` owns its own KV cache. Bundle with the MLX-continuous-batching future extension (same work). MPS is primary backend. **Trigger:** MLX continuous batching becomes a priority.

### [2026-06-13] int8 weight-only quantization
*tags: `benchmark`, `compile`, `decode`, `kernel`, `kv`, `memory`, `modal`, `numerics`, `quantization`* · `kb-20260613-015`

Context: decode is weight-bandwidth-bound (full-batch sweep: flat ms/step, 1481 tok/s @ N=32 A10G/E2B), so int8 weights = ~2× fewer bytes/step = the real lever.
**Built (`models/quant.py`, committed):** per-output-channel symmetric int8 RTN — `QuantizedLinear` + `quantize_model_int8` (in-place `nn.Linear` swap, `lm_head` excluded — tied to embeddings). Knob `CUSTOM_BACKEND_QUANT=int8`. **Accuracy near-lossless** (Gemma4, vs bf16): 0.93–0.96 top-1, 0.9999 logit cosine, ~0.97 ppl ratio — softcap/QK-norm/per-layer-embedding quirks don't break it.
**Why deferred:** the CUDA throughput kernel doesn't land on A10G. `torch._weight_int8pack_mm` is CPU/MPS-only (no CUDA kernel). The CUDA win needs the dequant fused into the GEMV via `torch.compile`; on A10G that OOMs during Inductor autotune (24 GB, mostly KV pools), and `dynamic=True` (required to avoid a 276-layer recompile storm) doesn't fuse — it materializes the bf16 weight (no win). Static-shape fusion (gpt-fast) needs a fixed-shape model + memory headroom A10G lacks.
**Decision (with user):** A10G is validation hardware; defer int8 *throughput* + E4B + vLLM head-to-head to the A100 80 GB / H100 phase, where (a) no OOM wall and (b) the fused paths work (A100: torchao int8 / static compile; H100: fp8 — the better method for that GPU). **Trigger:** the A100/H100 move. **Then:** wire a fusing GEMV into `QuantizedLinear.forward` (torchao / fp8 / custom Triton W8A16), flip `bench_quant_modal.py` to `gpu="A100"`, measure ~2×. Foundation + accuracy done; only kernel + measurement remain.

### [2026-05-30] Persistent `torch.inductor` cache via Modal Volume
*tags: `cache`, `compile`, `graph`, `modal`* · `kb-20260530-013`

Idea: mount a Modal Volume at `~/.cache/torch/inductor/` so compiled graphs survive cold starts (~20s saved per fresh container). Same pattern as our hf-cache volume. **Trigger:** revisit only after compile actually works on Gemma 4 (today it crashes before producing any artifact to cache, so persistence is moot). ~5 lines when the time comes.

## Resolved (31)

Settled. Kept because the reasoning still constrains new work.

### [2026-09-06] No saturation knee exists between 1 and 8 req/s — batching is working
*tags: `scheduling`, `batching`, `attribution`, `decode`, `tpot`* · `kb-20260906-f13d8e3d`

Across all four sweeps, TPOT p50 is FLAT or IMPROVING as request rate rises (113->87, 92->76, 128->84 ms on the slow draws) while throughput rises 1.6-2.2x (54.7->89.3, 60.2->133.2, 52.4->82.5, 97.7->187.6 tok/s). That is continuous batching doing exactly what it should: more concurrent rows amortize the same per-step cost.

CONSEQUENCE: there is no concurrency cliff in the measured range, so scheduling levers (fairness policy, wave shaping, admission tuning, queue depth) cannot move the headline number. The cost is per-decode-step latency at LOW batch — a kernel/model cost, not a scheduling cost. This is the same shape as the earlier 21ms-queue vs 203ms-prefill split, which killed an entire class of planned fixes.

The measured range stops at 8 req/s; a knee may exist above it. TTFT p95 at rate 1 (1036-1416ms on slow draws, 142ms on the fast one) is a small-sample artifact — 24 requests means p95 is worst-of-24, dominated by the first request after warmup.

**Revisit when:** sweeping above 8 req/s could still reveal a knee; a workload with longer prompts would shift cost back toward prefill

**Evidence:** run-20260906-15a33a53, run-20260906-b2d3a54f, run-20260906-47218cd4, run-20260906-5c565305, run-20260906-a494262c, run-20260906-eb44c245, run-20260906-c0f72f5a, run-20260906-52fd6e63, run-20260906-3d0d3821, run-20260906-46c24518, run-20260906-dddc76e3, run-20260906-b8d219ff, run-20260906-c5d86987, run-20260906-af161f76, run-20260906-ef337ea0, run-20260906-f00be70c, run-20260906-23dea7a8, run-20260906-274a67a2, run-20260906-c7bcd8e0, run-20260906-ce942b4d, run-20260906-fead7bcd, run-20260906-6e3a6297, run-20260906-91d88a72, run-20260906-f434514f

### [2026-09-06] A100-80GB draws are bimodal: 2.31x TPOT spread at identical config
*tags: `benchmarking`, `modal`, `variance`, `harness`, `validity`* · `kb-20260906-a81f6d18`

Four replicate sweeps at 5681560, byte-identical harness_config (A100-80GB, E4B, pool_size=4000, batch=256, compile=0, prefill_graph=1), separate Modal containers. TPOT p50 at rate 2: 108.7, 90.2, 110.9, 48.0 ms — a 2.31x spread, and bimodal rather than a smooth distribution: three slow draws clustered at 90-111ms and one fast draw at 48ms. The same split appeared across earlier sessions (iter9 39-47ms vs iter10 103-119ms on identical config). 'A100-80GB' is not one performance class.

CONSEQUENCE: machine draw is a larger effect than any optimization this project has merged. Cross-container A/B is therefore worthless — a 2.3x confound swamps the effect being measured. Arms MUST run in one process on one GPU (as BENCH_AB_TILED does). Absolute throughput/latency figures are only meaningful with the draw quoted alongside, and a single-container baseline cannot be compared to anything.

NOT established: the cause. Machine draw is the leading hypothesis; a warm Modal volume is an alternative (the fast sweep ran last in its batch, but the slow sweep in the previous batch also ran last, which argues against ordering).

**Revisit when:** a run whose TPOT p50 falls between 60-85ms would break the bimodal claim; pinning a GPU by device id or reserving capacity would test the machine-draw cause

**Evidence:** run-20260906-15a33a53, run-20260906-b2d3a54f, run-20260906-47218cd4, run-20260906-5c565305, run-20260906-a494262c, run-20260906-eb44c245, run-20260906-c0f72f5a, run-20260906-52fd6e63, run-20260906-3d0d3821, run-20260906-46c24518, run-20260906-dddc76e3, run-20260906-b8d219ff, run-20260906-c5d86987, run-20260906-af161f76, run-20260906-ef337ea0, run-20260906-f00be70c, run-20260906-23dea7a8, run-20260906-274a67a2, run-20260906-c7bcd8e0, run-20260906-ce942b4d, run-20260906-fead7bcd, run-20260906-6e3a6297, run-20260906-91d88a72, run-20260906-f434514f

### [2026-09-05] An isolated kernel win of 10x can be worth nothing end to end
*tags: `kernel`, `benchmark`, `loop`, `prefill`* · `kb-20260905-d2a675d5`

The tiled prefill kernel is **10.84x** faster than the untiled one in a tight microbenchmark, and produced **no measurable end-to-end benefit** — a significant +24.3% regression on prefill p95, everything else noise.

Why that is not a contradiction: the kernel is routed to sliding layers (D=256) on long prompts (S>=512) only, so it touches a minority of layers on a minority of requests; and the A/B ran with the prefill graph off, where dispatch dominates and kernel time is a smaller share.

**Rule: a microbenchmark sizes a kernel, it does not justify shipping one.** Always carry the isolated number through to a replicated service-level A/B before defaulting it on. This repo now has two examples of an isolated win evaporating.

**Revisit when:** a kernel is defaulted on from microbenchmark evidence alone

**Evidence:** iter8-replicated

### [2026-09-05] Harness null variance on A100/E4B (calibration)
*tags: `loop`, `benchmark`, `roofline`* · `kb-20260905-1b1a2520`

Measured with three identical runs, rate 2, 60s, pool 4000, compile off, prefill graph on:

| metric | values | spread | cv |
|---|---|---|---|
| ttft_p50 | 179 / 157 / 121 | 1.48x | 19% |
| ttft_p95 | 1494 / 401 / 229 | **6.52x** | 97% |
| ttft_prefill_p50 | 69.3 / 47.6 / 40.1 | 1.73x | 29% |
| ttft_prefill_p95 | 436.6 / 289.9 / 136.9 | **3.19x** | 52% |
| tpot_p50 | 152.2 / 118.4 / 104.4 | 1.46x | 20% |
| tpot_p95 | 247.2 / 173.6 / 110.0 | 2.25x | 39% |

**Implications.** p95 metrics are close to unusable as A/B endpoints at this run length — prefer p50, or replicate heavily. No claim below roughly 2x on a p95 means anything from single runs. Every earlier conclusion of that size on a p95 should be treated as unproven, including this session's merge.

Re-measure this calibration whenever the harness or instance type changes.

**Revisit when:** harness, run length or instance type changes

### [2026-09-05] Sequential A/B arms are warmup-biased; the second arm always wins
*tags: `loop`, `benchmark`* · `kb-20260905-38efb550`

Three identical back-to-back runs in one container: ttft_p95 1494 -> 401 -> 229, tpot_p50 152 -> 118 -> 104. Strictly decreasing. Something keeps warming — allocator, Triton/JIT caches, prefix cache, GPU clocks.

Interleaving arms in one container fixed the MACHINE confound but introduced an ORDER confound, and the order confound favours whichever arm runs second — the treatment, by convention. **Arms must be replicated AND order-alternated (A,B,B,A,...), with the first run of a container discarded as warmup.**

**Revisit when:** an A/B runs arms in a fixed order

### [2026-09-05] LOOP: the significance gate measured the WRONG variance
*tags: `loop`, `benchmark`, `gates`* · `kb-20260905-1f0727be`

`Validity.stderr` was computed from the TTFTs of individual requests WITHIN one run. The significance gate then used it as the variance of the run's p95. Those are different quantities by a wide margin: within-run request scatter is small, while run-to-run p95 varies **3.2x** on this harness with nothing changed.

So the gate was confidently significant (|t|=2.96) about an effect it had no way to detect. It authorised a merge on noise.

**Rule: significance requires REPLICATE RUNS per arm, and the variance must come from across those runs.** A single run per arm can never clear a p95 metric here, and the gate must say so rather than substitute a convenient number.

**Revisit when:** a gate reports significance from a single run per arm

### [2026-09-05] LOOP: never judge a kernel before sweeping its launch config
*tags: `loop`, `kernel`, `benchmark`* · `kb-20260905-a474d802`

Iteration 4 declared a correct, well-motivated kernel 'noise' end-to-end. It was noise — at BLOCK_M=32, which I picked by hand. A ~10-minute sweep found BLOCK_M=16/warps=4 is **6x faster** (10.84x vs 1.81x isolated), and the same A/B then gave -31.9% and merged.

**Rule:** a launch-config sweep is a tier-3 step that belongs BEFORE the end-to-end A/B, not after a disappointing one. Hand-picked tile sizes are a guess, and judging a kernel on a guess produces a confident false negative — the most expensive kind, because the idea gets written off.

**Revisit when:** a kernel is judged without a config sweep

**Evidence:** scripts/probes/sweep_tiled_launch_modal.py, exp-20260905-0d85ab30

### [2026-09-05] LOOP: three A/B attempts, three confounds — the gates earned their keep
*tags: `loop`, `benchmark`* · `kb-20260905-c6904d17`

Iteration 3 tried to land a kernel that is correct and 1.57x faster in isolation. Each end-to-end A/B was invalid for a different reason:

1. **Separate containers.** Same `run_group` label, different machines. Gates said CONFIRMED -48.2%; TPOT p50 differed 95.3 vs 46.4 ms between arms that differ only in a *prefill* flag, which is impossible. Caught by sanity, not by the gate.
2. **Warm cache between arms.** Arms run in sequence share a prefix cache, so the second measures a different workload. **Caught by the validity gate** (cache_miss_heavy vs mixed) — refused before it could mislead.
3. **CUDA-graph capture.** The flag flip had no effect on graphed prefills.

**Lesson:** an isolated kernel win is not evidence for a merge, and 'all four gates green' is only as strong as the arms. Add a same-run sanity check — a metric the change CANNOT affect (here TPOT for a prefill change) must not move; if it does, the arms are contaminated regardless of what the gates say.

**Revisit when:** gates pass while an unrelated metric moves

### [2026-09-05] LOOP: an in-process A/B is defeated by CUDA-graph capture
*tags: `loop`, `benchmark`, `graph`* · `kb-20260905-c1c2a5d5`

Flipping a module flag between arms inside one container does NOT change behaviour for anything already captured into a CUDA graph: the graph replays whichever kernel was active at capture time. The tiled-prefill treatment arm therefore mostly ran the UNTILED kernel and the comparison was meaningless.

**Rule:** an A/B toggle must take effect before capture, or the arms must be separate processes with the flag set from the environment — and then the machine confound returns, so the harness must additionally re-measure the baseline in each container. Any flag consumed inside a captured graph cannot be A/B'd by flipping it at runtime.

**Revisit when:** a future in-process A/B touches a graphed code path

### [2026-09-05] LOOP: tier-1 arithmetic can locate a gap but must not be used to attribute it
*tags: `loop`, `benchmark`* · `kb-20260905-ae5fd6c2`

Iteration 1 computed FLOPs, found prefill 4.4x off its ceiling, and concluded 'attention is only 1.5% of the work, so attention is not the cost'. Iteration 2 profiled it and found attention is 63.5% of the time. Both statements are true: the kernel does 1.5% of the FLOPs in 63.5% of the time.

**Rule:** FLOP arithmetic answers *is there a gap and how big*. It cannot answer *where the gap is*, because it assumes every op runs at the same efficiency — and the whole point of a gap is that something does not. Tier 1 sizes the prize; tier 3 attributes it. Never let a tier-1 result close an attribution question.

The ladder still worked exactly as intended — tier 1 cost $0, killed two hypotheses, and pointed at the right area. It was the *conclusion drawn* that overreached.

**Revisit when:** a future iteration attributes cost from arithmetic alone

### [2026-09-03] TTFT p95 is prefill COMPUTE, not queueing
*tags: `corrected`, `kernel`, `prefill`, `scheduler`* · `kb-20260903-001`

`admit_ts` splits TTFT into queue-wait and prefill. At the operating point the tail is entirely
prefill:

| rate | TTFT p95 | queue p95 | prefill p95 |
|---|---|---|---|
| 1 | 213ms | 21ms | 203ms |
| 2 | 315ms | 68ms | 292ms |
| 3 | 342ms | 165ms | 254ms |

**This rules out a whole class of fixes for this workload** — scheduling changes, admission
tuning, chunked prefill V-A — because they address queueing, which contributes 21ms of a 213ms
p95 at rate 1. The p95 prompt is ~830 tokens and most of its cost is real O(S^2) attention.
Refining the prefill bucket ladder (mean padding 17.6%) improved prefill p50 by 10-20% but moved
p95 by ~1%. **Rate 1 sits at TTFT p95 204ms against a 200ms budget; closing it needs a faster
long-prompt prefill kernel, not more scheduling.**

**CORRECTION (2026-09-05, loop iteration 1, tier 1, $0).** The claim above that the p95 tail is 'real O(S^2) attention' was asserted without doing the arithmetic, and it is wrong. At the p95 prompt (824 tok) attention is **1.5% of the FLOPs** (0.12 of 7.68 TFLOP); the dense GEMM dominates. Measured prefill p95 218ms against a 49ms arithmetic floor is **4.4x off the ceiling**, so the tail is reclaimable overhead, not irreducible attention. The conclusion 'this needs a faster attention kernel' does not follow. See `scripts/probes/probe_prefill_ceiling.py`.

### [2026-05-14] Preemption skipped in initial FairScheduler
*tags: `backpressure`, `scheduler`* · `kb-20260514-047`

Skipped at ship time; vLLM also treats as optional. Full design sketch + revisit triggers under the open "Preemption — DEFERRED" entry above.

### [2026-05-14] KV-pressure-aware admission gates
*tags: `backpressure`, `benchmark`, `cache`, `decode`, `kv`, `memory`, `scheduler`* · `kb-20260514-046`

Two gates in `_admit_pending` (peek → fit-check → pick):
- **Active-KV gate** (decode-OOM): tracks `(prompt_len + max_tokens)` reservations across `_active`; soft-hold when `reserved + new > MAX_ACTIVE_KV_TOKENS`; hard-reject single requests > budget. Load-bearing — `_batched_kv` is scheduler-local, separate from CacheManager blocks.
- **Cache-pool gate** (eviction-thrash): soft-hold when `blocks_needed > free_blocks`; hard-reject when `> total_blocks`.
Metrics in `/scheduler/stats`: `kv_pressure`, `kv_free_blocks`, `kv_admit_blocked`, `active_kv_reserved`, `active_kv_budget`. Validated by `scripts/bench/kv_pressure_benchmark.py`.

### [2026-05-14] bf16 weights, not FP16
*tags: `memory`, `numerics`* · `kb-20260514-045`

Roadmap "FP16" was outdated shorthand. `dtype=torch.bfloat16` already in `load_model`. Same memory as FP16; modern default.

### [2026-05-14] `CacheManager` exact-match index in front of radix
*tags: `cache`, `kv`* · `kb-20260514-044`

Radix `insert` silently bailed on mid-block divergence (chat-template prefix), causing repeat-prompt misses. Two-layer lookup (exact dict → radix prefix) now matches SGLang/vLLM shape.

### [2026-05-14] Pre-allocated per-layer KV pools
*tags: `cache`, `kernel`, `kv`, `memory`* · `kb-20260514-043`

`BlockManager` owns per-layer `k_pools[l]`, `v_pools[l]` shaped `(num_blocks, n_kv_heads, block_size, head_dim)`. Allocated once; `Block.k_tensor`/`v_tensor` are pool slices; `_store_kv_in_blocks` does in-place `copy_`. Eliminates per-request alloc churn; predictable peak memory.
**Per-layer (not stacked)** because Gemma 4 is heterogeneous (12 layers `(1,256)`, 3 layers `(1,512)`). Probed via `TorchBackend.kv_shape_per_layer()` — works for any HF causal LM.
**vLLM forward-compat:** layout matches PagedAttention. Migration path: add `block_table` per row, swap `_splice_in`/`_batched_kv` reconstruction for block-table tracking, plug in paged-attention kernel. Pool/BlockManager/RadixTree/eviction/CacheManager API all stay.
**Pool-less mode preserved** for tests that don't pass `layer_shapes` (store becomes no-op).

### [2026-05-29] Modal image pins CUDA torch wheel explicitly
*tags: `benchmark`, `modal`* · `kb-20260529-041`

`modal_app.py` installs `torch>=2.4` from `https://download.pytorch.org/whl/cu121` *before* `pip_install_from_pyproject`. Reason: default PyPI torch wheel can resolve to the CPU variant on Linux x86_64; our `_detect_device()` would silently fall through CUDA → MPS → CPU and we'd benchmark CPU on a GPU host without noticing. Explicit pin = guaranteed GPU torch + loud failure if the base image's CUDA version drifts from cu121. Trade-off: if Modal moves to a non-CUDA-12 base, we bump the URL.

### [2026-06-12] Sliding-window attention
*tags: `cache`, `decode`, `kernel`, `kv`, `memory`, `modal`, `numerics`, `prefill`, `scheduler`* · `kb-20260612-037`

Gemma 4 is 28/35 `sliding_attention` (window=512); the forward stored `sliding_window` but never applied it — all layers attended to full history. The 5-token parity fixture never caught it (5 ≪ 512), so long-context outputs were silently wrong. **Fix (all 3 attention paths, keyed on the layer's `sliding_window`):** prefill SDPA builds an explicit position-based causal(+window) mask; batched-decode gather adds a rightmost-W column cutoff; the Triton kernel takes a `window` arg (skips fully-out blocks via `start_b`, masks `offs >= L - window`). **Conditional design preserves parity:** keep the exact `is_causal` path when the window doesn't bite (`S_k ≤ window`) and there's no cache offset → all byte-identical (`torch.equal`) parity tests still pass. **Bonus:** the position-based mask also fixed a latent causality bug — prefix-cached prefill with a multi-token suffix was running NON-causal (old `is_causal = S_q==S_k` went `False` with a cache offset; only ever tested with a 1-token suffix). **Validation:** `tests/test_sliding_window.py` (CPU, 600-token prompt vs HF — argmax + logits within bf16 tol); kernel windowed-reference parity on Modal (D=256/512, seq_lens spanning the window). **Follow-up DONE:** windowed KV *storage* — sliding layers now free out-of-window blocks (see the windowed-KV-storage entry above). What remains is window-aware admission to turn the freed memory into higher concurrency.

### [2026-06-12] Windowed KV storage (free out-of-window blocks on sliding layers)
*tags: `cache`, `decode`, `kernel`, `kv`, `memory`, `modal`, `prefill`, `scheduler`* · `kb-20260612-036`

Follows the sliding-window-attention entry below. Sliding layers (28/35) only need the last 512 tokens of KV, but were storing the full history. **Decision — sentinel eviction, no kernel change:** `BlockPool.window` carries the per-layer window (set from `attn.sliding_window` in `make_pools_for_gemma`); `PagedKVCache.append` (and `BatchedPagedKVCache.append`) call `_evict()` after each write — once a block is fully out of window it's `pool.release()`d and its block-table slot set to `-1`. The decode kernel's existing `start_b = (L-window)//block_size` already skips exactly those slots (`start_b == evicted count`), so **the kernel is unchanged**; `free_all`/`get`/`block_table_tensor` treat `-1` as "skip / map to 0" (out-of-window positions are masked anyway). **Prefix-cache interaction:** only cache prefixes ≤ window — if a sliding layer evicted (prompt > 512), `prefill` skips `store()` (`cache.any_evicted`). This avoids offset-tracking in the PrefixCache; long-prefix sharing (rare, previously on wrong outputs) is dropped. **Validation:** unit test (blocks freed, in-window tail readable, clean `free_all`); 540-token windowed-decode-vs-HF (CPU); long-workload Modal sweep 0 errors, no perf regression. **Follow-up DONE:** window-aware per-pool admission now turns the freed memory into concurrency (see entry above). **Still OPEN (minor):** block-table list/tensor grows with context (sentinels kept, no offset trim) — tiny int32 data, a long-context micro-opt.

### [2026-06-12] Window-aware per-pool KV admission
*tags: `decode`, `kernel`, `kv`, `memory`, `modal`, `prefill`, `scheduler`* · `kb-20260612-035`

Turns the windowed-storage memory savings into actual long-context concurrency. **Insight:** with all pools equal-sized, the FULL-attention pools bind (they grow with sequence; sliding pools cap at the window), so windowing alone doesn't raise admission — and naively shrinking the reservation for sliding layers would under-protect the full pools → OOM. **Two changes together:** (1) **right-size pools** — `make_pools_for_gemma(sliding_blocks=...)` + `CUSTOM_BACKEND_SLIDING_BLOCKS`; sliding pools shrink (capped at window), freeing memory to enlarge the full pools. (2) **per-pool window-aware gate** — `CustomTorchBackend.kv_reserve/kv_release` (base class no-op so `TorchBackend` is unaffected) reserve per-request *per-pool* block footprints (sliding capped at `window//bs + 1`) and admit only if EVERY pool fits; wired into `_admit_pending` (soft-hold on False) with `kv_release` on every exit (evict / prefill-fail / chunk-fail / `_fail_all`). The scalar `MAX_ACTIVE_KV_TOKENS` stays as a coarse cap. **Demonstrated (Modal, A10G):** long (1000-token) workload to N=32 with 0 rejections at ~equal memory (full=2048 + sliding=1200 ≈ 422 MB vs baseline 432 MB), where the baseline's 22000 token cap blocked ~1/3 at N=32 → ~1.45× concurrency. Win scales with context/window ratio (≈1/16 at 8k context). Tests: `tests/test_kv_admission.py` (footprint cap, full-pool binding, reject doesn't mutate state). **Note:** footprint is reservation-based (worst case `prompt + max_tokens`), not live usage — conservative, prevents decode OOM.

### [2026-06-12] Scheduler mask width must not assume `== kv_length`
*tags: `backpressure`, `cache`, `kernel`, `kv`, `numerics`, `scheduler`* · `kb-20260612-034`

`_splice_in` sized the attention-mask `torch.cat` off `backend.kv_length(batched_kv)`. For `TorchBackend` (DynamicCache) that equals the mask width forever. But the **custom backend's `kv_length = max(per-row seq_len)` DROPS when the longest row is evicted**, while the mask width doesn't shrink → a later admit cat'd mismatched widths → `RuntimeError` → worker died (`_fail_all`) → all later requests timed out. This is exactly why **short (uniform lengths) survived but mixed (a long row finishing among short ones) crashed**, and why no single-process repro reproduced it (only the server's evict-longest-then-admit pattern triggers it). **Fix:** drive mask bookkeeping off the mask's own width (`self._attention_mask.shape[1]`), not `kv_length` — equivalent for `TorchBackend`, correct for paged. Regression test `tests/test_scheduler_splice.py` (CPU, fake paged backend; confirmed to hang pre-fix). **Lasting rule:** scheduler-owned tensor state must be self-consistent, not derived from a backend metric that has different invariants per backend.

### [2026-06-12] De-Python the decode step (vectorized scatter + one-shot block table)
*tags: `cache`, `decode`, `graph`, `kernel`, `kv`* · `kb-20260612-032`

Follows [[m2-3-paged-attention-kernel]]. The kernel removed the gather, but TPOT still sloped (N=32 173 ms vs N=1 50 ms): every decode step ran a Python per-row loop doing per-row tensor writes for the scatter-append (~N×non-shared-layers tiny GPU launches/step) and rebuilt the per-layer block table from Python lists. **Fix:** `BatchedPagedKVCache.append` now does ONE vectorized indexed write (`pool.k[block_ids, :, slots, :] = k_new`) for the whole batch (block-boundary alloc stays in Python — rare, no tensor writes); `block_table_tensor` builds with a single host→device copy. **Result:** N=32 TPOT 173→99 ms, decode throughput 185→323 tok/s (short workload). Net vs row-by-row at N=32: ~20×. Remaining slope (N=1 47 → N=32 99 ms) is partly real compute (A10G nearing compute-bound) + per-layer kernel launches — CUDA graphs is the lever for the rest.

### [2026-06-12] CUDA-graph decode: single max-batch graph, not bucketing
*tags: `compile`, `decode`, `graph`, `kernel`, `kv`, `memory`, `modal`* · `kb-20260612-031`

Profiling showed decode was ~95% CPU-bound (~50 ms/step, ~1000 tiny kernel launches across 35 unfused layers; GPU compute a few ms) and **flat across batch size** (49 ms @ N=8 → 57 ms @ N=32). So we're dispatch-bound, not compute-bound. **Decision:** capture ONE CUDA graph at `max_batch_size` and pad smaller batches up to it — NOT vLLM-style per-bucket graphs. Bucketing exists to avoid max-batch *compute* at low concurrency, but since our GPU compute is nearly free (flat curve), padding N=1→32 wastes ~2 ms while still removing ~50 ms of dispatch; one graph is simpler, lighter on memory, no bucket-selection. **Implementation:** persistent `BatchedDecodeState` (block tables/seq_lens as GPU tensors — prerequisite, the old `torch.tensor(python_list)` rebuild was uncapturable) → `_GraphCtx` over FIXED static buffers (tokens/positions/seq_lens/block_tables, block tables pre-sized to `context_window/bs`) → warmup (JITs Triton kernel, stabilizes allocator) → `torch.cuda.graph` capture → per-step: copy inputs into static buffers, `graph.replay()`, read `logits[:n]`. Inactive padding rows point at a scratch block. `prepare_step()` (alloc) + `advance()` (evict) run on the state *outside* the graph. Falls back to eager on capture failure (`CUSTOM_BACKEND_CUDA_GRAPH=0` to disable). **Result:** decode ~50→25 ms/step (~2×), byte-identical to eager (`scripts/gpu_tests/test_paged_kernel_integration_modal.py`: GRAPH vs EAGER OK). **Why not the full ~10×:** the graph removes CPU *launch* overhead, but the GPU still runs ~1000 unfused tiny kernels/step. Closing the rest needs **op fusion** (torch.compile of the custom forward, or hand-fusing the per-layer norm/proj/gate ops) — a separate effort. Earlier `torch.compile` trouble was with the HF model (DECISIONS 2026-05-30); our custom forward is untried under compile. Also still open: `prepare_step`/`advance` keep ~1–2 GPU→CPU syncs/step (alloc/evict bookkeeping) which prevent full CPU/GPU overlap.

### [2026-06-11] M2.3 — Triton paged-attention decode kernel
*tags: `compile`, `decode`, `graph`, `kernel`, `kv`, `modal`, `numerics`, `prefill`* · `kb-20260611-030`

<a name="m2-3-paged-attention-kernel"></a>Context: the batched gather+SDPA decode (above) regressed at N=32 because every step rebuilt a dense `[N,H,Lmax,D]` tensor with a Python per-row pad/cat loop + padded SDPA on short rows. **Decision:** write our own Triton kernel (`models/paged_attention_kernel.py::paged_decode_attention`) reading K/V directly from the scattered blocks via the block table — no gather, no padding, FlashAttention-style online softmax. **Triton, not raw CUDA C++:** Triton compiles to PTX/GPU (it *is* a CUDA kernel), ships with torch (no toolchain on the Modal image), ~90 lines vs many hundreds, and is what the serving stack (vLLM Triton kernels, FlashInfer) uses for exactly this. **Why our own and not vLLM's/FlashInfer's prebuilt kernel:** Gemma 4's full-attention layers have `head_dim=512`, the same ceiling that ruled out FA2 (see [2026-05-30 FA2]); prebuilt kernels cap at 256. A hand-written Triton kernel tiles `D` via a constexpr → handles 256 (sliding) and 512 (full) in one kernel. Also GQA (Hq query heads share num_kv_heads), scale=1.0 (norms absorb scaling). Wired behind a `paged_ctx` kwarg threaded `forward → layer → attention`; only the attention K/V-assembly+SDPA core is replaced, every norm/MLP/RoPE/projection path untouched. CPU/MPS keep the gather+SDPA path as portable reference (`decode_step_batched` branches on `device.type == "cuda"`). **Validation (GPU, Modal):** isolated kernel vs torch reference max|diff| 5e-4 across D∈{256,512} & GQA; single-step identical-input argmax agrees with gather on every row (full-model logit diff ~0.3 ≈ 1% of the ±30 softcap, from online-softmax vs SDPA reduction-order over 35 layers); full-sequence kernel output == gather. **Result:** N=32 TPOT 385→173 ms, decode throughput 83→185 tok/s and monotonic in batch (knee gone). Net vs original row-by-row: ~12× at N=32. **Parity caveat:** greedy tokens are NOT byte-identical to the SDPA/HF path — ~1% logit noise flips near-ties (same reason vLLM ≠ HF bit-exact); validation is by tolerance + argmax-agreement, not `torch.equal`. **Still open:** the slope isn't flat (N=32 TPOT 173 ms vs N=1 50 ms) — residual per-step Python (`block_table_tensor` built per-layer) + per-layer kernel-launch overhead across 35 layers; **CUDA graphs** is the next lever. Prefill still uses per-row SDPA (kernel is decode-only).

### [2026-06-11] Custom backend decode is row-by-row, not GPU-batched (M2.4)
*tags: `cache`, `decode`, `kernel`, `kv`, `numerics`, `scheduler`* · `kb-20260611-029`

Context: M2.4 made `CustomTorchBackend` scheduler-drivable. The batched KV is a plain `list[PagedKVCache]` (one per active row); `decode_step_batched` looped `self.model(...)` once per row — correct, but ~B sequential forwards, no GPU-batching win. CUDA load-test confirmed it: TPOT scaled perfectly linearly (64 ms × N) and aggregate decode throughput was pinned flat at ~15.7 tok/s regardless of batch size. **Resolution:** real batched decode — one forward over all rows. Two stages, both landed: (1) **batched gather+SDPA** — `BatchedPagedKVCache` left-pad-gathers all rows to `[N,H,Lmax,D]`, one masked SDPA; bf16-exact vs row-by-row in test. Throughput then scaled with batch (15→114 tok/s @ N=16, ~7×) but regressed at N=32 (114→83) from the Python per-row pad/cat loop. (2) **[[m2-3-paged-attention-kernel]]** removed that knee. See that entry. The scheduler's `attention_mask` is still ignored by the custom backend (we build our own).

### [2026-05-14] Chunked Prefill — Version A (alternating)
*tags: `benchmark`, `cache`, `decode`, `graph`, `kv`, `prefill`, `scheduler`* · `kb-20260514-022`

Two forward passes per iter: one batched decode + one prefill chunk for an admitting request. MPS-feasible. Captures HOL-blocking fix (long prompt no longer freezes active users' decode). `_prefilling` queue; rows promote to `_active` on final chunk. Backend `prefill_chunk(chunk, partial_kv) -> extended_kv`. Cache lookup at admit, store after final chunk. **Shipped + tested** (`tests/test_chunked_prefill.py`, 8 tests).
**Made disaggregation-ready (2026-06-13).** Two forward-compat seams so P/D disaggregation slots in without a scheduler rewrite:
- **`PREFILL_MODE` config** (`monolithic`|`chunked`; `mixed_batch`/`disaggregated` are future values of the same enum). Derived from `PREFILL_CHUNK_SIZE` when unset (back-compat); unimplemented modes raise at construction.
- **`_promote_to_decode(request, kv, kv_len, first_token)`** — the single point where a completed prefill's KV joins the decode batch (both monolithic admit + chunked final-chunk funnel through it). Today a local splice; under P/D disagg this is where the KV **transfer** from prefill-worker→decode-worker plugs in. The loop is already prefill/decode-decoupled (`_prefilling` phase + `prefill_chunk` primitive), so disagg = a transfer at this seam + a remote prefill loop, not a rewrite.
**P/D disaggregation itself stays deferred** (KV serialization/transfer over NCCL/RDMA, separate worker processes, control plane — real work, premature now). It attaches at the two seams above. Trigger: after the A100/H100 move + vLLM head-to-head, if prefill/decode interference is the measured tail-latency culprit. V-B (mixed-batch) is the intermediate step; see entry below.

### [2026-06-13] Prometheus metrics: aggregate, NOT session_id-labeled
*tags: `memory`, `scheduler`* · `kb-20260613-016`

Context: CLAUDE.md's architecture table originally said "Prometheus metrics — all labeled by `session_id`." But `session_id` is unbounded in production, and a per-session label makes one time series per session → Prometheus memory/cardinality blowup (a well-known anti-pattern).
**Decision (with user):** `/metrics` exposes **aggregate** metrics with only low-cardinality labels (outcome; HTTP method/endpoint/status). **Per-session detail stays in `/scheduler/stats`** (JSON, pull-on-demand, not stored as time series). Honors the spirit (per-session observability exists) without the cardinality footgun. Updated the CLAUDE.md line to match. If bounded per-session Prometheus metrics are ever needed (e.g. the `sim-*` simulator), add an opt-in low-cardinality label like a priority/tenant *class*, never raw session_id.

### [2026-09-01] Modal runs from a git worktree ship the WRONG source
*tags: `graph`, `modal`* · `kb-20260901-012`

`add_local_python_source("inference_server")` resolves the package via the local import path.
The venv has an editable install pointing at the MAIN checkout, so a `modal run` launched from
a worktree silently shipped main's code and the container ran the unchanged backend. Always
`PYTHONPATH=$PWD/src venv/bin/modal run ...` from a worktree (same for pytest). Cost one wasted
A100 run that failed with `AttributeError: no attribute '_graph_buckets'`.

### [2026-09-01] Decode output is not batch-invariant (bucketed CUDA graphs)
*tags: `benchmark`, `decode`, `graph`, `kernel`, `modal`, `numerics`* · `kb-20260901-011`

**Context.** Bucketing decode graphs by row count means the padded batch width now varies with
concurrency instead of being pinned at `max_batch_size`. A `[B, 1, H] @ [H, H]` GEMM picks a
different cuBLAS kernel (different accumulation order) per `B`, so logits shift by ~0.5 and a
near-tie argmax can flip. Consequence: **the same prompt can produce different tokens depending
on how many other requests are in flight.**

**This is bucketing, not our graphs.** `scripts/probes/diag_pad_numerics_modal.py` reproduces it EAGER
with no CUDA graph anywhere, purely by changing the pad width: the same 17 rows give one answer
at widths 17/128/256 and another at 32/64. Meanwhile graph replay is *bit-exact* against
width-matched eager (`maxdiff=0.000000` at every bucket, `bench_decode_buckets_modal.py`).

**Decision:** accept it. vLLM has the same property, the win is large (1.80x ms/step at n=1,
1.47x at n=32), and output is still deterministic at a fixed batch width. Do NOT gate parity
tests on "graphed tokens == unpadded eager tokens" — that test fails for legitimate numerical
reasons and will send you chasing a bug that is not there (it did, for an hour). Gate on
width-matched parity instead.

**Triggers to revisit:** (a) a user-visible reproducibility requirement, (b) an eval whose
variance is dominated by this, (c) a `deterministic=true` serving mode — that needs
batch-invariant kernels (fixed split-K reductions), a real project.

**Revisit when:** (a) a user-visible reproducibility requirement, (b) an eval whose

### [2026-09-01] Persistent Inductor cache
*tags: `cache`, `compile`, `decode`, `graph`, `kv`, `modal`, `prefill`* · `kb-20260901-010`

Bucketed decode graphs turned one Inductor compile into one per bucket. `dynamic=False`:
~92s per bucket, uniform (553s for 6 buckets, ~828s for the 9-bucket production ladder).
`dynamic=None` (now default) collapses that to 2 compiles + ~1.5s per remaining bucket, but
those 2 compiles are expensive and wildly variable (138+231s one run, 264+503s another).
`dynamic=True` is unusable: Inductor codegen `NameError: name 's6' is not defined` (same class
as the `s46` bug HANDOFF hit on prefill). With compile OFF the whole ladder captures in 5.6s,
so 100% of this cost is torch.compile. **Caching Inductor artifacts to a Modal volume is now
the highest-value startup fix, and it also unblocks any 2D graph ladder (see V-B above).**

### [2026-09-02] KV pressure is a first-class operating mode, not an error
*tags: `backpressure`, `benchmark`, `cache`, `kv`, `memory`, `modal`, `numerics`, `prefill`, `scheduler`* · `kb-20260902-003`

Under sustained distinct-prompt traffic the engine used to break rather than degrade, and no
existing benchmark could see it (all of them run below saturation against 64 reused prompts and
a never-evicting cache). `scripts/bench/bench_stress_modal.py` now drives cache-missing traffic past
the knee into a small pool. Five defects it exposed, all fixed:
1. PrefixCache never evicted — every distinct prompt leaked blocks permanently until alloc()
   raised. Now LRU with an entry cap AND a share-of-pool watermark (default 50%), plus
   reclaim-on-demand from BlockPool.
2. reclaim() counted release() calls rather than blocks that actually became free, so it
   reported success having freed nothing.
3. Partial prefill allocations leaked when alloc() raised mid-loop — the dominant bug: the pool
   drained to zero and stayed there, serving 0 requests at rate 48.
4. Any scheduler-iteration exception killed the worker thread AND leaked every active row's
   blocks. Now the loop survives, releases what it drops, and preempts the newest row under
   pressure instead of losing the batch.
5. Overload became unbounded latency (queue 1001 deep, p95 105s) instead of rejection. Added an
   admission deadline (`max_queue_wait_s`, default 30s).

**The invariant to preserve:** running out of KV blocks is an EXPECTED operating condition. It
must surface as `KVCacheExhausted` → a 429-style rejection, never as an engine failure, and no
path may abandon blocks it has already allocated.

## Rejected (9)

**Tried, measured, does not work.** Do not retry blind — these are the entries that stop the loop re-treading dead ends.

### [2026-09-05] Tiled prefill attention — correct, 10.84x isolated, NO end-to-end benefit (default OFF)
*tags: `attention`, `kernel`, `prefill`, `resolved-noise`, `triton`* · `kb-20260905-481ec50a`

FlashAttention-style tiling of `_paged_prefill_kernel` (one program per BLOCK_M queries, each K/V block loaded once instead of once per query). Written, correct, **not merged** — see below.

**Correctness: 9/9** vs the untiled kernel, incl. ragged tails, prefix hits, sliding windows, head_dim 512 and a ragged batch (rel <= 3.8e-3).

**Isolated speed (A100):**

| tokens | D | untiled | tiled | speedup |
|---|---|---|---|---|
| 240 | 256 | 0.392 | 0.470 | 0.83x |
| 512 | 256 | 1.545 | 1.222 | 1.26x |
| 824 | 256 | 3.154 | 2.004 | **1.57x** |
| 240 | 512 | 0.328 | 0.686 | 0.48x |
| 512 | 512 | 1.288 | 1.978 | 0.65x |
| 824 | 512 | 2.618 | 4.967 | **0.53x** |

The win needs enough queries to amortise the K/V load AND a head_dim small enough that the [BLOCK_M, D] accumulator stays in registers. At D=512 the tile must shrink to 16, halving reuse and spilling — consistently worse. Hence routing (D<=256 and S>=512), not blanket enabling.

**Why not merged: no trustworthy end-to-end evidence.** Three A/B attempts, three different confounds (see the loop entry). The only clean run showed prefill p95 *worse* (259 -> 457 ms), and that run was itself invalid because CUDA-graph capture defeats the toggle. Isolated kernel wins do not entitle a merge.

Branch `perf/tiled-prefill-attention` holds the work.

**RESOLVED (iteration 4).** A clean A/B — prefill graph OFF so the toggle actually takes effect, arms interleaved in one container, cold cache per arm, TPOT declared as a sanity metric (held at 84.9 vs 85.9 ms) — gives:

    prefill p50  80.1 -> 82.5 ms  (+3%)
    prefill p95 214.2 -> 194.0 ms  (-9.4%)

**Verdict: NOISE** (|t|=1.40 < 2.0). Predicted >15%; delivered an effect that does not clear the variance budget. **Not merged.**

The reason is the routing, and it points at the real lever: tiling only fires for D<=256 and S>=512, i.e. SLIDING layers on LONG prompts. But sliding layers cap attention at window=512 keys, so they are not where the attention work is — the FULL-attention layers (head_dim 512, unbounded context) do far more work on a long prompt, and those are exactly the ones where tiling currently LOSES 0.53x to register spill. The kernel is tiled where it matters least.

**MERGED (iteration 5).** The iteration-4 'noise' verdict was correct for the config I had hand-picked, and the config was wrong. A launch sweep found BLOCK_M=16/warps=4 gives **10.84x** isolated at D=256/824 where BLOCK_M=32 gave 1.81x — a 6x difference from one constant. Re-running the same A/B with the swept config: prefill p95 **-31.9%** (|t|=2.96), overall TTFT p95 615 -> 325ms, sanity held (TPOT 138.1 vs 133.1). All five gates green; merged.

**Lesson: sweep launch configs before concluding a kernel does not help.** Bigger tiles are not better — past a point the [BLOCK_M, D] accumulator costs more in occupancy than the reuse is worth.

**RETRACTION (iteration 7).** The -31.9% that authorised this merge is not supported. A null experiment — three IDENTICAL runs, same container, same config, nothing changed — gives `ttft_prefill_p95` = 436.6 / 289.9 / 136.9 ms, a **3.2x null spread**. A 31.9% effect is far inside that. Worse, the values fall monotonically, so it is warmup: the treatment arm always runs second and is systematically favoured.

**What still stands:** the kernel is correct (9/9) and **10.84x faster in isolation** at D=256/824 — a tight 20-rep microbenchmark, which is a far more reliable measurement than an end-to-end p95. Keeping the code on those grounds. **What does not stand:** any claim about end-to-end TTFT.

**SETTLED (iteration 8, properly replicated).** 4 runs per arm, ABBA order, first of each arm discarded as warmup, variance across runs:

    ttft_prefill_p95   +24.3%  SIGNIFICANT, against the prediction (|t|=2.27)
    ttft_prefill_p50    +4.0%  noise
    ttft_p95            +1.0%  noise
    tpot_p50            +2.7%  noise

Baseline after warmup: 73.2 / 88.5 / 75.0 ms. Treatment: 111.3 / 95.0 / 87.9 ms.

**Default flipped to OFF.** A 10.84x isolated kernel win that does not reach the service level is not a win, and the best-powered estimate says it slightly hurts.

**Open:** this A/B necessarily ran with the prefill CUDA graph OFF, since a flag consumed inside a captured graph cannot be toggled at runtime. With the graph ON dispatch is removed and attention is 63% of prefill, so tiling might matter there. Testing that needs capture-time variants, not a runtime flag.

**Revisit when:** an A/B that toggles tiling at CAPTURE time, not after, shows an end-to-end win; the D=512 register spill is solved (e.g. splitting D)

**Evidence:** scripts/gpu_tests/test_tiled_prefill_modal.py, iter3-tiled-prefill / iter3-ab / iter3-ab2 / iter3-ab3

### [2026-09-05] Prefill bucket padding costs grid slots, not compute
*tags: `prefill`, `graph`, `kernel`* · `kb-20260905-33842618`

Hypothesis: add a bucket near 896 so an ~824-token suffix stops rounding up into the 1024 bucket (19.6% padding).

**Falsified at tier 1 for $0.** The prefill kernel exits padding queries immediately (`if qj >= suffix_len: return`), so 200 padded positions x 8 heads = 1603 grid slots that do no work — padding is a launch cost, not a share of the compute. Refining the ladder cannot move prefill p95.

Note this does NOT contradict the earlier ladder refinement, which improved prefill p50 by 10-20%: that came from smaller buckets doing less *real* work on short prompts, not from removing padding waste.

**Revisit when:** the kernel stops early-exiting padding queries

**Evidence:** scripts/probes/probe_prefill_ceiling.py

### [2026-06-12] Triton constexpr on a data-varying value → recompile storm
*tags: `compile`, `kernel`, `kv`, `triton`* · `kb-20260612-033`

The paged kernel originally took `MAX_BLOCKS` (= block_tables width) as a `tl.constexpr` loop bound. Triton recompiles per distinct constexpr value, so under the mixed/long sweep (sequence lengths grow and vary widely) it recompiled for nearly every block-count — ~1 s each, holding the backend lock — which looked like a hang (worker stalls, requests pile to `pending` and time out, no traceback). Fixed by making the loop bound a **runtime** per-sequence `n_blocks` derived from `seq_len` (compiles once; also optimal — no padding-block iterations). Confirmed on GPU: block-counts 2→128 all 0.1–0.5 ms, no spikes. **Lasting rule:** never put a value that varies with input shape/length in a Triton `constexpr` — only genuinely-fixed things (head_dim, block_size).

### [2026-05-30] `torch.compile` + Gemma 4 forward
*tags: `benchmark`, `cache`, `compile`, `decode`, `graph`, `kernel`, `kv`, `memory`, `numerics`, `prefill`, `quantization`, `torch-compile`* · `kb-20260530-014`

Context: the original crash was in **HF's** `modeling_gemma4.py:2505` — `logits = self.lm_head(hidden_states[:, slice_indices, :])` with `slice_indices=None` under dynamo. That blocker is specific to HF's forward; it does **not** apply to our custom `models/gemma4.py`, whose `lm_head(h)` is unconditional.
**Resolution (2026-06-13):** the forward IS compilable, but op fusion is **not our decode lever** — measured and reverted. Two findings:
1. **Traceability (kept).** Probed with `torch._dynamo.explain` (`scripts/probes/probe_compile.py`, CPU — graph breaks are device-independent): square prefill and paged prefill trace with **0 graph breaks** (op_count ~1.9–2.3k). The *only* Dynamo-hostile code is the **cache/scatter bookkeeping** (`BatchedPagedKVCache.get/append`, `paged_ctx` — `if seq_len==0`, `max(...,default=)`, `L//bs>=len(bt)`), never the math. Clean seam: compile the math, leave bookkeeping eager. Numerics safe (RMSNorm byte-identical, prefill argmax-identical).
2. **Fusion is a no-op under the CUDA graph (the reason for reverting).** A10G, decode ms/step: eager 65.4 → graph 19.98 → compile(leaf, no graph) 54.3 → compile+graph **19.86** = **1.01×**, token-identical. The HANDOFF hypothesis ("residual ~25 ms = ~1000 unfused tiny kernels") is **falsified**: fusing them changed nothing under the graph. The graph step costs ~20 ms whether 4 or 32 rows are real → the floor is the **matmuls reading all weights from HBM each step** (small-batch decode is memory-bandwidth-bound), not launch/exec of tiny kernels. So compile (and the q/k/v-pack hand-fusion idea — same weight bytes) can't move it. Knob reverted (no-op + cold-start codegen tax); finding kept.
**Forward use:** because the forward is compile-clean, compile becomes load-bearing for a **quantized** forward (Inductor fuses dequant into the GEMV epilogue) — that's the actual bandwidth lever. Next: quantization (see CLAUDE.md). Also re-measure decode at full batch (32 real rows: 19.86/32 ≈ 0.62 ms/tok ≈ 1600 tok/s) — the current bench under-loads the max-batch graph.

### [2026-05-30] FlashAttention-2 ruled out on Gemma 4 + A10G
*tags: `backpressure`, `compile`, `flash-attention`, `kernel`, `modal`* · `kb-20260530-017`

Context: tried `attn_implementation="flash_attention_2"` in the Modal image. Lifespan crashed with `RuntimeError: FlashAttention forward only supports head dimension at most 256`. Gemma 4 E2B's head_dim exceeds 256; FA2's kernel ceiling on Ampere is hard. **Resolution:** stay on HF's default `sdpa` (which itself dispatches to FA2 when shapes allow, cuDNN otherwise). FA3 on H100 would lift the head_dim cap to 512 — revisit if/when we move off A10G. `torch.compile` kept (independent optimization).

### [2026-09-02] Decode-graph capture order does not fix the compile blow-up
*tags: `benchmark`, `cache`, `capture-order`, `compile`, `decode`, `graph`* · `kb-20260902-008`

Automatic-dynamic compile only generalizes over LARGE batch dims; Dynamo re-specializes on small
ones regardless of capture order. At `MAX_BATCH_SIZE=256` (8 buckets):
descending 1287.7s (256:155s, 128:262s, 64/32/16: ~1.8s each, then 8:295s, 4:283s, 2:288s);
ascending 1347.7s (2:212s, 4:386s, 8:368s, 16:368s, then 32/64/128/256: ~3.3s each).
Either way ~4 expensive compiles and ~21 minutes of startup. Order is not the lever; the number of
SMALL buckets is. Options, unresolved: a coarse ladder (e.g. 8/32/128/256, <=4x padding), dropping
torch.compile entirely (whole ladder captures in 5.6s without it — is compile still worth 1.47x now
that graphs, the de-Pythoned step and split-K have landed? unmeasured), or the persistent Inductor
cache. Kept descending: it front-loads the two cheap-to-generalize large shapes.

### [2026-09-02] Length-grouped prefill waves — BUILT, DEFAULT OFF, inert below saturation
*tags: `prefill`, `scheduler`, `wave-planning`* · `kb-20260902-006`

83-91% of prefill waves are K=1 at rates 1-6 (`wave_sizes` in scheduler stats), because
MAX_BATCH_SIZE=256 admits every arrival immediately and the queue never backs up. At K=1 there is
no padding, so grouping has nothing to do: A/B was noise at and below the knee, +6.8% only at
rate 6. **This also removes the varlen/framework option from the near-term table** — varlen and
length-grouping both attack padding waste, and there is none at K=1. Kept behind
`WAVE_WINDOW_MULT` (default 0); turn it on when the engine is actually driven at its stated
100-256 concurrency target, where waves are wide.

### [2026-09-01] Mixed-batch chunked prefill (V-B)
*tags: `benchmark`, `cache`, `compile`, `decode`, `graph`, `kernel`, `kv`, `modal`, `prefill`, `quantization`* · `kb-20260901-009`

**Context.** V-B packs decode tokens and a prefill chunk into one forward so a prefill never
stalls in-flight decodes. Correct for vLLM, and the earlier blocker (DECISIONS 2026-05-14:
"needs FlashAttention-varlen or a custom per-layer attention path") is now GONE — our
`paged_prefill_attention` is that path, and a decode row is just a prefill row with
`suffix_len=1`, so the ctx shape already supports a mixed batch.

**But it fails on a precondition we don't meet.** A mixed step has a variable `(B, S)` shape,
so it cannot replay the decode CUDA graph — it must run eager. Our eager forward is
dispatch-bound and nearly flat in size. Measured A100/E4B
(`scripts/probes/probe_mixed_step_premise_modal.py`):

| rows | graphed step | eager step | tax |
|---|---|---|---|
| 1 | 18.9 ms | 100.3 ms | 5.3x |
| 8 | 20.6 ms | 75.6 ms | 3.7x |
| 32 | 24.6 ms | 79.5 ms | 3.2x |

Chunking a 240-token prompt across 4 mixed steps = **302 ms** of step time. Today, with the
prefill CUDA graph landed, the same prompt is **one 43 ms stall** and every other step stays
graphed at 20.6 ms. **V-B as scoped is a large net regression here.**

**Decision:** don't build it. Attack prefill-vs-decode interference by keeping everything
graphed instead — extend the prefill graph to K>1 and prefix-hit cases (currently K=1,
matched=0, bucket<=512 only). Revisit V-B only if we get a cheap variable-shape forward.

**Triggers to revisit:** (a) piecewise/bucketed graphs over a 2D `(B, S)` ladder — blocked on
compile cost, see the Inductor-cache entry; (b) an eager path that isn't dispatch-bound;
(c) workloads with prompts long enough that one prefill can't fit in a single graphed call.
Note the 2026-06-11 entry already preferred P/D disaggregation over V-B; this is the
quantitative reason.

**Revisit when:** (a) piecewise/bucketed graphs over a 2D `(B, S)` ladder — blocked on

### [2026-09-03] Graph capture ORDER does not fix the compile cost
*tags: `benchmark`, `cache`, `compile`, `decode`, `graph`* · `kb-20260903-000`

Hypothesis: capturing decode graphs largest-first introduces the symbolic batch dim at a large
value, and its guards fail to cover the small end — measured at MAX_BATCH_SIZE=256, buckets
256/128 compiled, 64/32/16 then reused the dynamic artifact in ~1.8s each, and 8, 4 and 2 EACH
re-specialized for ~290s (1288s total). Capturing ascending should introduce the dynamic dim at
the small end where guards generalize upward.

**It does not.** Ascending measured **1348s** vs descending 1288s at an 8-bucket ladder — no
better, arguably worse. Dynamo re-specializes on small batch dims regardless of the order they
are seen in. The one-line change (`sorted(self._graph_buckets)` instead of `reverse=True` in
`_capture_all_graphs`) was never merged and its worktree is now deleted; this entry is the
record. **Do not re-try capture ordering as a compile-cost lever** — the levers that actually
move it are the number of distinct compiles (~140s each, linear) and, weakly, the persistent
Inductor cache (-19%).
