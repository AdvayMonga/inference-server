# Fast Inference from Transformers via Speculative Decoding

**Link:** https://arxiv.org/abs/2211.17192
**Venue:** ICML 2023
**Authors:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research)
**Status:** not read
**Relevant phase:** Phase 6 / Optional Extensions (advanced optimization)

## Core idea

Autoregressive decoding generates one token per forward pass through the large target model, and each forward pass is bottlenecked by memory bandwidth, not by FLOPs — the GPU is mostly idle waiting on weight reads. Speculative decoding exploits the slack: a small, cheap **draft model** generates `K` candidate tokens in `K` cheap forward passes, and the large target model then verifies all `K` candidates in **one** forward pass (which is nearly the same cost as generating one token, because the bottleneck is per-call HBM traffic, not per-token compute). A rejection-sampling check guarantees the output distribution is **identical** to plain target-model sampling — no quality loss. When the draft model is right most of the time, every target forward pass yields multiple accepted tokens, multiplying decoding throughput by 2-3× without changing the output.

## Why it matters

- One of the cleanest "free lunches" in inference: speedup with provably no change in output distribution. No quality/throughput tradeoff to argue about.
- Plays well with everything else in the engine — it does not change the KV cache, the scheduler, or the attention kernel. It changes the decode loop.
- Practical wins depend on the draft-target acceptance rate, which depends on model pairing. With Gemma 4 E4B as target, a smaller distilled Gemma or a small generic draft model would be the natural pairing.
- The technique generalizes (Medusa, EAGLE, lookahead decoding, n-gram drafters). Implementing the basic version first creates the seam those variants slot into.

## Key mechanism

1. **Two models, one tokenizer.** A large **target** model `Mq` (the one whose distribution is desired) and a small fast **draft** model `Mp`. Both must share a tokenizer so token IDs are interchangeable.
2. **Draft phase.** Starting from the current prefix, the draft model autoregressively samples `K` candidate tokens `x_1 ... x_K` along with their probabilities `p(x_i | prefix, x_1 ... x_{i-1})`. This costs `K` cheap forward passes through `Mp`.
3. **Target verification phase.** The target model is called **once** on the prefix plus all `K` draft tokens, producing target probabilities `q(x_i | prefix, x_1 ... x_{i-1})` for every position in parallel.
4. **Rejection sampling per token.** For each `i` from 1 to `K`: accept `x_i` with probability `min(1, q(x_i) / p(x_i))`. If accepted, move on. If rejected at position `j`, throw away `x_j ... x_K`, sample a replacement token from the **adjusted distribution** `max(0, q - p)` normalized, and stop. The math guarantees the resulting token sequence has exactly the same distribution as sampling from `q` directly.
5. **Bonus token.** If all `K` drafts are accepted, the target model's parallel forward pass already produced logits for position `K+1`, so one extra "free" token can be sampled there. Best case: `K+1` tokens per target forward pass.
6. **Why it speeds up.** A target forward pass with `K+1` tokens of input costs only marginally more than one with 1 token (memory-bound regime), but produces `1` to `K+1` accepted output tokens. Average tokens-per-target-call ends up well above 1 for well-matched draft/target pairs.

## Implementation sketch

```
1. SpeculativeDecoder behind a clean interface
   - Wrap the existing single-model decode path; choose between standard and speculative
     via config flag SPECULATIVE_ENABLED
   - Config: DRAFT_MODEL_NAME, SPEC_K (draft length per round)
   verify: with SPECULATIVE_ENABLED=false the path is bit-identical to today's decode

2. Draft model loading
   - Load draft model in the same startup hook as target (consistent with the single-startup
     constraint in CLAUDE.md)
   - Validate tokenizer compatibility (vocab IDs must match) at startup; fail fast if not
   verify: draft and target tokenizers produce identical token IDs for a sanity prompt

3. Draft + verify loop
   - Per decode iteration: K cheap draft steps → one target forward pass with K tokens
     of input → rejection sampling
   - Accepted tokens stream out of the engine in order; on rejection, the bonus replacement
     token streams next; loop continues from the new prefix
   verify: output is statistically indistinguishable from non-speculative sampling on a
           held-out prompt set (same temperature, same seed where possible)

4. KV cache handling
   - Both models maintain their own KV caches
   - On rejection at position j, both caches must be rolled back to position j-1
     (drop entries appended for x_j ... x_K)
   - This is straightforward with the paged cache: free the affected blocks
   verify: cache state after rejection matches what plain decode from the accepted prefix
           would produce

5. Acceptance-rate metric
   - Track per-request and engine-wide average accepted-tokens-per-target-call
   - Expose on /metrics
   verify: metric is positive and stable; compare against published numbers for similar
           draft/target pairs

6. Benchmark
   - End-to-end TPOT and tokens/sec with and without speculation, same workload
   verify: speedup proportional to acceptance rate; no change in output quality

7. (Future) Tree-style speculation (Medusa / EAGLE) reuses the draft+verify scaffold;
   only the draft proposal mechanism changes
```

## Notes on integration with other engine components

- **Paged KV cache:** rejection-driven rollback is a block-table truncation, not a tensor copy. The block-paged design makes speculative decoding *easier* to implement, not harder.
- **Continuous batching:** speculative decoding changes the per-request decode unit from "1 token per step" to "1-(K+1) tokens per step" with variance. The scheduler can stay as-is, but per-step throughput accounting needs to handle variable token counts cleanly.
- **Streaming:** accepted tokens stream as they are produced in order. The user-facing SSE stream sees no difference from non-speculative output.
- **Quantization / Loki / eviction:** all of these apply to the target model independently. Speculative decoding does not interact with them.
