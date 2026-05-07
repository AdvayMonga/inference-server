# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Link:** https://arxiv.org/abs/2205.14135
**Venue:** NeurIPS 2022
**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford)
**Status:** not read
**Relevant phase:** Phase 6 (CUDA / GPU optimization)

## Core idea

Attention's `O(N²)` memory footprint comes from materializing the full `N × N` score matrix in HBM. On modern GPUs, attention is bottlenecked by HBM reads and writes, not by FLOPs — the `softmax(QK^T) V` computation is fundamentally an IO problem. FlashAttention computes the same exact attention output without ever materializing the full score matrix. It tiles Q, K, and V into blocks that fit in on-chip SRAM, computes attention block-by-block using an online softmax that updates running max and sum statistics, and writes only the final output back to HBM. The result is the same numerical answer as standard attention, but with `O(N)` HBM traffic instead of `O(N²)` and substantially higher wall-clock throughput, especially at long context lengths.

## Why it matters

- Every modern serving stack uses some FlashAttention variant under the hood. vLLM's PagedAttention is itself a paged-memory variant of FlashAttention.
- The IO-awareness lens — count HBM accesses, not FLOPs — is the right way to reason about *every* attention kernel optimization that follows (FlashAttention-2, FlashAttention-3, FlashDecoding, paged variants). Reading this paper is the prerequisite for understanding the rest.
- Even on a non-CUDA dev box, the algorithm informs how a custom paged-attention kernel for the engine should be structured: tile-based, online softmax, single-pass. PyTorch's stock attention will become the limiting factor on long contexts well before the engine reaches vLLM-class throughput.

## Key mechanism

1. **Tile Q, K, V.** Split Q into blocks of `Br` rows; split K and V into blocks of `Bc` rows. Block sizes are chosen so that `Br × d` and `Bc × d` blocks fit in on-chip SRAM together with running statistics.
2. **Outer loop over K, V blocks; inner loop over Q blocks.** For each `(K_j, V_j)` block loaded into SRAM, iterate over Q blocks. For each `Q_i`, compute `S_ij = Q_i K_j^T` in SRAM, never writing it to HBM.
3. **Online softmax.** Maintain per-Q-row running statistics: max-so-far `m_i`, sum-of-exp `l_i`, and partial output `O_i`. When a new score block arrives, rescale the existing accumulators by `exp(m_i_old - m_i_new)` and add the new block's contribution. This produces the exact softmax output incrementally without ever holding the full `S_ij` row.
4. **Single HBM write.** Each tile's contribution to `O` is accumulated in registers/SRAM. Only the final `O` is written back to HBM. The softmax-normalization statistics never leave SRAM.
5. **Backward pass uses recomputation.** During training, the `S` matrix is recomputed from Q, K, V on the backward pass instead of being stored, trading FLOPs (cheap) for HBM traffic (expensive). Inference does not need this since there is no backward pass.
6. **Causal masking is fused.** For autoregressive attention, blocks above the diagonal are skipped entirely; the causal mask never materializes.

## Implementation sketch

```
1. Verify FlashAttention availability via PyTorch SDPA
   - torch.nn.functional.scaled_dot_product_attention auto-selects FA on CUDA
   - Confirm on the target CUDA backend that FA path is used (not the math fallback)
   verify: profiler shows FA kernel; HBM traffic scales with N, not N²

2. Replace HF model's attention path with SDPA-based attention
   - Most recent HF models support attn_implementation="sdpa" or "flash_attention_2"
   - Configurable via config flag ATTN_IMPL
   verify: end-to-end outputs match the eager-attention baseline within tolerance;
           per-step latency drops at long context

3. Long-context benchmark
   - TTFT and per-token latency vs context length, eager vs SDPA-FA
   - Sweep context from 512 to model max
   verify: per-token latency at long contexts drops noticeably; eager path becomes
           limiting factor before SDPA path does

4. (Custom paged-attention kernel work) Adopt FA-style tiling and online softmax
   - When implementing the custom paged attention for the block-paged KV cache,
     structure the kernel as a FA tile loop over (K, V) blocks indexed through the
     block table, with online softmax accumulators per Q row
   verify: matches non-paged FA output within tolerance on equivalent inputs

5. (FlashAttention-2 / 3) Track the variant available on the deployment hardware
   - FA-2 reduces non-matmul FLOPs and improves parallelism; FA-3 uses Hopper-specific
     features
   - Use whatever the installed PyTorch + CUDA version exposes through SDPA
   verify: deployment notes record which variant is active per hardware target
```

## Notes on integration with other engine components

- **Paged KV cache:** the engine's eventual custom paged-attention kernel should be a FlashAttention kernel that gathers K, V through the block table inside the tile loop rather than from a contiguous tensor. The two ideas compose; this composition is exactly what vLLM's PagedAttention is.
- **Loki:** Loki's gather-then-attend final stage is a small attention call over the selected top-k tokens. That call should itself use FA. Loki's win is on top of FA, not instead of it.
- **MPS / non-CUDA:** FlashAttention is CUDA-only in practice. On the MPS dev box, SDPA falls back to a math implementation. Performance claims belong to the CUDA path.
