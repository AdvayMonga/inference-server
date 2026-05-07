# Loki: Low-Rank Keys for Efficient Sparse Attention

**Link:** https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf
**Venue:** NeurIPS 2024
**Authors:** Prajwal Singhania et al. (UMD)
**Status:** read
**Relevant phase:** Phase 6 (CUDA / GPU optimization), composes with eviction in Phase 5

## Core idea

Key vectors in a transformer's attention live in a much lower-dimensional subspace than their nominal `head_dim` would suggest. Loki exploits this by computing **approximate** attention scores in a low-rank PCA subspace, using those scores to pick a small top-k of relevant tokens, and only running full-precision attention over that top-k. The KV cache itself is not compressed and not evicted — Loki sparsifies the *compute*, not the storage.

## Why it matters

- Orthogonal to eviction policies (LRU, Attention-Sink-LRU, H2O): eviction decides what to drop from the cache, Loki decides what subset of the kept tokens to attend over per step. They compose.
- Attacks per-step attention cost rather than memory footprint, complementing the paged KV cache rather than competing with it.
- The reported ~45% throughput win on Llama2-13B with a Triton kernel makes it a credible Phase 6 lever once the engine is on CUDA.
- Quality degradation (~6.8% reported) is small enough that it's a viable always-on optimization rather than a long-context-only trick.

## Key mechanism

1. **Offline calibration.** A small calibration corpus (paper used ~16 WikiText sequences) is run through the model. K vectors are collected per layer per KV-head. PCA is run independently on each (layer, kv-head) bucket. All principal components are stored, but only the top `d` (≈25-50% of `head_dim`) will be used at inference.
2. **Per-step approximate scoring.** At each decode step, the current Q and the cached K are projected into the top-`d` PCA subspace. Approximate attention scores are computed in that low-dim space — cheap because both the matmul and the memory read are `d / head_dim` smaller.
3. **Top-k selection.** From the approximate scores, the top-`k` tokens (≈12.5-25% of the sequence) are selected. Special tokens (attention sinks, beginning-of-sequence) are always retained regardless of score.
4. **Full-precision attention on the subset.** Full-dimensional K and V are gathered for the selected `k` tokens, and the final softmax-weighted attention is computed over that subset.
5. **Triton kernel.** The paper's reported speedup comes from a fused Triton kernel that does the projection, top-k, gather, and final attention without materializing intermediate tensors. Pure-PyTorch implementations are correct but do not realize the speedup.

## Implementation sketch

```
1. scripts/calibrate_loki.py
   - Hook the model's attention modules; collect K per layer per kv-head over the calibration corpus
   - Run PCA per (layer, kv-head) bucket via torch.linalg.svd
   - Save components as a single .pt artifact: tensor of shape (num_layers, num_kv_heads, head_dim, head_dim)
   verify: artifact loads, shapes match the model config

2. src/inference_server/attention/loki.py
   - LokiAttention forward replacing the model's attention path
   - Loads PCA components at backend startup (runs in the single startup hook, see CLAUDE.md Modal constraint)
   - Config flags: LOKI_ENABLED, LOKI_D (rank), LOKI_K (top-k count or fraction), LOKI_PROTECT_SINKS
   verify: with d=head_dim and k=seq_len, output exactly matches baseline attention

3. Backend integration
   - Monkey-patch the loaded HF model's attention modules at startup when LOKI_ENABLED=true
   - No changes to the scheduler, cache manager, or server
   verify: end-to-end generation still produces sensible outputs

4. Quality benchmark
   - Perplexity on a small held-out eval set with and without Loki across (d, k) sweep
   - Compare absolute PPL delta to the paper's ~6.8% claim
   verify: quality degradation in line with paper for at least one (d, k) configuration

5. (CUDA phase) Triton kernel
   - Fused projection + top-k + gather + attention
   - Throughput benchmark vs PyTorch baseline on the same hardware
   verify: per-step attention latency reduction matches paper's order of magnitude

6. Compose with eviction
   - Run Loki on top of LRU and on top of Attention-Sink-LRU
   - Confirm sink protection is consistent between Loki's "always keep sinks" and the eviction policy
   verify: no quality regression beyond the sum of the two techniques' individual deltas
```

## Notes on integration with other engine components

- **Paged KV cache:** Loki reads K through the same indirection path as full attention; the gather step naturally handles non-contiguous K. No extra work.
- **H2O eviction:** H2O's bookkeeping uses attention scores. If Loki only computes scores on its top-k subset, H2O's signal is incomplete. Two options: (a) track approximate scores from Loki's low-rank pass for H2O's per-token accumulator (cheap, slightly biased), or (b) periodically run a full-attention pass to refresh H2O's scores (expensive, accurate). Decision deferred until both are implemented.
- **Continuous batching:** Loki is per-request and per-layer; nothing in the scheduler needs to change.
