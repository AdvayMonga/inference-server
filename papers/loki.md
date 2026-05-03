# Loki: Low-Rank Keys for Efficient Sparse Attention

**Link:** https://proceedings.neurips.cc/paper_files/paper/2024/file/1e027da6bec9ceb2ec37951ceeccae93-Paper-Conference.pdf
**Venue:** NeurIPS 2024
**Authors:** Prajwal Singhania et al. (UMD)
**Status:** understood, deferred
**Relevant phase:** Phase 6 (multi-user / CUDA), revisit after eviction policies land

## Core idea (in your words)

Key vectors live in a much lower-dimensional subspace than their nominal head_dim suggests. Loki exploits this: project keys into a small PCA subspace, compute *approximate* attention scores cheaply there, pick the top-k tokens, then run full-precision attention only on those k. KV cache stays full — this sparsifies the *compute*, not the storage.

## Why it matters for this project

- Orthogonal to LRU / Attention-Sink-LRU / H2O eviction in Phase 5 — those decide what to drop, Loki decides what to attend over. Could compose with any of them.
- Attacks the per-step attention cost rather than memory footprint, so it complements the paged KV cache rather than competing with it.
- Real CUDA speedup (~45% on Llama2-13B in the paper) only shows up with custom Triton kernels — fits naturally into the Phase 6 GPU/precision optimization work.

## Key mechanism

1. **Offline calibration:** run a small calibration corpus (paper used ~16 WikiText sequences) through the model, collect K vectors per layer per KV-head, run PCA on each, save principal components.
2. **At inference, per attention step:**
   - Project current Q and cached K into top-d PCA components (d ≈ 25-50% of head_dim).
   - Compute approximate scores in this low-dim space.
   - Top-k selection on approximate scores (k ≈ 12.5-25% of tokens).
   - Gather full-dim K, V for those k tokens, run full-precision softmax-attention.
3. Always protect special tokens (sinks) from being filtered out.

## Open questions / things to figure out before implementing

- How to override HF Transformers' attention cleanly on Gemma 4 — monkey-patch the attention module vs. running layers manually in our backend vs. `attn_implementation` registry?
- GQA correctness: PCA per KV-head, but query heads map to KV-heads in groups — confirm projection happens on the right axis.
- On MPS without Triton, expect *slowdown* not speedup. Implementation here is for **quality validation**, not throughput. Speedup story belongs on CUDA.
- Interaction with H2O: H2O tracks attention scores for eviction decisions. If Loki only computes scores on a top-k subset, H2O's score signal is incomplete. Need to think about whether we use approximate scores or full scores for H2O's bookkeeping.
- Calibration set choice — does in-domain calibration matter, or is WikiText fine for a chat model?

## Implementation sketch (when we come back to it)

```
1. scripts/calibrate_loki.py
     - Hook into model forward, collect K per layer per kv-head
     - Run PCA (torch.linalg.svd or sklearn), save .pt file
     verify: components file exists, shapes match num_layers × num_kv_heads × head_dim

2. src/inference_server/attention/loki.py
     - LokiAttention module replacing model's attention forward
     - Loads PCA components at startup
     verify: with d=head_dim and k=seq_len, output matches baseline exactly

3. Backend integration (config flag LOKI_ENABLED, LOKI_D, LOKI_K)
     verify: generation still works, sensible outputs

4. Quality benchmark
     verify: perplexity degradation on small eval set in line with paper (~6.8%)

5. (Phase 6, CUDA only) Triton kernel + throughput benchmark vs our own baseline
```
