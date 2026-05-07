# TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference

**Link:** https://arxiv.org/abs/2505.11329
**Venue:** arXiv preprint, May 2025 (v4 Oct 2025)
**Authors:** Raja Gond, Nipun Kwatra, Ramachandran Ramjee
**Status:** not read
**Relevant phase:** future / multi-GPU extension (out of scope for the current roadmap)

## Core idea

In tensor-parallel inference, every transformer layer ends with an `AllReduce` across GPUs to sum partial outputs. At low batch sizes — the regime that matters for low-latency serving — this collective sits on the critical path and cannot be hidden behind compute, because there is too little compute to overlap with. TokenWeave splits the tokens of each in-flight batch into two roughly equal subsets and pipelines them: while subset A is doing its `AllReduce`, subset B is doing its compute, and vice versa. The communication and the compute then overlap by construction, even at small batch sizes. A fused `AllReduce + RMSNorm` kernel using NVSHARP/Multimem (Hopper/Blackwell) does the comm step on 2-8 SMs, leaving the rest of the GPU free for the other subset's compute. Reported wins: 1.29× latency, 1.26× throughput.

## Why it matters

- Tensor parallelism is in the optional-extensions section of the roadmap, not a near-term priority. This paper is reference material for that future work, not something to implement now.
- It is the right paper to read *before* designing a TP path, because the standard "do compute, then AllReduce, then next layer" pipeline is what this work argues against. Building TP naively and then bolting on overlap later means rewriting the model code.
- The fused `AllReduce + RMSNorm` kernel is a hardware-specific optimization (Hopper/Blackwell with NVSHARP) — it is the speedup mechanism, but the *architectural* contribution (split-batch overlap) applies to any multi-GPU setup.
- Notable that vLLM, SGLang, and TensorRT-LLM all leave overlap off by default in TP serving — suggests the integration cost has historically been higher than the benefit, and that real wins require the kernel-level work the paper does.

## Key mechanism

1. **Token-subset splitting.** Each in-flight batch is partitioned into two subsets of roughly equal token count. The subsets are processed in an interleaved schedule rather than as one monolithic batch.
2. **Two-phase pipeline per layer.** Within one transformer layer, subset A's compute (Q/K/V projection, attention, FFN partials) runs while subset B's `AllReduce` is in flight, then subset B's compute runs while subset A's `AllReduce` is in flight. The naive pipeline (compute-all → comm-all) is replaced by an interleaved one.
3. **Fused AllReduce + RMSNorm kernel.** The next layer normalizes its inputs with RMSNorm immediately after `AllReduce`. Fusing these two operations avoids materializing the post-AllReduce tensor in HBM and reduces the comm step's footprint to 2-8 SMs via NVSHARP/Multimem (Hopper and Blackwell hardware features for in-network reduction).
4. **Low SM occupancy for comm.** Because the fused kernel uses only 2-8 SMs, the remaining ~100+ SMs on the GPU are free to run the other subset's compute. This is what makes overlap practical at small batch sizes — naive `AllReduce` saturates many more SMs and leaves no compute headroom.
5. **Code-level integration.** Each linear layer that ends in `AllReduce` is replaced with the fused variant, and the per-layer execution loop is rewritten to issue subset-A and subset-B operations on alternating streams.

## Implementation sketch

```
0. Prerequisite: tensor-parallel inference path
   - Not in the current roadmap; would require splitting model weights across GPUs,
     adding NCCL/NVSHARP communicators, and a TP-aware backend
   verify: a baseline TP path runs the model with naive compute → AllReduce → next-layer

1. Token-subset splitter
   - Given an in-flight batch, partition tokens into two subsets balanced by count
     (and ideally by sequence boundary so attention masks stay contiguous)
   verify: round-trip a batch through split/merge and recover identical tensors

2. Two-stream pipeline executor
   - Per layer: alternate compute(subset_A) || allreduce(subset_B) on two CUDA streams
   - Synchronize at layer boundaries
   verify: with overlap disabled, output bit-identical to baseline; with overlap enabled,
           output matches within numerical tolerance

3. Fused AllReduce + RMSNorm kernel
   - Hopper/Blackwell only, NVSHARP/Multimem path
   - Falls back to non-fused on older hardware (correctness preserved, no speedup)
   verify: kernel output matches naive (allreduce → rmsnorm) within tolerance

4. Benchmark
   - TP=2, TP=4, TP=8 across batch sizes from low (latency regime) to high (throughput regime)
   - Compare end-to-end latency and throughput vs the same TP path with overlap off
   verify: latency improvement at low batch sizes consistent with paper's order of magnitude

5. Defer until multi-GPU is on the project roadmap
```

## Notes on integration with other engine components

- **Out of scope for the current engine.** This project is single-GPU through Phase 11. TokenWeave only matters once a TP path exists.
- **Architectural seam:** if and when TP is added, the per-layer execution loop in the backend must already be structured so that the comm and compute steps are separable operations — otherwise integrating overlap means rewriting the layer loop. Worth noting now even though no code is written.
- **Modal deployment:** Modal supports multi-GPU containers, so TokenWeave-style overlap is reachable from the planned deployment target without changing the deployment model.
