# TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference

**Link:** https://arxiv.org/abs/2505.11329
**Venue:** arXiv preprint, May 2025 (v4 Oct 2025)
**Authors:** Raja Gond, Nipun Kwatra, Ramachandran Ramjee
**Status:** to read
**Relevant phase:** future / multi-GPU extension (not in current roadmap)

## One-line summary (from abstract, not yet verified by reading)

Distributed (tensor-parallel) LLM inference: split each batch's tokens into two roughly equal subsets so communication on one subset overlaps with compute on the other. Includes a fused AllReduce + RMSNorm kernel for Hopper/Blackwell. Reports 1.29× latency, 1.26× throughput improvements.

## Why I want to read it

Tensor parallelism is in the "Optional Extensions" section of our roadmap, not a near-term priority. But this is the kind of paper that's worth understanding before designing a multi-GPU path — overlap-with-comm is the dominant axis once you go past one GPU.

## Core idea (in your words)

_(fill in after reading)_

## Why it matters for this project

_(fill in after reading)_

## Key mechanism

_(fill in after reading)_

## Open questions / things I don't get yet

_(fill in after reading)_

## Implementation sketch

_(fill in after reading, if we want to implement — likely deferred until multi-GPU is on the table)_
