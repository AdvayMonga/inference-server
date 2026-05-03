# BatchLLM: Optimizing Large Batched LLM Inference

**Link:** https://arxiv.org/abs/2412.03594
**Venue:** MLSys 2026 (accepted)
**Authors:** Zhen Zheng, Xin Ji, Taosong Fang, Fanghao Zhou, Chuanjie Liu, Gang Peng
**Status:** to read
**Relevant phase:** likely Phase 5 / Phase 6 (prefix sharing + scheduling)

## One-line summary (from abstract, not yet verified by reading)

Global prefix-sharing across batched requests + reordering of requests to maximize KV reuse, mixing prefill and decode. Reports 1.3×–10.8× over vLLM / SGLang.

## Why I want to read it

Directly relevant to our radix-tree prefix sharing and continuous batching design. Sounds like it pushes the prefix-sharing idea further than vLLM by being scheduler-aware about which requests to batch together.

## Core idea (in your words)

_(fill in after reading)_

## Why it matters for this project

_(fill in after reading)_

## Key mechanism

_(fill in after reading)_

## Open questions / things I don't get yet

_(fill in after reading)_

## Implementation sketch

_(fill in after reading, if we want to implement)_
