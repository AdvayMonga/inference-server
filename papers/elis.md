# ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor

**Link:** https://arxiv.org/abs/2505.09142
**Venue:** arXiv preprint, May 2025
**Authors:** Seungbeom Choi, Jeonghoe Goo, Eunjoo Jeon, Mingyu Yang, Minsung Jang
**Status:** to read
**Relevant phase:** Phase 6 (scheduler / fairness)

## One-line summary (from abstract, not yet verified by reading)

Iterative Shortest-Remaining-Time-First scheduler for LLM inference, using a trained predictor for response length to drive prioritization. Targets head-of-line blocking. ~19.6% job-completion-time reduction on real traces. Implemented as a K8s-native scheduler.

## Why I want to read it

Our `FairScheduler` (Phase 6) needs to make admit / preempt decisions. ELIS's idea — predict output length, schedule SRTF-style — is one concrete policy worth comparing against per-session fairness. Could plug in as another `SchedulerInterface` implementation.

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
