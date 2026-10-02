# ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor

**Link:** https://arxiv.org/abs/2505.09142
**Venue:** arXiv preprint, May 2025
**Authors:** Seungbeom Choi, Jeonghoe Goo, Eunjoo Jeon, Mingyu Yang, Minsung Jang
**Status:** not read
**Relevant phase:** Phase 6 (scheduler / fairness)

## Core idea

Iteration-level batching (vLLM-style) admits new requests at every generation step, but if it admits them in FIFO order, a long-output request can sit at the head of the queue and force every shorter request behind it to wait or share a degraded slot — classic head-of-line blocking. ELIS sidesteps this by **predicting the output length of each request before it runs** using a trained encoder model (BGE-based), then running an **Iterative Shortest-Remaining-Time-First (ISRTF)** scheduler that prefers requests with fewer predicted-remaining tokens. Implemented as a Kubernetes-native serving system, ELIS reports ~19.6% reduction in average job completion time on real-world traces.

## Why it matters

- The project's planned `FairScheduler` needs at least one concrete admission/preemption policy. ISRTF is a credible second policy alongside per-session fairness — they target different workloads (ISRTF wins on JCT for mixed-length traffic, fairness wins when one session would starve others).
- A length predictor is a small, well-defined component that plugs in behind a clean interface (`predict_remaining_tokens(request) → int`). Different predictors (BGE, regex over prompt patterns, no predictor) become swappable without touching the scheduler.
- The 19.6% JCT reduction is on top of an already iteration-batched baseline, so the win is purely from scheduling, not from any kernel or memory work — clean to attribute and measure.

## Key mechanism

1. **Response length predictor.** A BGE encoder model is fine-tuned to map a prompt (and optionally request metadata) to an expected output token count. The predictor runs once per request at admission time, not per step. Training data comes from logged prompt → actual output length pairs.
2. **ISRTF scheduling.** At each iteration, the scheduler ranks pending and in-flight requests by predicted remaining tokens (`predicted_total - generated_so_far`) and prefers smaller values. New short requests can preempt or out-prioritize long in-flight requests when the engine has admit/preempt slots available.
3. **Iteration-batching aware.** Unlike classical SRTF on whole jobs, ISRTF operates at the iteration boundary. Each step the scheduler reconsiders the running set; long requests are not killed mid-flight, but newly admitted short requests are placed ahead of pending long ones.
4. **Prediction-error tolerance.** Predictor errors are unavoidable. The "remaining" count is updated each step using `max(0, predicted_total - generated_so_far)`. If a request runs past its predicted total, its remaining hits zero and it loses priority on subsequent ties — long-tail requests degrade gracefully instead of catastrophically.
5. **Kubernetes-native deployment.** ELIS ships as a K8s scheduler component, meaning request routing across replicas is part of the system, not just per-engine scheduling. For this project, only the per-engine ISRTF logic is directly applicable; the multi-replica routing is platform-layer (deferred per the Forward-Compatibility Constraint in CLAUDE.md).

## Implementation sketch

```
1. LengthPredictor abstract interface
   - predict(request) -> int (predicted total output tokens)
   - Implementations: NaivePredictor (constant), BGEPredictor (trained encoder),
     HeuristicPredictor (regex/feature rules over prompt)
   verify: NaivePredictor returns the configured constant for any input

2. ISRTFScheduler implementation behind SchedulerInterface
   - At each tick: rank candidates by (predicted_total - generated_so_far) ascending
   - Apply fairness / priority-class as outer constraint, ISRTF as inner ordering
   verify: synthetic workload of mixed short/long requests shows shorter ones complete
           sooner under ISRTF than under FIFO

3. BGE-based predictor (optional, behind the interface)
   - Load a BGE encoder via HF Transformers in the startup hook
   - Add a calibration script: run prompts → log actual output lengths → fine-tune
   - Predict once at admission, cache the result on the request object
   verify: predicted vs actual length correlation > 0.5 on a held-out set

4. Prediction-error handling
   - When generated_so_far >= predicted_total, set remaining = 0 (do not go negative)
   - Optionally: emit a metric for prediction error so the predictor can be retrained
   verify: a request that runs past its prediction does not crash or hang the scheduler

5. Benchmark
   - Mixed short/long workload from a representative trace
   - Compare avg JCT, p95 JCT, and throughput under ISRTF vs FIFO vs FairScheduler
   verify: ISRTF reduces avg JCT vs FIFO without catastrophic p99 regression on long requests
```

## Notes on integration with other engine components

- **FairScheduler vs ISRTF:** these are alternative `SchedulerInterface` implementations, not layers. Configuration selects one. A composite scheduler (fair-across-sessions, ISRTF-within-session) is a future option but unnecessary for a first cut.
- **Continuous batching:** ISRTF assumes iteration-level scheduling — it is an admission-ordering policy that runs at every batch step. No changes to batching mechanics required.
- **Modal deployment:** the predictor model loads at startup (consistent with the single-startup-hook constraint in CLAUDE.md). Predictor inference per-request is cheap (encoder forward pass on a short prompt) and lives in-process; no external service.
