# Step 2 — Hypothesize (the runbook)

The one step that is judgement, not code. Everything either side of it is deterministic, so this
is where the loop can be wrong in interesting ways — and where it must be disciplined.

**Inputs:** `gaps.json` from `loop attribute`, and the knowledge base.
**Output:** `hypotheses.json` — a list of `Hypothesis` records.

---

## Before proposing anything

```bash
python -m inference_server.research.loop kb --status rejected      # the dead ends
python -m inference_server.research.loop kb --tags prefill cache   # what is known here
```

If a candidate restates a `rejected` entry, it is not a hypothesis — it is a retry, and it needs
a reason the earlier measurement no longer applies. `loop screen` flags likely rehashes, but it
matches on keywords: read the entry, do not trust the flag.

## Every hypothesis must carry

| field | why it is mandatory |
|---|---|
| `predicted_magnitude` | **written before measuring.** Roughly half the hypotheses in this project were wrong; recording the prediction is what makes that visible instead of rationalised afterwards. "lm_head slicing will be a big win" was predicted large and measured 3-8%. |
| `falsification_test` | the *cheapest* thing that could prove this wrong — not the thing that would prove it right |
| `falsification_tier` | 1 static · 2 CPU · 3 single-GPU probe · 4 full sweep. The KV block leak was proven on CPU in ~2s; it would have cost a 20-minute GPU run to find the same thing. |
| `requires` | what the test needs to *run*: any of `cuda` `triton` `cuda_graphs` `compile` `vllm` `linux` `bf16` `large_vram`. Empty means any machine. Tier says how much it costs; this says where `loop screen` sends it. A scheduler change needs nothing and runs on MPS for $0; a Triton kernel needs `cuda` and goes to the cheapest CUDA venue you have enabled. |
| `gap_id` | which attributed gap this attacks. No gap, no hypothesis. |
| `predicted_metric` | a real panel field — and the one the gates will judge |

## Kinds — the loop can point at itself

- `engine_change` — the usual case.
- `instrument_change` — "we cannot see X". The TTFT queue/prefill split was this, and it ruled
  out an entire class of fix in one run.
- `harness_change` — "this harness cannot test X". The K=1 wave histogram was this, and it
  killed two planned projects.

**When attribution reports a `harness-*` gap, the next hypothesis must be a `harness_change`.**
Optimising against a harness you already know is blind is how a month gets spent on the wrong
bottleneck.

## Good and bad shapes

> **Good:** "Prefill dominates TTFT p95 (203ms of 213ms). The graph is gated to `matched==0` and
> the cache is warm, so it rarely fires. Letting it fire on hits should cut TTFT p50 >30%.
> Falsify at tier 3: one A/B on a warm cache — if p50 does not move, the gate is not why."

Concrete, cites the gap, predicts a number, and names a cheap way to be wrong.

> **Bad:** "Try quantisation, it usually helps."

No gap, no magnitude, no falsification, and the knowledge base already says why it is not the
lever here.

## Rules

1. Never propose a change to something attribution did not flag.
2. Prefer the hypothesis whose falsification is cheapest, not the one with the biggest prize.
3. One variable per experiment. Two changes make a confirmed result unattributable.
4. If a `harness-*` gap exists, fix the harness first.
5. A hypothesis you cannot state a way to falsify is an opinion. Do not run it.
