# Ranker bake-off — pre-registration

Written and committed before any card is written or any ranker is run. Git history is the
timestamp.

## Question

Is an AI hypothesis generator worth building for this repo's research loop? The cheapest
falsifier: hide the outcomes of past experiments, ask rankers to predict which proposed changes
paid off, and measure rank correlation against what actually happened. A ranker that cannot
order *past* hypotheses by payoff is no use for choosing *future* ones.

## Decision rule

**A ranker with Spearman rho < 0.3 against realized outcomes is REJECTED as a
hypothesis-selection aid.** rho >= 0.3 means "not rejected", not "accepted": acceptance also
needs the permutation p-value below 0.05, and at this n those two conditions are not the same
(see Power). If every LLM ranker is rejected, the answer to the question is "not worth
building on this evidence".

## Population

`experiments/*.json`, excluding:
- records with a non-empty `no_behaviour_change` (they claim no delta), and
- correctness fixes with a non-empty `regression_test` (the assertion is the measurement; no
  predicted performance delta).

The remainder is the 2026-09-05 loop iterations, `exp-20260917-2ee69bde` and the
`exp-backfill-*` records (26 records).

## Unit

One card per experiment whose **treatment** differs. Two or more records are merged into one
card only if they test the identical change, established mechanically: same branch and
hypothesis family, and `git diff` between their treatment SHAs touches nothing under
`src/inference_server/` outside `research/` (i.e. only the harness or gates changed). A merged
card takes the outcome of its **latest** record by `started_at` — the ledger's final judgment.

Known at registration time (the ledger was read to define the unit, before any card was
written): the three `perf/tiled-prefill-attention` records for hypotheses iter3/iter4 test the
same kernel (diffs between 7998c8b, a7e11de and ce61cd8 touch only `research/`) and merge; the
BLOCK_M=16 variant (326ad2f) changes the kernel and is its own card; the four
`feat/prefill-graph-variants` records (1b33900, 6ae983c: only `research/` differs) merge.
Expected n = 21.

## Outcome score (per card)

From the record's `verdict` and its first `delta` entry:

| score | condition |
|---|---|
| 3 | `confirmed`, and the delta moved >= 30% in the improving direction |
| 2 | `confirmed`, smaller than that |
| 1 | `noise` |
| 0 | `rejected` |
| 0 / excluded | `invalid` — scored 0 in the **primary** analysis, EXCLUDED in the **secondary** |

Mechanical clarifications, fixed now:
- **Improving direction** is lower-is-better for latency, time, ms/step, seconds and error
  counts; higher-is-better for throughput, hit rate and capacity.
- **pct is null** (baseline 0): an improving move from zero counts as >= 30%.
- **Confirmed without an improvement.** A `confirmed` record whose significance gate did not
  pass, or passed only because the change was declared an instrument change ("instrument
  change" / "no measurable change, as intended"), confirmed neutrality, not a win. It scores 1,
  as noise. The rankers are asked about improvement, so this is the outcome they are predicting.
- A `confirmed` record whose delta moved in the non-improving direction and is not covered by
  the previous rule scores 2 (literal reading of "confirmed smaller").

## Statistic

Spearman rho (Pearson on mid-ranks; ties get average ranks) between a ranker's card scores and
the outcome scores. One-sided permutation p-value, 10,000 permutations of the ranker scores
against the outcomes, H1: rho > 0. Reported per ranker:
- rho and p per sample (3 samples per LLM model, each with a different card order);
- mean rho over samples;
- rho of the per-card averaged score over samples (the **headline** per model);
- both primary (invalid = 0) and secondary (invalid excluded) analyses;
- inter-model agreement: Spearman rho between models' averaged scores, and between samples of
  one model (self-consistency).

## Rankers

1. **LLM rankers**: haiku, sonnet, opus via the Claude CLI, print mode, all tools disabled, run
   in an empty temp directory outside the repo with no project settings/CLAUDE.md/memory, cards
   on stdin, no answer key. Prompt (fixed): "You are an LLM-inference-engine performance
   engineer. For each proposed change, score 0-3 how likely it is to produce a confirmed,
   significant end-to-end improvement on its target metric in a careful A/B with >=3 runs per
   arm on a single rented A100: 0 = no effect/invalid/regression, 3 = large confirmed win.
   Output JSON {card_id: score, ...} only, plus a one-line reason each." 3 samples per model.
2. **Random**: the permutation null distribution itself.
3. **Naive prior** (defined here): each card carries a `category` assigned from the diff (the
   change, not its result). Score = 2 for `kernel`, `cuda_graph`, `launch_overhead`; 1 for
   `scheduling`; 1 for every other category.
4. **Jev** (TypeSafe System One, `jev-1.13.0`): same cards, same scorer; runs only with
   `TYPESAFE_API_KEY`. No results are fabricated when it is absent.
5. **Human**: the repo owner scores the shuffled cards cold on `HUMAN_SHEET.md`.

## Power

n ~ 21 is **low power**. The one-sided critical rho at alpha = 0.05 for n = 21 is about 0.37
(two-sided about 0.44); the exact value for the n obtained is computed from the permutation
distribution and reported. The 0.3 threshold therefore sits *below* significance: a ranker can
clear 0.3 by luck, and a true rho of 0.3 would be detected well under half the time. A
rejection here is "no evidence of useful signal at this n", which is the pre-registered
purpose — a cheap falsifier, not a proof.

## Known threats, stated in advance

- The card writer (the same agent) read the ledger, including outcomes, before writing cards.
  Blinding is by construction (strip numbers, verdict words, hindsight phrasing) plus a
  mechanical leakage grep, not by writer ignorance.
- The naive prior was defined after the ledger was read; its category map is the one proposed
  in the task brief, with every other category at the neutral 1, to limit discretion.
- Outcome is a coarse ordinal with many ties.
- The `exp-backfill-*` records were reconstructed from a tuning log, not run through the gates.
