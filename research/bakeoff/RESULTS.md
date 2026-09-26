# Ranker bake-off — results

Pre-registration: [`PREREGISTRATION.md`](PREREGISTRATION.md) (committed first, 5082733).
Cards: [`cards.json`](cards.json) · answer key: [`answer_key.json`](answer_key.json) ·
raw ranker replies: [`raw/`](raw/) · scorer: [`score.py`](score.py) → [`results.json`](results.json).

## Question and rule

Can a ranker, shown only pre-result information about past changes, order them by how they
actually paid off? **Spearman rho < 0.3 against realized outcomes ⇒ REJECTED as a
hypothesis-selection aid.**

## Data

26 eligible ledger records → **21 cards** after the pre-registered merge (three tiled-prefill
records test one kernel; four graph-variant records test one change). Outcomes: eight 3s, four
2s, five 1s, four 0s (two of the 0s are `invalid`, excluded in the secondary analysis → n = 19).
Critical rho (one-sided, alpha 0.05, from the permutation null): **0.372** at n = 21, **0.388**
at n = 19.

## Results

rho(avg) = Spearman of the per-card score averaged over a model's 3 samples (the headline).
p = one-sided, 10,000 permutations. Primary: `invalid` = 0. Secondary: `invalid` excluded.

| ranker | per-sample rho (primary) | mean rho | rho(avg) primary | p | rho(avg) secondary | p | vs rho < 0.3 |
|---|---|---|---|---|---|---|---|
| haiku (claude-haiku-4-5) | 0.11, −0.20, 0.58 | 0.17 | 0.09 | 0.36 | 0.23 | 0.17 | **REJECT** |
| sonnet (claude-sonnet-5) | 0.63, 0.57, 0.64 | 0.61 | 0.71 | < 0.001 | 0.80 | < 0.001 | not rejected |
| opus (claude-opus-5) | 0.67, 0.72, 0.76 | 0.72 | 0.77 | < 0.001 | 0.76 | < 0.001 | not rejected |
| naive prior (category map) | 0.11 | 0.11 | 0.11 | 0.37 | 0.12 | 0.34 | **REJECT** |
| random | — | — | null distribution; 95th pct = 0.372 | — | 0.388 | — | — |
| Jev (jev-1.13.0) | not run — `TYPESAFE_API_KEY` unset | | | | | | pending |
| human | not run — `HUMAN_SHEET.md` unfilled | | | | | | pending |

Agreement (Spearman between averaged scores): sonnet~opus 0.76, haiku~sonnet 0.52,
haiku~opus 0.23; each vs naive prior ≤ 0.55. Self-consistency (mean pairwise rho between a
model's own samples): haiku 0.27, sonnet 0.73, opus 0.75.

Exploratory, not pre-registered: a "tier" ranker (probe-tier cards scored above sweep-tier
cards) gets rho 0.32, p 0.09 — so part of the signal is simply that isolated probes are easier
to win than end-to-end sweeps; sonnet and opus clear it by a wide margin. Dropping C21 (the one
card whose own mechanism says it is not meant to change speed) leaves sonnet 0.72 and opus
0.77.

## Verdict

- **haiku: REJECTED** (0.09 < 0.3; its samples disagree with each other about as much as with
  the outcomes).
- **naive category prior: REJECTED** (0.11).
- **sonnet and opus: NOT rejected** — 0.71 and 0.77, both above the critical value, and stable
  across card orders. The cheap falsifier did not kill the idea of a model-based hypothesis
  ranker for this loop; it does kill doing it with a small model or a keyword prior.

Not rejected is not accepted. What this supports is running the next step — ranking
*prospective* hypotheses before their A/B and scoring the ranker when the verdicts land — not
building a generator on the strength of a retrospective test.

## Caveats

- **n = 21, low power, coarse ordinal outcome with many ties.** A rho of 0.3 would be detected
  well under half the time; the 0.7+ results are large enough that power is not what limits
  them — the threats below are.
- **Blinding is imperfect.** The same agent read the ledger (verdicts, deltas, knowledge
  entries) before writing the cards. Numbers, verdict words, hindsight words and identifiers
  were stripped and the grep in `leakage_check.py` is clean on all 21 cards and all 9 ranker
  prompts, but word choice in the gap/mechanism fields could still carry the writer's knowledge.
  Several cards also state a *pre-change* observation that nearly determines the outcome
  (e.g. "the engine fails under stress", "most of load time is the init fill") — that is
  legitimate pre-result information, and it is what a real hypothesis carries, but it makes
  those cards easy.
- **Models may know the field's folklore** (tiled attention in Triton, mixed-batch prefill vs
  CUDA graphs, capture-order vs Dynamo specialisation). That is the point of an expert ranker,
  but it is also contamination risk the grep cannot see. None of this repo's content was in the
  prompt.
- **17 of 21 cards are back-filled records** reconstructed from a tuning log, with verdicts
  that predate the gates; a few (C09, C14) were scored 1 by the pre-registered
  "confirmed-without-improvement" rule, which is a judgment call made in advance.
- **Outcome is on each record's own metric**, often an isolated probe (kernel ms, capture
  seconds, hit rate), not end-to-end — while the ranker prompt asks about end-to-end. The
  rankers were told each card's metric and tier.
- **One opus sample (s2) was drawn three times**: the first two replies omitted one or two card
  ids and failed the parser (the first two were not kept; the runner now saves failed replies as
  `*.failedN.json`). The kept sample used the same card order. Sonnet/opus runs also list haiku
  in `modelUsage`: that is the CLI's auxiliary model, not the ranker.

## How isolation was enforced

Each ranker call ran `claude -p --model <alias> --tools "" --setting-sources ""
--strict-mcp-config --disable-slash-commands --no-session-persistence --output-format json` in a
fresh `tempfile.TemporaryDirectory()` outside the repo, cards on stdin, no answer key. A probe run
the same way confirmed no CLAUDE.md, memory or project file reached the model (only the CLI's
generic environment block: temp cwd, OS, date). `--bare` was not usable because it requires an
API key and none is set. `run_rankers.assert_isolated` refuses any prompt matching experiment /
hypothesis / knowledge ids, SHAs, branch names or dates; the saved prompts in `raw/` pass it.

## To run the pending arms

- **Jev:** `TYPESAFE_API_KEY=... python research/bakeoff/jev_ranker.py` then
  `python research/bakeoff/score.py`. Without the key it exits and writes nothing.
- **Human:** fill every `Score:` line in `HUMAN_SHEET.md` cold, then run `score.py`; it picks the
  sheet up once all 21 are filled.
