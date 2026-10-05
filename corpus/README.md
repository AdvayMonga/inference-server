# corpus/ — frozen workload traces

The corpus defines the landscape the lab measures. Anything not in it is invisible.

Since 2026-10-01 it is built from **real public traces**: real arrival times and sessions from
BurstGPT, real conversation text (with the real assistant replies) from WildChat-1M. The
previous synthetic corpus (Poisson arrivals over `scripts/bench/prompt_bank.py`, multi-turn by
prompt concatenation, corpus_version `659ea3b6…`) remains in git history. **Every panel
measured against the old version is non-comparable with this one** — the eval harness refuses
across `corpus_version`, and the noise bands in `knowledge/noise/` do not carry over.

## A trace

`<class>/<split>.jsonl`, one request per line: `arrival_s` (offset from the window's first
request), `session_id`, `turn_index`, `prompt`, `max_tokens`, `sampling` (temperature 0, so
output is deterministic and the correctness gate can compare it), `expected_output_hash`
(`null` until a reference run fills it), `build_prompt_tokens` (the request's templated length
under the build tokenizer, so floor/cap checks need no tokenizer), and — on a multi-turn request
only — `messages`.

- **Turn 0** has no `messages`: `prompt` is the user's first message.
- **Turn k > 0** carries `messages`, a real WildChat conversation `[user, assistant, …, user]`
  ending in the user's k-th follow-up, with WildChat's stored assistant replies (from
  GPT-3.5/GPT-4). `prompt` is the same contents joined by blank lines, for the raw route and
  the simulator's length model.
- `max_tokens` is 2048 everywhere, deliberately generous so answers end on their own stop token
  (the trace's response tokens are NOT used). `expected_output_tokens` is unset until a GPU
  reference run measures it.

An open-loop replayer (the eval harness's job) fires each request at its `arrival_s` on the client's own clock — open loop —
and posts `messages` (or one user message) to `/v1/chat/completions`, so the serving model's
chat template is applied **server-side**: traces never store templated text and stay
model-independent. A chat-route panel carries a verified `chat_template` fingerprint (checked
against a single-message request) and a changed one makes two runs non-comparable.
`--prompt-format raw` posts `prompt` to `/v1/completions`. `session_id` / `turn_index` ride as
`X-Session-Id` / `X-Turn-Index`, and `X-Trace-Id=<class>-<split>-<nonce>-<index>` joins each CSV
row to its telemetry row.

## Classes

| class | window | requests (seen / heldout) | proposed SLO |
|---|---|---|---|
| `cold_start` | first 60 requests after a ≥10 min idle gap, all within 20 min, varied lengths | 60 / 60 | p95 TTFT < 5000 ms |
| `steady_interactive` | a 10-min window of 200–300 requests, no minute above 2× the mean | 238 / 241 | p95 TTFT < 1000 ms, TPOT < 100 ms |
| `long_context` | 60 consecutive trace requests of ≥3000 tokens | 60 / 60 | p95 TTFT < 3000 ms |
| `spike` | a minute of 40–150 requests after ten quiet minutes, 5 min lead-in, 3 min tail | 290 / 272 | p95 TTFT < 2000 ms |
| `mixed_1`–`mixed_3` | 20 real minutes crossing regimes (see below), one trace day each | 938 / 1060, 596 / 384, 1029 / 1231 | per request, by its `regime` |

SLOs are **proposed 2026-10-01, pending owner confirmation** (also in `manifest.notes`).
Confirming or changing one is a new corpus version.

**Mixed classes.** Real traffic switches regimes; these test the switching. Each is a 20-minute
window holding at least 20 post-idle, 20 burst and 50 steady requests, with steady traffic in at
least 10 of its minutes and request-token CV ≥ 0.3 (so not one scripted client's burst), capped
at 1.1 req/s and 150 in any minute so one replica can take it in real time. Each mixed class
comes from a different trace day, so the three together are one GPU-hour of replay per split.
Every request carries `regime`, the class whose SLO judges it, labelled from its context in the
whole trace, first match wins: prompt ≥ 3000 tokens → `long_context`; one of the first 60
requests within 20 min after a ≥ 10 min gap → `cold_start`; arriving in a burst minute (≥ 40
requests, ≥ 6× the median and > 2× the max of the ten minutes before) → `spike`; else
`steady_interactive`. Mixed classes have no class SLO (`slo_ttft_ms` is null). Windows that fit
are rare: three days pass in each split, so these are those days, not a sample. To find them
the mixed classes also draw on BurstGPT's two earlier files, which have no session ids (so the
mixed traces are nearly all single requests), on `BurstGPT_3`'s clock (earlier months are
negative times), so one week numbering splits every class.

**Splits.** `seen` windows come from even weeks of the BurstGPT trace, `heldout` from odd
weeks — disjoint time, never the same window. No WildChat conversation is used twice anywhere
in the corpus, so prompts and sessions are disjoint across splits too.

**Rate scale.** Windows are *selected* to fit one replica (each class's criteria above), then
every inter-arrival is stretched by one factor, `dilation = max(1, rate / target)`, so the burst
shape is kept exactly. The target is the class's `arrival_rate_rps` (mean rate; the peak minute
for `spike`). Each trace's factor is recorded in `manifest.json` → `dilation` (hashed into the
corpus_version); **a replayer's rate dilation restores the real timing**. Every
trace in this version has dilation 1.0, i.e. is already in real time. BurstGPT timestamps have
1 s resolution, so several requests often share an `arrival_s`.

**What a `cold_start` window is, in BurstGPT terms.** Post-idle traffic in BurstGPT is mostly
night-time (trace-local 00:00–09:00) and mostly API-log requests from scripted clients: of the
post-gap windows that reach 60 requests within 20 min, almost all are one client firing
near-identical prompts (e.g. 60 requests of 343–449 tokens in 33 s, or 60 pings of 12–40
tokens in 12 s). Those are rejected: a window must have request-token CV ≥ 0.3 and at most 60%
of its requests within ±25% of the median. Only a handful of windows pass (one in the seen
weeks, two in the heldout weeks), so `cold_start` is "the first organic-looking burst after
quiet", not a typical one — and its seen and heldout traces are single windows, so treat a
cold_start p95 as a property of those windows rather than of BurstGPT at large.

**Pairing.** Each BurstGPT session in the window gets one unused WildChat conversation; each
of its requests gets the conversation turn whose templated prompt length (Qwen3-30B-A3B
tokenizer, `enable_thinking=False`) is closest in ratio to the trace's `Request tokens`, turns
strictly increasing within the session. A session longer than 16 requests is split into
consecutive conversations. API-log requests have no session: each is its own session and may
land on any turn, so many single requests are mid-conversation turns whose earlier turns are
not in the trace (a cold prefix). Failed requests (logged with 0 request tokens, 7.3% of
BurstGPT) are dropped. Returning sessions are rare in the real trace (0–7 per trace).

**Why no Azure split.** The Azure 2024 trace was evaluated as a cross-source held-out and does
not fit cleanly: it has no session ids (no multi-turn, no prefix reuse); the manifest schema
has exactly two splits per class, and a third would ripple through `corpus.py`, the replay CLI,
the held-out gate; and at 45 req/s mean it needs ~100× thinning, not dilation,
to fit one replica, so its burst shape cannot be preserved the same way. It is used as a sanity
check instead (`build_corpus.py --azure-check`):

| trace | mean req/s | inter-arrival CV | per-minute p50 / p99 / max | prompt tokens p50 / p90 / p99 |
|---|---|---|---|---|
| BurstGPT_3 (no failures) | 0.52 | 17.2 | 4 / 450 / 917 | 340 / 884 / 3482 |
| Azure conv 2024 | 45.2 | 1.7 | 2756 / 4499 / 5112 | 928 / 3830 / 6683 |

BurstGPT is far burstier and somewhat shorter-prompted than Azure's conversation service; a
policy tuned on this corpus should be re-checked against a smoother, longer-context mix.

## Rebuilding

```bash
uv sync --extra dev --extra corpus                          # pyarrow, for the WildChat shards
.venv/bin/python scripts/corpus/fetch_traces.py              # ~1.8 GB into ~/.cache/inference-server/traces
.venv/bin/python scripts/corpus/build_corpus.py               # seed 20261001
```

`fetch_traces.py` pins every URL to a release or revision and every file to a sha256, refuses
a mismatch, and is idempotent. Raw data never enters git (`TRACE_CACHE` overrides the cache
dir). The build is deterministic in (seed, cached inputs, tokenizer); `--tokenizer` makes the
pairing tokenizer a setting, and a different one is a new corpus version.
`tests/test_lab_corpus.py` rebuilds from the cache and asserts the committed bytes (it
skips when the cache is absent).

## The rule

`corpus_version` is a sha256 over every trace's hash and the class table (SLOs, rates, paths).
`load_manifest()` re-hashes each file and refuses the corpus on any mismatch, and two
measurements whose versions differ are not comparable. So: **never edit a trace or an SLO in place.** Any
change is a rebuild that produces a new version.

## Sources and attribution

- **BurstGPT** — HPMLL, <https://github.com/HPMLL/BurstGPT>, release v2.0, `BurstGPT_3.csv`
  (all classes) and `BurstGPT_1.csv`, `BurstGPT_2.csv` (mixed classes only).
  Licensed CC-BY-4.0. Yuxin Wang, Yuhan Chen, Zeyu Li, Xueze Kang, Zhenheng Tang, Rui Guo,
  Xin Wang, Qiang Wang, Amelie Chi Zhou, Xiaowen Chu. "BurstGPT: A Real-world Workload Dataset
  to Optimize LLM Serving Systems", arXiv:2401.17644 (2024). Changes: failed rows dropped;
  windows selected and time-dilated as above.
- **WildChat-1M** — Allen Institute for AI, <https://huggingface.co/datasets/allenai/WildChat-1M>,
  revision `7d6490e4`, shards 0–1. Licensed ODC-BY 1.0
  (<https://opendatacommons.org/licenses/by/1-0/>). Wenting Zhao, Xiang Ren, Jack Hessel,
  Claire Cardie, Yejin Choi, Yuntian Deng. "WildChat: 1M ChatGPT Interaction Logs in the Wild",
  ICLR 2024 (arXiv:2405.01470). Changes: English, non-toxic, unflagged, unredacted conversations
  only; any conversation containing a credential-like string (API keys, tokens, private keys,
  placeholders included; `SECRET_RE` in `build_corpus.py`) dropped; per-user metadata (IP hashes, location, headers) dropped; conversations truncated at
  the paired turn.
- **Azure LLM inference trace 2024** — Microsoft, <https://github.com/Azure/AzurePublicDataset>,
  `AzureLLMInferenceTrace_conv_1week.csv`. Licensed CC-BY-4.0. Jovan Stojkovic, Chaojie Zhang,
  Íñigo Goiri, Josep Torrellas, Esha Choukse. "DynamoLLM: Designing LLM Inference Clusters for
  Performance and Energy Efficiency", HPCA 2025. Used for the statistics above only; nothing
  from it is in the traces.
- **Tokenizer** — `Qwen/Qwen3-30B-A3B` (revision `ad44e777`, Apache-2.0), tokenizer files only,
  used to measure lengths at build time; no tokens are stored.
