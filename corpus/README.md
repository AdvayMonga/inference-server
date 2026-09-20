# corpus/ — frozen workload traces

The corpus defines the landscape the research loop searches. Anything not in it is invisible.

**A trace** (`<class>/<split>.jsonl`) is one request per line: `arrival_s` (offset from trace
start), `session_id`, `turn_index`, `prompt`, `max_tokens`, `sampling` (temperature 0, so output
is deterministic and the correctness gate can compare it), and `expected_output_hash`, which is
`null` until a reference run fills it. `replay_trace.py` fires each request at its `arrival_s` on
the client's own clock regardless of what the server is doing — open loop, never closed.

**A class** (`manifest.json` → `classes`) is a name, an SLO (`slo_ttft_ms`, optional
`slo_tpot_ms`), a nominal arrival rate, and two traces: `seen`, which the loop optimises against,
and `heldout`, on which a confirmed win must replicate before merge. Every full prompt string and
session id is disjoint between the splits, but both draw base content from the same small
`prompt_bank.py` (long_context has two base documents), so what differs today is the unique
per-request preamble plus the arrival schedule and lengths. Expanding the bank is the real fix,
and is a new corpus version.

`replay_trace.py` posts each prompt to `/v1/chat/completions` as a single user message, so the
model's own chat template is applied **server-side** (`--prompt-format raw` restores the old
`/v1/completions` path). The traces store plain prompt text and stay model-independent: baking
Gemma's template into the corpus would invalidate every trace the moment Phase 0 picks a
different model. It is not a cosmetic choice — sent raw, `gemma-4-E2B-it` answers most of these
prompts with an immediate end-of-turn token, which is why `cold_start/seen` generated 220 tokens
of its 626-token budget and six of eight `cold_start/heldout` prompts generated nothing at all
(`kb-20260919-6ef4e6bf`). What templating costs is a variable that is not in the trace — which
template, applied with which options — so a chat-route panel carries a verified `chat_template`
fingerprint in its validity block, and `compare.py` refuses across a changed one, across
`harness_config.prompt_format`, and a noise band no longer gates a run from the other route.

Correlation is unchanged on either route: `session_id` / `turn_index` ride as the `X-Session-Id`
/ `X-Turn-Index` headers, so a turn 1 reaches the engine on the same session as its turn 0, and
`X-Trace-Id=<class>-<split>-<nonce>-<index>` is the key its per-request CSV row shares with the
engine's telemetry row; the nonce is minted per invocation (two replays against one server
process write into one telemetry file) and recorded in the panel's `harness_config.trace_prefix`,
so a run's rows are joined by prefix. `sampling` is sent as-is (temperature, top_p, top_k) —
`top_k` is not an OpenAI-standard field and the chat route carries it exactly as the raw one does.

A turn's prompt still carries the full conversation so far, as ONE user message, and that stays
true under templating: the engine has no conversation store, prefix sharing is what makes the
repeated prefix cheap, and an honest two-message rendering would need the assistant's reply from
turn 0 — which the corpus does not record and could not record without tying itself to one
model's outputs.

| class | shape | placeholder SLO |
|---|---|---|
| `cold_start` | 8 sparse requests to a fresh replica | p95 TTFT < 2000 ms |
| `steady_interactive` | ~130 mixed-length requests at 4 req/s, 30% of sessions return for a turn 1 | p95 TTFT < 200 ms, TPOT < 50 ms |
| `long_context` | 40 long prompts at 1 req/s, high KV pressure | p95 TTFT < 1000 ms |

SLO numbers are placeholders pending the plan's Phase 0 decision (recorded in `manifest.notes`).

## The rule

`corpus_version` is a sha256 over every trace's hash and the class table (SLOs, rates, paths).
`load_manifest()` re-hashes each file and refuses the corpus on any mismatch, and `compare.py`
refuses two panels whose versions differ. So: **never edit a trace or an SLO in place.** Any
change — a prompt, an arrival, a new class, the placeholder SLOs being decided — is a rebuild that
produces a new version, and every panel measured against the old version stops being comparable
to the new ones. That is the point.

## Adding or changing a class

1. Edit the class table in `scripts/tools/build_corpus.py` (or `prompt_bank.py` for prompts).
2. `PYTHONPATH=src python scripts/tools/build_corpus.py` — regenerates every trace and the manifest
   from the fixed seed.
3. Commit `corpus/` with the code change. `tests/test_research_corpus.py` asserts the committed
   data is what the builder produces and that hashes verify.
