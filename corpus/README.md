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

`replay_trace.py` posts to `/v1/completions` with `session_id` / `turn_index` as the
`X-Session-Id` / `X-Turn-Index` headers, so a turn 1 reaches the engine on the same session as
its turn 0, and `X-Trace-Id=<class>-<split>-<index>`, which is the key its per-request CSV row
shares with the engine's telemetry row. `sampling` is sent as-is (temperature, top_p, top_k).
A turn's prompt still carries the full conversation so far: the engine has no conversation
store, and prefix sharing is what makes the repeated prefix cheap.

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
