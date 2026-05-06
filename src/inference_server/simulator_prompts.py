"""Prompt bank for the load simulator. Three buckets: short, medium, long.

Bucket sizes are roughly calibrated to Gemma's tokenizer:
  short:  ~10–30 tokens
  medium: ~150–500 tokens
  long:   ~1200–2200 tokens (multi-chunk territory at PREFILL_CHUNK_SIZE=256)
"""

SHORT_PROMPTS: list[str] = [
    "Explain how a computer works in simple terms",
    "What is the capital of France?",
    "Write a short poem about the ocean",
    "Summarize the theory of relativity in three sentences",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis",
    "What happened during the French Revolution?",
    "Explain recursion to a five year old",
    "Name three primary colors and what they mix to make",
    "What is the speed of light?",
    "Give a one-line definition of machine learning",
    "Why is the sky blue?",
]

MEDIUM_PROMPTS: list[str] = [
    (
        "I'm trying to decide between two database options for a new web service. "
        "The service expects roughly 10,000 reads per second and 500 writes per second, "
        "with mostly small payloads (under 4 KB) and occasional larger blobs (up to 2 MB). "
        "Latency budget is 20ms p99 for reads. Walk me through the tradeoffs between "
        "PostgreSQL and a key-value store like Redis or DynamoDB for this workload. "
        "Be specific about consistency, durability, indexing, and operational complexity."
    ),
    (
        "Review the following code snippet and tell me if there are any concurrency bugs. "
        "It's a Python function that increments a shared counter from multiple threads:\n\n"
        "counter = 0\n"
        "def worker():\n"
        "    global counter\n"
        "    for _ in range(1000):\n"
        "        counter += 1\n\n"
        "threads = [threading.Thread(target=worker) for _ in range(10)]\n"
        "for t in threads: t.start()\n"
        "for t in threads: t.join()\n"
        "print(counter)\n\n"
        "Explain what counter will end up being, why, and how to fix it without "
        "introducing significant lock contention. Discuss at least two approaches."
    ),
    (
        "Explain the bias-variance tradeoff in machine learning. Cover what each term means, "
        "why minimizing one tends to increase the other, how the tradeoff manifests differently "
        "in linear regression versus deep neural networks, and what practical techniques "
        "(regularization, ensembling, early stopping, more data) shift the tradeoff in each "
        "direction. Use concrete examples where helpful and avoid hand-waving."
    ),
    (
        "I'm preparing a 30-minute talk for a junior engineering audience on the topic of "
        "'why distributed systems are hard.' I want to cover: (1) the difference between "
        "fail-stop and Byzantine failures, (2) why the CAP theorem is more nuanced than "
        "'pick two,' (3) what split-brain is and why it matters, and (4) one concrete "
        "real-world incident that illustrates the points. Give me an outline with talking "
        "points for each section, plus a memorable closing line."
    ),
    (
        "Write a short technical blog post (about 400 words) explaining how a CPU cache works. "
        "Cover the L1/L2/L3 hierarchy, cache lines, the difference between temporal and spatial "
        "locality, and why false sharing can wreck performance in multi-threaded code. "
        "Aim it at a working software engineer who has never thought about caches before."
    ),
]

# Long prompts (~1500–2200 tokens). Calibrated to clear PREFILL_CHUNK_SIZE=256
# multiple times so chunked prefill is actually exercised.
_LONG_DOC_1 = """Below is a transcript of an internal architecture review meeting. \
Read it carefully and produce a written summary suitable for a stakeholder who could not \
attend. Highlight the key decisions made, the open questions, and the action items with \
owners. Note any places where the discussion seems to have papered over a real disagreement.

--- TRANSCRIPT BEGINS ---

ALICE: Okay, let's get started. The agenda is the new ingestion pipeline. Bob's team has \
been running into scaling issues with the current Kafka-based fanout, and we need to \
decide whether to keep extending it or do something more invasive. Bob, you want to kick \
off?

BOB: Sure. So the short version is, we're at about 80,000 events per second across the \
fleet, and we're seeing consumer lag spike anywhere from 30 seconds to two minutes during \
the daily peak. The lag itself isn't the end of the world, but it's correlated with \
downstream timeouts in the enrichment service, which is what's actually paging people. \
We've tuned the consumer group settings, we've added more partitions, we've scaled up the \
enrichment workers, and we keep buying ourselves a few weeks before it gets bad again. \
The fundamental issue is that our enrichment work is heavy — we're hitting three external \
APIs per event, two of which have their own rate limits — and the consumer model means \
slow events block the partition.

CAROL: Can you say more about the rate limits? I thought we negotiated higher quotas with \
the vendor last quarter.

BOB: We did, but the new quotas are per-tenant, not per-source, so they don't help us in \
the way I expected. We're still bottlenecked because all our traffic looks like one tenant \
to them.

DAVE: What's the proposal then?

BOB: The strawman is to decouple ingestion from enrichment. We keep Kafka for the \
firehose, but instead of having enrichment run inside the consumer, we drop the events \
into a queue per enrichment provider, with its own concurrency control, and have a small \
service that fans out the API calls. The enrichment results come back into a second \
Kafka topic which downstream services consume. The win is that a slow API call only \
backs up its own queue, not the entire pipeline.

ALICE: So you're proposing two queues plus a new service in between.

BOB: Yes.

ALICE: That's a lot of new infrastructure for a problem we've been managing.

BOB: It is, but the alternative is that every quarter we burn another sprint scaling the \
current setup, and the headroom keeps shrinking. We've also had two near-misses where the \
enrichment service ran out of file descriptors because of a slow API, and the only thing \
that saved us was the on-call manually killing pods.

DAVE: What's the migration plan? We can't take this offline.

BOB: We'd run them in parallel for two to three weeks. The new pipeline reads the same \
input topic, writes to a different output topic, and we have downstream consumers \
double-read and verify they match. Once we're confident, we cut over and decommission the \
old enrichment workers.

CAROL: I'm worried about the queue-per-API design. We have, what, eight enrichment APIs \
now? That's eight queues, eight sets of monitoring, eight runbooks. The operational \
surface gets a lot bigger.

BOB: True. But each queue is dead simple — it's a thin wrapper around a Redis stream with \
a few workers. The complexity isn't in the queue, it's in the API client behavior, which \
we already have today, just buried inside the consumer.

ALICE: How sure are you that the API calls are the real bottleneck? Have we profiled?

BOB: We have. Median enrichment time is 180ms, p99 is 4.2 seconds, and the long tail is \
almost entirely waiting on one specific vendor that has unpredictable latency. Decoupling \
that vendor's calls into their own queue means a single bad minute on their side doesn't \
back up the whole pipeline.

DAVE: What about ordering guarantees? Some downstream consumers care about per-key order.

BOB: We preserve per-key ordering by routing all events for a given key to the same queue \
worker. Cross-key ordering was never guaranteed.

DAVE: Is that documented?

BOB: It's documented in the design doc but not in the consumer-facing API spec. I'll add \
it.

ALICE: Action item. Let's also schedule a review with the consumer teams to confirm \
nobody is depending on cross-key order in a way we don't know about.

CAROL: I want to revisit the build-vs-buy question. There are managed services that do \
this fan-out pattern. Have we evaluated any?

BOB: We looked at AWS Step Functions and Temporal. Step Functions doesn't scale to our \
event volume cost-effectively. Temporal is interesting but it's a much bigger lift to \
adopt — we'd need to rewrite the enrichment logic as Temporal workflows, and the team has \
no Temporal experience.

ALICE: Could we adopt Temporal incrementally for this use case and learn it gradually?

BOB: Possibly, but I don't think this is the right first project for it. The pattern we \
need is fan-out with rate-limited workers, which is closer to a queue pattern than a \
workflow pattern. Temporal would be overkill.

DAVE: What's the expected timeline for the proposed approach?

BOB: Six weeks. Two weeks to build the fan-out service and the queue infrastructure, two \
weeks to run in parallel and validate, two weeks to migrate consumers and tear down the \
old path.

ALICE: That feels optimistic. What's the realistic worst case?

BOB: Ten weeks if we hit a serious issue with one of the vendor APIs during validation.

ALICE: Okay. Let's frame it as eight weeks externally. Bob, can you put together a more \
detailed design doc by end of next week, with specific call-outs for the migration risks \
and the operational handoff?

BOB: Yes.

CAROL: I'd like to be a reviewer on the design doc. I want to make sure the operational \
story is solid before we commit.

ALICE: Done. Anyone else?

DAVE: I'll review.

ALICE: Great. So action items: Bob owns the design doc, due next Friday. Carol and Dave \
are reviewers. Bob also documents the per-key ordering guarantee in the API spec. I'll \
schedule the consumer team review for the week after. Anything else?

BOB: One more thing — we'll need a new Redis cluster for the queues. I'll work with \
infra on capacity planning.

ALICE: Good. Add that to the doc. Let's wrap.

--- TRANSCRIPT ENDS ---

Now produce the written summary."""

_LONG_DOC_2 = """You are reviewing the following pull request and will write a thorough \
code review comment. Be specific about correctness issues, edge cases, performance \
concerns, and style. The PR is described as 'add basic rate limiting to the public API \
endpoint.'

PR diff:

```python
import time
from collections import defaultdict
from threading import Lock
from fastapi import HTTPException, Request

# Module-level state holding rate limit counters per IP
_request_counts = defaultdict(list)
_lock = Lock()

# Configuration
MAX_REQUESTS_PER_WINDOW = 100
WINDOW_SECONDS = 60


def rate_limit_middleware(request: Request):
    \"\"\"Rate limit by client IP. Raises 429 if over limit.\"\"\"
    client_ip = request.client.host
    now = time.time()

    with _lock:
        # Drop entries older than the window
        _request_counts[client_ip] = [
            t for t in _request_counts[client_ip]
            if now - t < WINDOW_SECONDS
        ]

        if len(_request_counts[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )

        _request_counts[client_ip].append(now)


# Wired into the API endpoint
@app.post("/api/v1/process")
async def process_endpoint(request: Request, body: ProcessRequest):
    rate_limit_middleware(request)
    result = await do_work(body)
    return result


# Test added in test_rate_limit.py
def test_rate_limit_blocks_after_threshold():
    for _ in range(MAX_REQUESTS_PER_WINDOW):
        rate_limit_middleware(make_fake_request("1.2.3.4"))
    with pytest.raises(HTTPException) as exc:
        rate_limit_middleware(make_fake_request("1.2.3.4"))
    assert exc.value.status_code == 429
```

The commit message says: "Adds rate limiting at 100 requests per minute per IP. \
Threadsafe via a module-level lock. Includes a unit test." The PR is from a junior \
engineer who explicitly asked for thorough feedback because they want to learn.

Things to consider as you review:

1. Correctness: does this actually rate-limit the way the description claims? Are there \
race conditions? Does it behave correctly across multiple worker processes?

2. Memory: the _request_counts dict grows over time. Under what conditions could this \
become a leak? How big could it get?

3. Performance: what's the cost per request? What happens at high QPS? Does the lock \
become a bottleneck?

4. API design: is rate_limit_middleware actually a middleware in the FastAPI sense, or \
just a function being called manually? What's the idiomatic FastAPI way to do this?

5. Edge cases: what happens behind a load balancer or proxy where request.client.host is \
the proxy's IP instead of the real client? What about IPv6?

6. Security: can this be bypassed? Are there obvious attacks (e.g., one bad actor \
causing memory pressure that affects rate limits for others)?

7. Test quality: does the included test actually validate the behavior the PR claims to \
provide, or does it just exercise the trivial case?

8. Production-readiness: is this appropriate for a real public API endpoint, or is it \
'good enough for now' with caveats? If caveats, what should be in a follow-up?

Write a code review comment that is direct, specific, and constructive. Treat the \
junior engineer as someone who can handle blunt technical feedback if it's substantive \
and respectful. Do not pad with generic encouragement. Reference specific lines where \
relevant. End with a clear recommendation (approve / request changes / block) and a \
prioritized list of what must change versus what would be nice to fix in a follow-up."""

LONG_PROMPTS: list[str] = [_LONG_DOC_1, _LONG_DOC_2]


# Weighted mix used by the simulator. Tweak weights to skew traffic.
PROMPT_MIX = [
    ("short", SHORT_PROMPTS, 0.60),
    ("medium", MEDIUM_PROMPTS, 0.30),
    ("long", LONG_PROMPTS, 0.10),
]

# Per-bucket max_tokens distribution: (lo, hi) drawn uniformly per request.
MAX_TOKENS_RANGE = {
    "short": (16, 80),
    "medium": (60, 250),
    "long": (100, 500),
}
