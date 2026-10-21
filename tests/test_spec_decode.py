"""Speculative decoding by prompt lookup (scheduler.SPEC_K): the output equals the plain decode's, more than one token
lands per verify step on repetitive text, EOS and max_tokens cut the accepted run exactly, a sampled row gets none."""

import asyncio

import pytest

from inference_server import scheduler as S
from inference_server.sampling import SamplingParams
from inference_server.scheduler import ContinuousBatchScheduler, NgramDrafter, ScheduledRequest
from tests.test_chunked_prefill import _FakeKV
from tests.test_mixed_prefill import FakeMixedBackend

SEQ = [300, 301, 302, 303, 304, 305]            # the fake model's answer: this, repeated


class PeriodicBackend(FakeMixedBackend):
    """Next token = SEQ[position % 6], whatever the input: the text repeats, so prompt lookup finds it."""

    def __init__(self, eos_at=None):
        super().__init__()
        self.eos_at = eos_at                      # a position whose token is EOS
        self.verify_calls = 0

    def _tok(self, p):
        return 999 if p == self.eos_at else SEQ[p % len(SEQ)]

    def prefill_chunk_batch(self, items, defer_tokens=False):
        out = []
        for kv, chunk, final, _ in items:
            n = (kv.seq_len if kv is not None else 0) + len(chunk)
            out.append((_FakeKV(1, n), self._tok(n) if final else None, n))
        return out

    def decode_step_batched(self, current_tokens, batched_kv, attention_mask, position_ids, sampling_per_row=None):
        import torch
        batched_kv.seq_len += 1
        return torch.tensor([self._tok(batched_kv.seq_len)] * current_tokens.shape[0]), batched_kv

    def verify_step(self, batched_kv, r, tokens):
        self.verify_calls += 1
        p0 = batched_kv.seq_len
        return [self._tok(p0 + 1 + i) for i in range(len(tokens))]

    def commit_verify(self, batched_kv, r, accepted):
        batched_kv.seq_len += accepted

    def is_eos(self, t):
        return t == 999


async def _run(backend, max_tokens=40, temperature=0.0):
    sched = ContinuousBatchScheduler(backend, max_batch_size=4, prefill_mode="mixed", prefill_step_tokens=256,
                                     async_decode=True)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        req = ScheduledRequest(token_ids=list(range(1000, 1010)), max_tokens=max_tokens, session_id="s",
                               future=loop.create_future(), token_queue=asyncio.Queue(),
                               sampling=SamplingParams(temperature=temperature))
        sched.enqueue(req)
        streamed = []
        while (t := await asyncio.wait_for(req.token_queue.get(), 10)) is not None:
            streamed.append(t)
        out = await req.future
        assert streamed == out
        return out, sched.stats()
    finally:
        await sched.stop()


def test_ngram_drafter_proposes_what_followed_the_last_match():
    d = NgramDrafter([1, 2, 3, 4, 5, 1, 2, 3, 9, 1, 2])
    assert d.propose(3) == [3, 9, 1]              # after the last "1 2" (3-gram "9 1 2" is unseen)
    d.observe([3])
    assert d.propose(2) == [9, 1]                 # "1 2 3" last led to 9
    assert NgramDrafter([7]).propose(3) == []


@pytest.mark.asyncio
async def test_spec_output_equals_plain_decode_with_fewer_steps(monkeypatch):
    plain, _ = await _run(PeriodicBackend())
    monkeypatch.setattr(S, "SPEC_K", 3)
    b = PeriodicBackend()
    spec, st = await _run(b)
    assert spec == plain and len(spec) == 40
    assert st["spec_steps"] == b.verify_calls > 0
    assert st["spec_tokens"] > st["spec_steps"]  # drafts accepted: more than one token a step
    assert st["spec_tokens"] / st["spec_steps"] > 2.0


@pytest.mark.asyncio
async def test_spec_stops_at_eos_and_max_tokens_exactly(monkeypatch):
    monkeypatch.setattr(S, "SPEC_K", 3)
    plain, _ = await _run(PeriodicBackend(eos_at=27), max_tokens=100)
    assert 999 not in plain and len(plain) < 100
    monkeypatch.setattr(S, "SPEC_K", 0)
    ref, _ = await _run(PeriodicBackend(eos_at=27), max_tokens=100)
    assert plain == ref
    monkeypatch.setattr(S, "SPEC_K", 3)
    capped, _ = await _run(PeriodicBackend(), max_tokens=7)
    assert len(capped) == 7


@pytest.mark.asyncio
async def test_sampled_rows_are_not_speculated(monkeypatch):
    monkeypatch.setattr(S, "SPEC_K", 3)
    b = PeriodicBackend()
    out, st = await _run(b, temperature=0.7)
    assert b.verify_calls == 0 and st["spec_steps"] == 0 and len(out) == 40
