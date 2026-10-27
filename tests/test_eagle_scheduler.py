"""The scheduler's EAGLE loop (backend.eagle_on) on a fake backend whose head drafts the target's tokens except every
fourth: the streamed output equals the plain decode's, more than one token lands a step, EOS, max_tokens and
ignore_eos cut exactly, the host's position is never more than one step behind the device's, a row without its
prompt's aux and a sampled row decode plainly."""

import pytest
import torch

from tests.test_spec_decode import PeriodicBackend, _run
from tests.test_chunked_prefill import _FakeKV

K = 3


class _Ev:
    def synchronize(self):
        pass


class EagleBackend(PeriodicBackend):
    def __init__(self, eos_at=None, aux=True):
        super().__init__(eos_at)
        self.aux, self.begins, self.launches, self.lag = aux, 0, 0, []

    def prefill_chunk_batch(self, items, defer_tokens=False):
        out = []
        for kv, chunk, final, _ in items:
            p0 = kv.seq_len if kv is not None else 0
            n = p0 + len(chunk)
            new = _FakeKV(1, n)
            new.eagle_aux = (getattr(kv, "eagle_aux", []) + [(p0, torch.zeros(len(chunk), 3))]) if self.aux else None
            out.append((new, self._tok(n) if final else None, n))
        return out

    def eagle_on(self):
        return True

    def _drafts(self, p0):
        return [0 if (p0 + 1 + i) % 4 == 0 else self._tok(p0 + 1 + i) for i in range(K)]

    def eagle_begin(self, state, r, prompt, first, aux):
        if sum(rows.shape[0] for _, rows in aux) != len(prompt):
            return False
        self.begins += 1
        self.pos0 = len(prompt)
        self.cur = [first] + self._drafts(self.pos0)
        return True

    def eagle_launch(self, state, r, pos0):
        self.launches += 1
        self.lag.append(self.pos0 - pos0)
        greedy = [self._tok(self.pos0 + 1 + j) for j in range(K + 1)]
        a = 0
        while a < K and greedy[a] == self.cur[a + 1]:
            a += 1
        self.pos0 += a + 1
        self.cur = [greedy[a]] + self._drafts(self.pos0)
        return torch.tensor(greedy + [a + 1]), _Ev()

    def eagle_end(self, state, r, pos0):
        assert pos0 == self.pos0
        state.seq_len = pos0


@pytest.mark.asyncio
async def test_eagle_output_equals_plain_decode_with_several_tokens_a_step():
    plain, _ = await _run(PeriodicBackend())
    b = EagleBackend()
    out, st = await _run(b)
    assert out == plain and len(out) == 40
    assert b.begins == 1 and st["spec_steps"] > 0
    assert st["spec_tokens"] / st["spec_steps"] > 2.0
    assert max(b.lag) <= K + 1 and min(b.lag) >= 0          # the host is at most one step behind


@pytest.mark.asyncio
async def test_eagle_stops_at_eos_and_max_tokens_exactly():
    ref, _ = await _run(PeriodicBackend(eos_at=27), max_tokens=100)
    out, _ = await _run(EagleBackend(eos_at=27), max_tokens=100)
    assert out == ref and 999 not in out
    for m in (1, 2, 7):
        capped, _ = await _run(EagleBackend(), max_tokens=m)
        assert capped == (await _run(PeriodicBackend(), max_tokens=m))[0]
    kept, _ = await _run(EagleBackend(eos_at=27), max_tokens=100, ignore_eos=True)
    assert len(kept) == 100 and 999 in kept


@pytest.mark.asyncio
async def test_no_aux_or_sampling_decodes_plainly():
    b = EagleBackend(aux=False)
    out, st = await _run(b)
    assert b.begins == 0 and b.launches == 0 and len(out) == 40
    b = EagleBackend()
    out, st = await _run(b, temperature=0.7)
    assert b.launches == 0 and len(out) == 40
