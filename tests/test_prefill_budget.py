"""Batched admission caps a wave at max_prefill_tokens of right-padded tokens (rows x longest prompt)."""

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from tests.stub_backend import StubBackend


class WaveBackend(StubBackend):
    def __init__(self):
        super().__init__()
        self.waves: list[list[int]] = []
        self.last_batch_cache_hit_tokens = None

    def prefill_batch(self, prompts, session_ids=None):
        self.waves.append([len(p) for p in prompts])
        return [(len(p), 1, len(p)) for p in prompts]


class _Future:
    def done(self): return True
    def set_exception(self, e): pass


def _sched(budget):
    b = WaveBackend()
    s = ContinuousBatchScheduler(b, max_batch_size=8, prefill_mode="batched", max_prefill_tokens=budget)
    for n in (60, 60, 60, 20):
        s.enqueue(ScheduledRequest(token_ids=[1] * n, max_tokens=4, session_id="s", future=_Future()))
    return s, b


def test_a_wave_stops_before_its_padded_tokens_exceed_the_budget():
    s, b = _sched(130)
    s._admit_pending("cpu")
    assert b.waves == [[60, 60]]                      # 3 x 60 = 180 > 130
    s._admit_pending("cpu")
    assert b.waves == [[60, 60], [60, 20]]            # padded 2 x 60 = 120 fits
    assert s.stats()["max_prefill_tokens"] == 130


def test_one_prompt_always_goes_and_no_cap_means_one_wave():
    s, b = _sched(50)                                 # smaller than any prompt
    s._admit_pending("cpu"); s._admit_pending("cpu")
    assert b.waves == [[60], [60]]
    s, b = _sched(0)
    s._admit_pending("cpu")
    assert b.waves == [[60, 60, 60, 20]]


def test_step_is_one_iteration_of_the_loop():
    """`step()` does what one `_run` iteration did: admit, prefill, evict, decode; the loop and a simulator share it."""
    s, b = _sched(1000)
    s.step()                                          # admits all four as one wave, decodes nothing yet
    assert b.waves == [[60, 60, 60, 20]] and len(s._active) == 4 and s._pending_count == 0
    s.step()                                          # a decode step: every row advanced one token
    assert all(len(r.request.generated) >= 1 for r in s._active)
    assert s.stats()["active_high_water"] == 4
