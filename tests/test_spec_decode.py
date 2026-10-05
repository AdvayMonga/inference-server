"""N-gram speculative decoding: greedy output with speculation on equals off, token for token (tiny Qwen3-MoE, CPU)."""

from __future__ import annotations

import asyncio

import pytest
import torch

from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest, _ActiveRow, ngram_draft
from tests.stub_backend import StubBackend

# A repeating prompt, so prompt-lookup drafts exist from the first step.
REPEAT = [5, 17, 300, 42, 8, 99, 123] * 4 + [5, 17]
PLAIN = [5, 17, 300, 42, 8, 99, 600]


@pytest.fixture(scope="module")
def backend(tiny_qwen3_moe_dir):
    from inference_server.backends.custom_torch_backend import CustomTorchBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUSTOM_BACKEND_BLOCKS", "256")
        mp.setenv("CUSTOM_BACKEND_BLOCK_SIZE", "4")
        b = CustomTorchBackend(device="cpu", model_name=tiny_qwen3_moe_dir)
        b.load_model(tiny_qwen3_moe_dir)
    return b


def test_ngram_draft():
    assert ngram_draft([1, 2, 3, 4, 1, 2, 3], 2) == [4, 1]       # 3-gram [1,2,3] seen at 0
    assert ngram_draft([7, 9, 8, 9], 3) == [8, 9]                  # falls back to the 1-gram [9]
    assert ngram_draft([1, 2, 1, 3, 1], 1) == [3]                  # the most recent occurrence wins
    assert ngram_draft([1, 2, 3], 4) == [] and ngram_draft([1, 2, 1], 0) == []


async def _run(backend, prompts, max_tokens, spec_max_rows):
    sched = ContinuousBatchScheduler(backend, max_batch_size=8, spec_max_rows=spec_max_rows, spec_k=4)
    sched.start()
    try:
        loop = asyncio.get_running_loop()
        reqs = [ScheduledRequest(token_ids=list(p), max_tokens=max_tokens, session_id=f"s{i}",
                                 future=loop.create_future()) for i, p in enumerate(prompts)]
        outs = await asyncio.gather(*(sched.submit(r) for r in reqs))
    finally:
        await sched.stop()
    return outs, sched.stats()


@pytest.mark.parametrize("prompts", [[REPEAT], [PLAIN], [REPEAT, PLAIN]])
async def test_spec_matches_plain_greedy(backend, prompts):
    off, off_stats = await _run(backend, prompts, 24, 0)
    free = [p.free_count for p in backend.pools]       # prompts now prefix-cached, as on the spec run
    on, stats = await _run(backend, prompts, 24, 4)
    assert on == off
    assert off_stats["spec_drafted"] == 0 and stats["spec_drafted"] > 0
    assert stats["spec_drafted"] >= stats["spec_accepted"]
    if REPEAT in prompts:
        assert stats["spec_accepted"] > 0
    assert [p.free_count for p in backend.pools] == free   # no block leaked by rollback


async def test_spec_off_above_threshold(backend):
    """Two concurrent rows over a threshold of 1 decode batched until one finishes."""
    off, _ = await _run(backend, [REPEAT, PLAIN], 12, 0)
    on, _ = await _run(backend, [REPEAT, PLAIN], 12, 1)
    assert on == off


def _greedy_decode(backend, cache, tok, n):
    out = []
    for _ in range(n):
        nxt, _ = backend.decode_step_batched(torch.tensor([[tok]]), [cache], None,
                                             torch.tensor([[cache.seq_len]]))
        tok = int(nxt[0])
        out.append(tok)
    return out


def test_rejected_draft_leaves_kv_as_plain_decode(backend):
    """Partly-wrong draft: accepts the right prefix, rolls back the rest, and decoding continues identically."""
    ref, first, _ = backend.prefill(PLAIN)
    want = _greedy_decode(backend, ref, first, 8)
    blocks_ref = [list(bt) for bt in ref.block_tables]
    ref.free_all()

    cache, first, _ = backend.prefill(PLAIN)
    draft = want[:2] + [(want[2] + 1) % 900, want[3]]      # 2 right, then a wrong one
    out = backend.verify_greedy([cache], 0, [first] + draft)
    assert out == want[:3]
    assert cache.seq_len == len(PLAIN) + 3
    assert [len(bt) for bt in cache.block_tables] == [(len(PLAIN) + 3 + 3) // 4] * 2
    assert _greedy_decode(backend, cache, out[-1], 5) == want[3:8]
    assert [len(bt) for bt in cache.block_tables] == [len(b) for b in blocks_ref]
    cache.free_all()


def test_stop_inside_accepted_run_is_honoured():
    """EOS or max_tokens inside an accepted run ends the row there; nothing after it is emitted."""
    be = StubBackend()
    be.is_eos = lambda t: t == 99
    sched = ContinuousBatchScheduler(be)
    for accepted, max_tokens, want in [([5, 99, 7], 10, [5]), ([5, 6, 7], 2, [5, 6])]:
        req = ScheduledRequest(token_ids=[1], max_tokens=max_tokens, session_id="s", future=None)
        sched._active.append(_ActiveRow(request=req, current_token=8, real_kv_len=1, accepted=accepted))
        sched._batched_kv = [1]
        sched._evict_finished()
        assert req.generated == want and not sched._active


def test_decode_state_row_round_trip(backend):
    """The CUDA verify path's view of a BatchedDecodeState row: grow, roll back, free — block-exact."""
    from inference_server.models.paged_kv_cache import BatchedDecodeState, PagedKVCache
    free0 = [p.free_count for p in backend.pools]
    cache = PagedKVCache(backend.pools)
    with torch.no_grad():
        backend.model(torch.tensor([PLAIN]), kv_cache=cache)            # 7 tokens -> 2 blocks of 4
    state = BatchedDecodeState(backend.pools, torch.device("cpu"))
    state.add_row(cache)
    view = state.row_cache(0)
    kv = torch.zeros(1, 2, 6, 16, dtype=backend.pools[0].k.dtype)
    for L in range(2):
        view.append(L, kv, kv)                                            # 13 tokens -> 4 blocks
    state.set_row(0, view)
    assert (int(state.seq_lens[0]), int(state.n_alloc[0])) == (13, 4)
    view = state.row_cache(0)
    view.truncate(9)                                                      # 9 tokens -> 3 blocks
    state.set_row(0, view)
    assert (int(state.seq_lens[0]), int(state.n_alloc[0])) == (9, 3)
    assert [free0[i] - p.free_count for i, p in enumerate(backend.pools)] == [3, 3]
    state.remove_row(0)
    assert [p.free_count for p in backend.pools] == free0
