"""The draft model's drafter (drafter.py): its proposals are the draft model's own greedy continuation, through any
mix of accepted, rejected and plain-decoded tokens, and after another request displaced its context in the shared
cache; the scheduler keeps a drafter's context equal to the request's (prompt + generated, no token twice)."""

import pytest
import torch

from inference_server.drafter import DraftModel, ModelDrafter


@pytest.fixture(scope="module")
def draft():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    c = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
    return DraftModel(Qwen3ForCausalLM(c), max_len=256)


def greedy(model, ids, k):
    out, ids = [], list(ids)
    with torch.no_grad():
        for _ in range(k):
            t = int(model(torch.tensor([ids])).logits[0, -1].argmax())
            out.append(t)
            ids.append(t)
    return out


def test_proposals_are_the_draft_models_greedy_continuation_through_accept_reject_and_displacement(draft):
    ids = [3, 17, 5, 9, 41, 2, 8]
    d = ModelDrafter(draft, ids)
    ctx = list(ids)
    g = torch.Generator().manual_seed(1)
    for rnd in range(8):
        p = d.propose(4)
        assert p == greedy(draft.model, ctx, 4), rnd
        # the target keeps a prefix of the drafts and adds its own token (sometimes different from the next draft)
        keep = int(torch.randint(0, 4, (1,), generator=g))
        own = p[keep] if rnd % 2 else (p[keep] + 1) % 64
        new = p[:keep] + [own]
        d.observe(new)
        ctx += new
        if rnd == 4:                                   # another request uses the shared cache in between
            other = ModelDrafter(draft, [7, 7, 1])
            assert other.propose(3) == greedy(draft.model, [7, 7, 1], 3)
        if rnd == 6:                                   # tokens decoded without speculation
            d.observe([11, 12])
            ctx += [11, 12]
    assert d.ids == ctx


def test_the_scheduler_gives_a_drafter_the_requests_context_once():
    import asyncio
    from inference_server import scheduler as S
    from inference_server.sampling import SamplingParams
    from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
    from tests.test_spec_decode import PeriodicBackend

    seen = []

    class B(PeriodicBackend):
        def new_drafter(self, ids):
            seen.append(list(ids))
            return S.NgramDrafter(ids)

    async def run():
        sched = ContinuousBatchScheduler(B(), max_batch_size=4, prefill_mode="mixed", prefill_step_tokens=256,
                                         async_decode=True)
        sched.start()
        try:
            loop = asyncio.get_running_loop()
            req = ScheduledRequest(token_ids=list(range(1000, 1010)), max_tokens=20, session_id="s",
                                   future=loop.create_future(), sampling=SamplingParams(temperature=0.0))
            out = await sched.submit(req)
            return req, out
        finally:
            await sched.stop()

    old = S.SPEC_K
    S.SPEC_K = 3
    try:
        req, out = asyncio.run(run())
    finally:
        S.SPEC_K = old
    assert len(seen) == 1
    ctx = seen[0]
    assert ctx[:10] == list(range(1000, 1010)) and ctx[10:] == out[:len(ctx) - 10]   # prompt, then generated, once
    assert req.drafter.ids == list(range(1000, 1010)) + out
