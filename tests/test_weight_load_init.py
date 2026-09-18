"""from_hf must initialise nothing and leave nothing unwritten.

Two ways the same optimisation goes wrong, both of which happened here:

  * a construction OUTSIDE the skip window still pays Kaiming-uniform for a tensor the next
    line discards (lm_head: 403M elements for E2B, ~2s), which is the waste this exists to
    remove and is invisible to any check made after the load;
  * a parameter that `load_hf_weights` stops writing ships uninitialised memory, which is
    silent garbage rather than a crash.

The poison check walks `named_parameters(remove_duplicate=False)` on purpose. After the tie,
`lm_head.weight` IS `embed_tokens.weight` — the same object — so the deduplicated view drops
it, and the first version of this test missed the lm_head hole for exactly that reason.
"""

from __future__ import annotations

import threading

import pytest
import torch
import torch.nn as nn

from inference_server.models import gemma4
from inference_server.models.gemma4 import GemmaForCausalLM, GemmaModel

MODEL_NAME = "google/gemma-4-E2B-it"


# --------------------------------------------------------------------------- the patch itself

def test_restores_on_normal_exit():
    real = nn.Linear.reset_parameters
    with gemma4._skip_param_init():
        assert nn.Linear.reset_parameters is not real
    assert nn.Linear.reset_parameters is real
    assert nn.Embedding.reset_parameters is not None


def test_restores_when_the_body_raises():
    real = nn.Linear.reset_parameters
    with pytest.raises(RuntimeError):
        with gemma4._skip_param_init():
            raise RuntimeError("load failed")
    assert nn.Linear.reset_parameters is real


def test_concurrent_callers_are_serialised():
    """The failure the lock buys off: a second thread saving the first thread's no-op as if it
    were the real initialiser, and restoring THAT — after which nn.Linear never initialises
    again, process-wide, with no error anywhere.

    There is deliberately no nesting test. from_hf's two windows are sequential, and the lock is
    a plain Lock, so same-thread nesting deadlocks by design — loudly, at the call site — rather
    than silently holding the process-wide patch open across unrelated work.
    """
    real = nn.Linear.reset_parameters
    holder_inside, follower_inside, release = (threading.Event() for _ in range(3))

    def holder():
        with gemma4._skip_param_init():
            holder_inside.set()
            release.wait(5)

    def follower():
        holder_inside.wait(5)
        with gemma4._skip_param_init():
            follower_inside.set()

    threads = [threading.Thread(target=holder), threading.Thread(target=follower)]
    for t in threads:
        t.start()
    # If the two windows were allowed to overlap, the follower is inside by now. It must not be.
    entered_concurrently = follower_inside.wait(0.2)
    release.set()
    for t in threads:
        t.join(5)
        assert not t.is_alive()

    assert not entered_concurrently, "the second caller entered while the first held the patch"
    assert nn.Linear.reset_parameters is real


# --------------------------------------------------------------------------- the real load

@pytest.mark.heavy
def test_from_hf_initialises_nothing_and_writes_every_parameter():
    """No random fill anywhere in from_hf, and no parameter survives the load unwritten."""
    initialised: list[str] = []
    in_transformers: list[bool] = []

    def recorder(cls):
        real = cls.reset_parameters

        def reset(self):
            # transformers builds its own model between our two windows; its init is not ours
            # to skip, so it is not evidence of waste on our side.
            if not in_transformers:
                initialised.append(f"{type(self).__name__}{tuple(self.weight.shape)}")
            return real(self)

        return real, reset

    from transformers import AutoModelForCausalLM

    real_from_pretrained = AutoModelForCausalLM.from_pretrained

    def marked_from_pretrained(*a, **kw):
        in_transformers.append(True)
        try:
            return real_from_pretrained(*a, **kw)
        finally:
            in_transformers.pop()

    real_load = GemmaModel.load_hf_weights

    def poisoning_load(self, sd):
        with torch.no_grad():
            for _, p in self.named_parameters(remove_duplicate=False):
                p.fill_(float("nan"))
        return real_load(self, sd)

    saved = {}
    for cls in (nn.Linear, nn.Embedding):
        real, reset = recorder(cls)
        saved[cls] = real
        cls.reset_parameters = reset
    AutoModelForCausalLM.from_pretrained = marked_from_pretrained
    GemmaModel.load_hf_weights = poisoning_load
    try:
        model = GemmaForCausalLM.from_hf(MODEL_NAME)
    finally:
        for cls, real in saved.items():
            cls.reset_parameters = real
        AutoModelForCausalLM.from_pretrained = real_from_pretrained
        GemmaModel.load_hf_weights = real_load

    assert initialised == [], (
        f"from_hf randomly initialised {len(initialised)} module(s) it then overwrote: "
        f"{initialised[:5]} — every construction of ours must sit inside _skip_param_init()")

    poisoned = [n for n, p in model.named_parameters(remove_duplicate=False)
                if torch.isnan(p).any()]
    assert not poisoned, f"load_hf_weights never wrote {len(poisoned)} parameter(s): {poisoned[:5]}"

    assert model.lm_head.weight is model.model.embed_tokens.embed_tokens.weight, \
        "lm_head is no longer tied to the embedding; the poison check above stops covering it"
