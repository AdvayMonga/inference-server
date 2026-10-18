"""The loader must initialise nothing and leave nothing unwritten.

Two ways the same optimisation goes wrong:

  * a construction OUTSIDE the skip window still pays Kaiming-uniform for a tensor the next
    line discards, which is the waste this exists to remove and is invisible to any check made
    after the load;
  * a parameter the loader stops writing ships uninitialised memory, which is silent garbage
    rather than a crash.
"""

from __future__ import annotations

import threading

import pytest
import torch
import torch.nn as nn

from inference_server.models import common
from inference_server.models.qwen3_moe import Qwen3MoeForCausalLM


# --------------------------------------------------------------------------- the patch itself

def test_restores_on_normal_exit():
    real = nn.Linear.reset_parameters
    with common._skip_param_init():
        assert nn.Linear.reset_parameters is not real
    assert nn.Linear.reset_parameters is real
    assert nn.Embedding.reset_parameters is not None


def test_restores_when_the_body_raises():
    real = nn.Linear.reset_parameters
    with pytest.raises(RuntimeError):
        with common._skip_param_init():
            raise RuntimeError("load failed")
    assert nn.Linear.reset_parameters is real


def test_concurrent_callers_are_serialised():
    """The failure the lock buys off: a second thread saving the first thread's no-op as if it
    were the real initialiser, and restoring THAT — after which nn.Linear never initialises
    again, process-wide, with no error anywhere.

    There is deliberately no nesting test. The lock is a plain Lock, so same-thread nesting
    deadlocks by design — loudly, at the call site — rather than silently holding the
    process-wide patch open across unrelated work.
    """
    real = nn.Linear.reset_parameters
    holder_inside, follower_inside, release = (threading.Event() for _ in range(3))

    def holder():
        with common._skip_param_init():
            holder_inside.set()
            release.wait(5)

    def follower():
        holder_inside.wait(5)
        with common._skip_param_init():
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

def test_from_safetensors_initialises_nothing_and_writes_every_parameter(tiny_qwen3_moe_dir):
    """No random fill anywhere in from_safetensors, and no parameter survives the load unwritten."""
    initialised: list[str] = []

    def recorder(cls):
        real = cls.reset_parameters

        def reset(self):
            initialised.append(f"{type(self).__name__}{tuple(self.weight.shape)}")
            return real(self)

        return real, reset

    real_from_config = Qwen3MoeForCausalLM.from_config

    def poisoning_from_config(cfg, dtype=torch.bfloat16):
        model = real_from_config(cfg, dtype=dtype)
        with torch.no_grad():
            for _, p in model.named_parameters(remove_duplicate=False):
                p.fill_(float("nan"))
        return model

    saved = {}
    for cls in (nn.Linear, nn.Embedding):
        real, reset = recorder(cls)
        saved[cls] = real
        cls.reset_parameters = reset
    Qwen3MoeForCausalLM.from_config = poisoning_from_config
    try:
        model = Qwen3MoeForCausalLM.from_safetensors(tiny_qwen3_moe_dir)
    finally:
        for cls, real in saved.items():
            cls.reset_parameters = real
        Qwen3MoeForCausalLM.from_config = real_from_config

    assert initialised == [], (
        f"from_safetensors randomly initialised {len(initialised)} module(s) it then overwrote: "
        f"{initialised[:5]} — every construction of ours must sit inside _skip_param_init()")

    poisoned = [n for n, p in model.named_parameters(remove_duplicate=False)
                if torch.isnan(p).any()]
    assert not poisoned, f"from_safetensors never wrote {len(poisoned)} parameter(s): {poisoned[:5]}"
