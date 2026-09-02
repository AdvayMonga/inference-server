"""lm_head position slicing: `logits_index` must equal slicing the full logits afterwards.

Prefill needs one logit row per sequence, but running lm_head over every position is the
largest FLOP-and-bytes item in a prefill wave (a K x Smax x 262144 projection to read K rows).
Slicing the hidden states first must not change the answer.

NOT bit-exact, and that is expected: lm_head over [B, 1, H] is a different GEMM shape than
over [B, S, H], so the reduction order differs (~3e-6 in fp32 here). Same phenomenon as the
batch-width note in DECISIONS.md. The invariant that matters for greedy decoding is that the
argmax agrees, so both are asserted.
"""

from __future__ import annotations

import torch

from inference_server.models.gemma4 import GemmaForCausalLM, GemmaModel

VOCAB = 64


def _tiny_model():
    torch.manual_seed(0)
    model = GemmaModel(
        vocab_size=VOCAB, hidden_size=32, num_layers=2, intermediate_size=64,
        num_q_heads=4, num_kv_heads=1, head_dim=8, global_head_dim=8,
        hidden_per_layer=8, layer_types=["sliding_attention", "full_attention"],
        sliding_window=16, num_kv_shared_layers=0, use_double_wide_mlp=False,
        eps=1e-6, dtype=torch.float32,
    )
    return GemmaForCausalLM(model, final_logit_softcapping=30.0).eval()


def test_logits_index_matches_full_then_slice():
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (3, 7))
    idx = torch.tensor([6, 3, 0])
    with torch.no_grad():
        full = model(ids)
        sliced = model(ids, logits_index=idx)
    assert sliced.shape == (3, 1, full.shape[-1])
    for row, j in enumerate(idx.tolist()):
        assert torch.allclose(sliced[row, 0], full[row, j], rtol=1e-5, atol=1e-5), f"row {row}"
        assert sliced[row, 0].argmax() == full[row, j].argmax(), f"row {row} argmax"


def test_logits_index_none_is_unchanged():
    """Default path must be byte-identical to before the parameter existed."""
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (2, 5))
    with torch.no_grad():
        a = model(ids)
        b = model(ids, logits_index=None)
    assert torch.equal(a, b)
    assert a.shape[1] == 5


def test_logits_index_ragged_last_positions():
    """The real prefill case: a right-padded wave where each row wants its own last real token."""
    model = _tiny_model()
    ids = torch.randint(0, VOCAB, (4, 6))
    lens = torch.tensor([6, 4, 1, 5])
    with torch.no_grad():
        full = model(ids)
        got = model(ids, logits_index=lens - 1)
    for row in range(4):
        ref = full[row, lens[row] - 1]
        assert torch.allclose(got[row, 0], ref, rtol=1e-5, atol=1e-5)
        assert got[row, 0].argmax() == ref.argmax()
