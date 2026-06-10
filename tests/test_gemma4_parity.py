"""Parity harness for the custom Gemma 4 forward (M1).

Snapshots HF Gemma 4 E2B-it's outputs on a fixed prompt to a fixture file,
then any custom forward can be compared byte-for-byte against the fixture.

Run modes:
    pytest tests/test_gemma4_parity.py::test_capture_hf_fixture    # produce ground truth
    pytest tests/test_gemma4_parity.py::test_custom_matches_hf     # check custom forward (skipped until M1 lands)

The fixture lives in tests/fixtures/gemma4_e2b_parity.pt and is checked into the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURE_DIR / "gemma4_e2b_parity.pt"
MODEL_NAME = "google/gemma-4-E2B-it"
PROMPT = "The capital of France is"
ATOL = 1e-3  # bf16 tolerance — tighten to 1e-5 if we move to fp32


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.mark.skipif(os.environ.get("CAPTURE_HF_FIXTURE") != "1", reason="set CAPTURE_HF_FIXTURE=1 to regenerate")
def test_capture_hf_fixture():
    """Produce the ground-truth fixture. Run with CAPTURE_HF_FIXTURE=1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()

    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)

    hidden_states = [h.detach().cpu() for h in out.hidden_states]  # [embed_out, layer_0_out, ..., final_norm_out]
    logits = out.logits.detach().cpu()
    next_token_logits = logits[:, -1, :]
    top5 = torch.topk(next_token_logits, k=5, dim=-1)

    FIXTURE_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "prompt": PROMPT,
            "model_name": MODEL_NAME,
            "input_ids": input_ids.cpu(),
            "hidden_states": hidden_states,
            "logits": logits,
            "top5_token_ids": top5.indices.cpu(),
            "top5_token_logits": top5.values.cpu(),
            "num_layers": len(hidden_states) - 1,
        },
        FIXTURE_PATH,
    )


def test_fixture_present():
    """Sanity: someone has captured the HF fixture."""
    assert FIXTURE_PATH.exists(), (
        f"Missing {FIXTURE_PATH}. Run with CAPTURE_HF_FIXTURE=1 once to generate."
    )


def _load_hf_state_dict_cached():
    """Load Gemma 4 state_dict from HF (uses local cache once warmed)."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).state_dict()


def test_attention_matches_hf_sliding_layer0():
    """GemmaAttention vs HF Gemma4TextAttention on sliding layer 0, real hidden_states[0] input."""
    from inference_server.models.gemma4 import (
        GemmaAttention, GemmaRotaryEmbedding,
    )
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    cfg = AutoConfig.from_pretrained(MODEL_NAME).text_config
    cfg._attn_implementation = "sdpa"
    hf_sd = _load_hf_state_dict_cached()
    fixture = torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)

    # Ours
    ours = GemmaAttention(
        hidden_size=cfg.hidden_size,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        eps=cfg.rms_norm_eps,
        sliding_window=cfg.sliding_window,
    ).eval()
    ours.load_hf_weights(hf_sd, layer_idx=0)

    # HF — needs to be bf16
    theirs = Gemma4TextAttention(cfg, layer_idx=0).to(torch.bfloat16).eval()
    p = "model.language_model.layers.0.self_attn"
    theirs.q_proj.weight.data.copy_(hf_sd[f"{p}.q_proj.weight"])
    theirs.k_proj.weight.data.copy_(hf_sd[f"{p}.k_proj.weight"])
    theirs.v_proj.weight.data.copy_(hf_sd[f"{p}.v_proj.weight"])
    theirs.o_proj.weight.data.copy_(hf_sd[f"{p}.o_proj.weight"])
    theirs.q_norm.weight.data.copy_(hf_sd[f"{p}.q_norm.weight"])
    theirs.k_norm.weight.data.copy_(hf_sd[f"{p}.k_norm.weight"])

    # cos/sin from our sliding rotary (already parity-tested)
    rope = GemmaRotaryEmbedding(head_dim=cfg.head_dim, base=10000.0, partial_rotary_factor=1.0).eval()
    position_ids = torch.arange(5).unsqueeze(0)
    with torch.no_grad():
        cos, sin = rope(position_ids)
        cos_bf, sin_bf = cos.to(torch.bfloat16), sin.to(torch.bfloat16)

        x = fixture["hidden_states"][0]
        a = ours(x, cos, sin)
        b, _ = theirs(
            hidden_states=x,
            position_embeddings=(cos_bf, sin_bf),
            attention_mask=None,
            shared_kv_states={},
            past_key_values=None,
        )

    assert a.shape == b.shape, f"{a.shape} vs {b.shape}"
    assert torch.equal(a, b), "attention output diverges from HF"


def test_mlp_matches_hf():
    """GemmaMLP byte-identical to HF on layer 0, using the embedding output as a real input."""
    from inference_server.models.gemma4 import GemmaMLP
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

    cfg = AutoConfig.from_pretrained(MODEL_NAME).text_config
    hf_sd = _load_hf_state_dict_cached()
    fixture = torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)

    ours = GemmaMLP(hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size).eval()
    ours.load_hf_weights(hf_sd, layer_idx=0)
    theirs = Gemma4TextMLP(cfg, layer_idx=0).to(torch.bfloat16).eval()
    theirs.gate_proj.weight.data.copy_(hf_sd["model.language_model.layers.0.mlp.gate_proj.weight"])
    theirs.up_proj.weight.data.copy_(hf_sd["model.language_model.layers.0.mlp.up_proj.weight"])
    theirs.down_proj.weight.data.copy_(hf_sd["model.language_model.layers.0.mlp.down_proj.weight"])

    x = fixture["hidden_states"][0]  # real bf16 embedding-output, [1, 5, 1536]
    with torch.no_grad():
        a = ours(x)
        b = theirs(x)
    assert torch.equal(a, b), "MLP output diverges from HF"


def test_rotary_matches_hf():
    """GemmaRotaryEmbedding cos/sin must equal HF's per layer-type, and apply_rotary too."""
    from inference_server.models.gemma4 import GemmaRotaryEmbedding, apply_rotary
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    cfg = AutoConfig.from_pretrained(MODEL_NAME).text_config
    head_dim = cfg.head_dim

    hf_rope = Gemma4TextRotaryEmbedding(cfg).eval()
    position_ids = torch.arange(5).unsqueeze(0)  # [1, 5]
    dummy = torch.zeros(1, 1, 5, head_dim)  # x ignored by HF rotary forward besides device

    # Sliding-attention layer
    ours_s = GemmaRotaryEmbedding(head_dim=head_dim, base=10000.0, partial_rotary_factor=1.0).eval()
    with torch.no_grad():
        cos_h, sin_h = hf_rope(dummy, position_ids, layer_type="sliding_attention")
        cos_o, sin_o = ours_s(position_ids)
    assert torch.equal(cos_o, cos_h), "sliding cos diverges"
    assert torch.equal(sin_o, sin_h), "sliding sin diverges"

    # Full-attention layer uses global_head_dim=512 (wider lens), base=1e6, partial=0.25.
    # See modeling_gemma4.py::Gemma4TextAttention — head_dim swaps to global_head_dim there too.
    global_head_dim = cfg.global_head_dim
    ours_f = GemmaRotaryEmbedding(head_dim=global_head_dim, base=1_000_000.0, partial_rotary_factor=0.25).eval()
    with torch.no_grad():
        cos_h, sin_h = hf_rope(dummy, position_ids, layer_type="full_attention")
        cos_o, sin_o = ours_f(position_ids)
    assert torch.equal(cos_o, cos_h), "full cos diverges"
    assert torch.equal(sin_o, sin_h), "full sin diverges"

    # apply_rotary parity on a dummy Q tensor — last dim must match cos/sin last dim
    torch.manual_seed(0)
    q = torch.randn(1, 8, 5, global_head_dim, dtype=torch.bfloat16)
    with torch.no_grad():
        out_ours = apply_rotary(q, cos_o, sin_o, unsqueeze_dim=1)
        out_hf = apply_rotary_pos_emb(q, cos_h, sin_h, unsqueeze_dim=1)
    assert torch.equal(out_ours, out_hf), "apply_rotary diverges from HF"


def test_rmsnorm_matches_hf():
    """GemmaRMSNorm output must equal HF's Gemma4RMSNorm on the same input + weight."""
    from inference_server.models.gemma4 import GemmaRMSNorm
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    fixture = torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)
    hf_sd = _load_hf_state_dict_cached()
    w = hf_sd["model.language_model.layers.0.input_layernorm.weight"]

    ours = GemmaRMSNorm(dim=w.shape[0], eps=1e-6).eval()
    ours.weight.data.copy_(w)
    theirs = Gemma4RMSNorm(dim=w.shape[0], eps=1e-6).eval()
    theirs.weight.data.copy_(w.float())  # HF stores it as float-promoted at init

    x = fixture["hidden_states"][0]
    with torch.no_grad():
        a = ours(x)
        b = theirs(x)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a, b), "RMSNorm output diverges from HF"


def test_embedding_matches_hf():
    """GemmaEmbedding output must equal HF's hidden_states[0] (post-scaling embed output)."""
    from inference_server.models.gemma4 import GemmaEmbedding

    fixture = torch.load(FIXTURE_PATH, map_location="cpu", weights_only=False)
    hf_sd = _load_hf_state_dict_cached()

    embed = GemmaEmbedding(vocab_size=262144, hidden_size=1536).eval()
    embed.load_hf_weights(hf_sd)

    with torch.no_grad():
        out = embed(fixture["input_ids"])

    expected = fixture["hidden_states"][0]
    assert out.shape == expected.shape, f"shape {out.shape} vs expected {expected.shape}"
    assert out.dtype == expected.dtype, f"dtype {out.dtype} vs expected {expected.dtype}"
    assert torch.equal(out, expected), "embedding output not byte-identical to HF"
