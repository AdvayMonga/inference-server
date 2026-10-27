"""EAGLE-3 head and runner (models/eagle.py) on a tiny random head: rows fed together or one by one give the same
outputs; the runner's drafts at every step equal a reference that recomputes them from the whole history with a fresh
cache (so the cache, the positions and the re-feed after a verify are right); its emitted tokens are the target's
sequence exactly, whatever the drafts; the CUDA-graph step equals the eager one."""

import pytest
import torch
from torch import nn

from inference_server.models.eagle import Eagle3Head, Runner

H, NQ, NKV, HD, INTER, V, DV = 32, 4, 2, 8, 64, 50, 40


@pytest.fixture(scope="module")
def head():
    torch.manual_seed(0)
    cfg = {"hidden_size": H, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD,
           "rms_norm_eps": 1e-6, "rope_theta": 1e4}
    r = lambda *s: torch.randn(*s) * 0.3
    m = "midlayer."
    d = torch.arange(DV)
    sd = {"fc.weight": r(H, 3 * H), m + "self_attn.q_proj.weight": r(NQ * HD, 2 * H),
          m + "self_attn.k_proj.weight": r(NKV * HD, 2 * H), m + "self_attn.v_proj.weight": r(NKV * HD, 2 * H),
          m + "self_attn.o_proj.weight": r(H, NQ * HD), m + "mlp.gate_proj.weight": r(INTER, H),
          m + "mlp.up_proj.weight": r(INTER, H), m + "mlp.down_proj.weight": r(H, INTER),
          m + "input_layernorm.weight": 1 + r(H), m + "hidden_norm.weight": 1 + r(H),
          m + "post_attention_layernorm.weight": 1 + r(H), "norm.weight": 1 + r(H), "lm_head.weight": r(DV, H),
          "d2t": (torch.randperm(V)[:DV] - d)}                        # target id = d + d2t[d]: distinct, < V
    emb = nn.Embedding(V, H)
    return Eagle3Head(cfg, emb, sd, "cpu", dtype=torch.float32)


def _cache(n=256):
    return torch.zeros(n, NKV, HD), torch.zeros(n, NKV, HD)


def test_rows_together_equal_rows_one_by_one(head):
    torch.manual_seed(1)
    S = 9
    toks, hid, pos = torch.randint(0, V, (S,)), torch.randn(S, 3 * H), torch.arange(S)
    d1, h1 = head(toks, hid, pos, *_cache(), 64)
    kc, vc = _cache()
    rows = [head(toks[i:i + 1], hid[i:i + 1], pos[i:i + 1], kc, vc, 64) for i in range(S)]
    assert torch.equal(d1, torch.cat([d for d, _ in rows]))
    assert torch.allclose(h1, torch.cat([h for _, h in rows]), atol=1e-5)


def _reference_drafts(head, seq, aux, pos0, K):
    """Drafts after the token at pos0, from scratch: the target's (aux_p, seq[p+1]) for p < pos0, then K - 1 chained."""
    kc, vc = _cache()
    p = torch.arange(pos0)
    d, h = head(torch.tensor(seq[1:pos0 + 1]), aux[:pos0].reshape(pos0, -1), p, kc, vc, 256)
    d, h = d[-1:], h[-1:]
    out, q = [int(d)], pos0
    for _ in range(K - 1):
        d, h = head(d, h, torch.tensor([q]), kc, vc, 256)
        out.append(int(d))
        q += 1
    return out


@pytest.mark.parametrize("K", [1, 2, 3])
def test_runner_drafts_match_the_reference_and_emit_the_target_sequence(head, K):
    torch.manual_seed(2)
    n, S = 7, 40
    aux = torch.randn(S + K + 2, 3, H)
    seq = torch.randint(0, V, (S + K + 2,)).tolist()
    # make some drafts right: the target's tokens after a few positions are what the head would draft
    run = Runner(head, K, 256, "cpu")
    run.begin(seq[:n], seq[n], aux[:n].reshape(n, -1))
    emitted = [seq[n]]
    accepted = 0
    while len(emitted) < S - n:
        p0 = int(run.pos0)
        assert run.cur.tolist() == [seq[p0]] + _reference_drafts(head, seq, aux, p0, K)
        vpos = run.vpos
        run.greedy.copy_(torch.tensor(seq)[vpos + 1])                  # the target: its own next token at each row
        run.aux.copy_(aux[vpos])
        run.post(p0 + 3 * (K + 1))
        a = int(run.acc) - 1
        accepted += a
        emitted += run.out[:a + 1].tolist()
        assert int(run.pos0) == p0 + a + 1
    assert emitted[:S - n] == seq[n:S]


def test_runner_accepts_drafts_the_target_agrees_with(head):
    """A target whose next token is always the head's own first draft accepts every draft."""
    torch.manual_seed(3)
    K, n = 3, 5
    run = Runner(head, K, 256, "cpu")
    seq = torch.randint(0, V, (n + 1,)).tolist()
    aux = torch.randn(n, 3, H)
    run.begin(seq[:n], seq[n], aux.reshape(n, -1))
    for _ in range(4):
        run.greedy.copy_(torch.cat([run.cur[1:], torch.tensor([7])]))   # agrees with every draft, then 7
        run.aux.copy_(torch.randn(K + 1, 3, H))
        p0 = int(run.pos0)
        run.post(p0 + 3 * (K + 1))
        assert int(run.acc) == K + 1 and int(run.pos0) == p0 + K + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_graph_step_equals_eager_step(head):
    dev = "cuda"
    hd = Eagle3Head({"hidden_size": H, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": HD,
                     "rms_norm_eps": 1e-6, "rope_theta": 1e4}, head.embed.to(dev),
                    {k: v for k, v in _state(head).items()}, dev, dtype=torch.float32)
    torch.manual_seed(4)
    K, n = 2, 6
    seq = torch.randint(0, V, (64,)).tolist()
    aux = torch.randn(64, 3, H, device=dev)
    runs = [Runner(hd, K, 1024, dev, graphs=g) for g in (False, True)]
    for r in runs:
        r.begin(seq[:n], seq[n], aux[:n].reshape(n, -1))
    for _ in range(10):
        for r in runs:
            r.greedy.copy_(torch.tensor(seq, device=dev)[r.vpos + 1])
            r.aux.copy_(aux[r.vpos])
            r.post(int(r.pos0) + 3 * (K + 1))
        assert torch.equal(runs[0].cur, runs[1].cur) and torch.equal(runs[0].acc, runs[1].acc)


def _state(head):
    m = "midlayer."
    q, k = head.nq * head.hd, head.nkv * head.hd
    return {"fc.weight": head.fc, m + "self_attn.q_proj.weight": head.wqkv[:q], m + "self_attn.k_proj.weight": head.wqkv[q:q + k],
            m + "self_attn.v_proj.weight": head.wqkv[q + k:], m + "self_attn.o_proj.weight": head.wo,
            m + "mlp.gate_proj.weight": head.wgu[:head.inter], m + "mlp.up_proj.weight": head.wgu[head.inter:],
            m + "mlp.down_proj.weight": head.wd, m + "input_layernorm.weight": head.n_in, m + "hidden_norm.weight": head.n_hid,
            m + "post_attention_layernorm.weight": head.n_post, "norm.weight": head.n_final, "lm_head.weight": head.lm_head,
            "d2t": head.d2t}
