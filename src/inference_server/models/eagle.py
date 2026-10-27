"""EAGLE-3 speculative decoding (SPEC_EAGLE_MODEL, e.g. Tengyunw/qwen3_30b_moe_eagle3): a one-layer draft head that
reads the target's residual stream entering three of its layers (`aux_layers`) and the next token's embedding, and
proposes K tokens; the target verifies [current] + drafts in one forward, greedy, so the output is the target's own.

Pairing (EAGLE-3): the head's row at position p reads the target's aux at p and the token at p + 1, and predicts the
token at p + 2; its own output (the residual before its final norm) stands in for the target's aux when it chains.
After a verify of [x_n, d_1..d_K] at n..n+K with greedy outputs g_0..g_K, a = the drafts accepted, the head is re-fed
(aux_{n+j}, g_j) for every j at n+j (rows past a are overwritten later), and row a's prediction starts the next drafts.

`Runner` keeps one request's state on the device: the head's K/V cache, the verify inputs (tokens, positions, the
block table), the verify outputs (greedy tokens, aux) and the accepted count; `post()` (acceptance, the head's update,
the next drafts) is one CUDA graph per attention length bucket, so a step is one target launch and one replay."""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn


def aux_layers(n_layers: int) -> tuple[int, int, int]:
    """The target layers whose input residual EAGLE-3 reads (SGLang's and EAGLE's choice): low, middle, high."""
    return (2, n_layers // 2, n_layers - 3)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Llama RMSNorm: fp32 statistics, the weight applied in the input's dtype."""
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rotate(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Eagle3Head(nn.Module):
    """The draft head: fc (3H -> H) on the target's aux, one Llama decoder layer whose attention reads
    [RMSNorm(embed(token)), RMSNorm(hidden)] (2H wide), the final norm, lm_head over the draft vocabulary, and d2t back
    to the target's ids. The embedding is the target's."""

    def __init__(self, cfg: dict, embed: nn.Embedding, sd: dict, device, dtype=torch.bfloat16):
        super().__init__()
        self.H = cfg["hidden_size"]
        self.nq, self.nkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.embed = embed
        w = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in sd.items()}
        m = "midlayer."
        self.fc = w["fc.weight"]
        self.wq, self.wk, self.wv = w[m + "self_attn.q_proj.weight"], w[m + "self_attn.k_proj.weight"], w[m + "self_attn.v_proj.weight"]
        self.wqkv = torch.cat([self.wq, self.wk, self.wv]).contiguous()           # one GEMM for q, k, v
        self.wo = w[m + "self_attn.o_proj.weight"]
        self.wgu = torch.cat([w[m + "mlp.gate_proj.weight"], w[m + "mlp.up_proj.weight"]]).contiguous()
        self.wd = w[m + "mlp.down_proj.weight"]
        self.n_in, self.n_hid = w[m + "input_layernorm.weight"], w[m + "hidden_norm.weight"]
        self.n_post, self.n_final = w[m + "post_attention_layernorm.weight"], w["norm.weight"]
        self.lm_head = w["lm_head.weight"]
        self.d2t = w["d2t"].to(torch.long)
        self.inter = self.wd.shape[1]
        self.inv_freq = 1.0 / (cfg.get("rope_theta", 1e6) ** (torch.arange(0, self.hd, 2, device=device, dtype=torch.float32) / self.hd))
        assert self.fc.shape == (self.H, 3 * self.H) and self.wqkv.shape[1] == 2 * self.H

    @classmethod
    def load(cls, name: str, embed: nn.Embedding, device, dtype=torch.bfloat16) -> "Eagle3Head":
        """From a local dir or a Hub repo holding config.json and pytorch_model.bin (or model.safetensors)."""
        if os.path.isdir(name):
            path = name
        else:
            from huggingface_hub import snapshot_download
            path = snapshot_download(name, allow_patterns=["config.json", "*.bin", "*.safetensors"])
        cfg = json.load(open(os.path.join(path, "config.json")))
        st = os.path.join(path, "model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file
            sd = load_file(st)
        else:
            sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        return cls(cfg, embed, sd, device, dtype)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor, pos: torch.Tensor,
                kc: torch.Tensor, vc: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows (tokens [S], hidden [S, 3H] target aux or [S, H] the head's own, pos [S]): each row's K/V into the
        caches kc/vc [max_len, nkv, hd] at its position, attention over positions <= its own among the first L ->
        (greedy next token in the target's ids [S], the row's output residual [S, H])."""
        S, H = tokens.shape[0], self.H
        if hidden.shape[-1] != H:
            hidden = hidden @ self.fc.T
        resid = hidden
        x = torch.cat([_rms(self.embed(tokens), self.n_in, self.eps), _rms(hidden, self.n_hid, self.eps)], dim=-1)
        qkv = x @ self.wqkv.T
        q = qkv[:, :self.nq * self.hd].view(S, self.nq, self.hd)
        k = qkv[:, self.nq * self.hd:(self.nq + self.nkv) * self.hd].view(S, self.nkv, self.hd)
        v = qkv[:, (self.nq + self.nkv) * self.hd:].view(S, self.nkv, self.hd)
        f = pos.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat([f, f], dim=-1)
        cos, sin = emb.cos().to(q.dtype)[:, None, :], emb.sin().to(q.dtype)[:, None, :]
        q, k = q * cos + _rotate(q) * sin, k * cos + _rotate(k) * sin
        kc.index_copy_(0, pos, k)
        vc.index_copy_(0, pos, v)
        keys = torch.arange(L, device=tokens.device)
        mask = keys[None, :] <= pos[:, None]                                         # [S, L]
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], kc[:L].transpose(0, 1)[None],
                                           vc[:L].transpose(0, 1)[None], attn_mask=mask[None, None], enable_gqa=True)
        o = o[0].transpose(0, 1).reshape(S, self.nq * self.hd) @ self.wo.T
        r1 = resid + o
        gu = _rms(r1, self.n_post, self.eps) @ self.wgu.T
        m = (F.silu(gu[:, :self.inter]) * gu[:, self.inter:]) @ self.wd.T
        r2 = r1 + m
        logits = _rms(r2, self.n_final, self.eps) @ self.lm_head.T
        d = logits.argmax(-1)
        return d + self.d2t[d], r2


_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768)


class Runner:
    """One request's EAGLE state, all on the device. `begin` feeds the prompt; then each step: the caller's verify
    writes `greedy` [K+1] and `aux` [K+1, 3, H] for `cur` at `vpos` (block table `vbt`), then `post()` computes the
    accepted count `acc` (= a + 1 tokens, greedy[0..a]), updates the head and writes the next `cur`/`vpos`."""

    def __init__(self, head: Eagle3Head, K: int, max_len: int, device, graphs: bool = True):
        self.head, self.K, self.max_len, self.device = head, K, max_len, device
        H, dt = head.H, head.wo.dtype
        self.kc = torch.zeros(max_len, head.nkv, head.hd, dtype=dt, device=device)
        self.vc = torch.zeros_like(self.kc)
        n = K + 1
        self.cur = torch.zeros(n, dtype=torch.long, device=device)       # verify inputs: [current, drafts]
        self.vpos = torch.zeros(n, dtype=torch.long, device=device)      # their positions
        self.greedy = torch.zeros(n, dtype=torch.long, device=device)    # verify outputs
        self.aux = torch.zeros(n, 3, H, dtype=dt, device=device)
        self.pos0 = torch.zeros(1, dtype=torch.long, device=device)      # the current token's position
        self.acc = torch.zeros(1, dtype=torch.long, device=device)       # tokens the last step produced (a + 1)
        self.out = torch.zeros(n, dtype=torch.long, device=device)       # the last step's greedy tokens
        self._ar = torch.arange(n, device=device)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._use_graphs = graphs and torch.device(device).type == "cuda"

    def bucket(self, upto: int) -> int:
        """The smallest attention length holding positions < upto."""
        for b in _BUCKETS:
            if upto <= b and b <= self.max_len:
                return b
        return self.max_len

    @torch.no_grad()
    def begin(self, prompt: list[int], first: int, aux: torch.Tensor, chunk: int = 512) -> None:
        """Feed the prompt (aux [n, 3H] at positions 0..n-1, paired with prompt[1:] + [first]), then draft K."""
        n = len(prompt)
        toks = torch.tensor(prompt[1:] + [first], dtype=torch.long, device=self.device)
        L = self.bucket(n + 1)
        d = h = None
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            pos = torch.arange(s, e, device=self.device)
            d, h = self.head(toks[s:e], aux[s:e].to(self.device), pos, self.kc, self.vc, L)
        self.pos0.fill_(n)
        self._draft(first_tok=torch.tensor([first], device=self.device), d=d[-1:], h=h[-1:], L=self.bucket(n + 2 * self.K + 2))

    def _draft(self, first_tok, d, h, L) -> None:
        """cur = [first_tok, d, then K-1 chained drafts]; vpos from pos0. The head's row at pos0 (token d, the token at
        pos0 + 1) predicts the token at pos0 + 2, and so on."""
        drafts, p = [d], self.pos0.clone()
        for _ in range(self.K - 1):
            d, h = self.head(d, h, p, self.kc, self.vc, L)
            drafts.append(d)
            p = p + 1
        self.cur.copy_(torch.cat([first_tok] + drafts))
        self.vpos.copy_(self.pos0 + self._ar)

    def _post(self, L: int) -> None:
        K = self.K
        match = (self.greedy[:K] == self.cur[1:]).long()
        a = torch.cumprod(match, 0).sum().view(1)                        # drafts accepted
        d, h = self.head(self.greedy, self.aux.view(K + 1, -1), self.vpos, self.kc, self.vc, L)
        self.acc.copy_(a + 1)
        self.out.copy_(self.greedy)
        bonus = self.greedy.gather(0, a)
        self.pos0.add_(a + 1)
        self._draft(first_tok=bonus, d=d.gather(0, a), h=h.index_select(0, a), L=L)

    @torch.no_grad()
    def post(self, upto: int) -> None:
        """After a verify: acceptance, the head's update, the next drafts. `upto` bounds every position this step
        may touch (the caller's host view of pos0 + 3(K+1)): it picks the attention length bucket."""
        L = self.bucket(upto)
        if not self._use_graphs:
            self._post(L)
            return
        g = self._graphs.get(L)
        if g is None:
            g = self._capture(L)
        g.replay()

    def _capture(self, L: int) -> torch.cuda.CUDAGraph:
        """Capture `_post` at length L on copies of the state, so the warmup runs leave the live state untouched."""
        live = [t.clone() for t in (self.kc, self.vc, self.cur, self.vpos, self.greedy, self.aux, self.pos0, self.acc, self.out)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._post(L)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._post(L)
        for t, v in zip((self.kc, self.vc, self.cur, self.vpos, self.greedy, self.aux, self.pos0, self.acc, self.out), live):
            t.copy_(v)
        self._graphs[L] = g
        return g
