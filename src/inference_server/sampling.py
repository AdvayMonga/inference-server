"""Token sampling from logits — temperature, top-k, top-p, greedy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """Per-request sampling configuration. Defaults reproduce greedy (argmax)."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables

    @classmethod
    def greedy(cls) -> "SamplingParams":
        return cls()


GREEDY = SamplingParams.greedy()


def sample(logits: torch.Tensor, params: SamplingParams = GREEDY) -> torch.Tensor:
    """Sample one token per row.

    logits: [..., vocab_size] — last dim is vocab. Returns shape with last dim dropped.
    """
    if params.temperature <= 0.0:
        return logits.argmax(dim=-1)

    scaled = logits.float() / params.temperature

    if params.top_k > 0 and params.top_k < scaled.shape[-1]:
        kth = torch.topk(scaled, params.top_k, dim=-1).values[..., -1:]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

    if 0.0 < params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cumprobs > params.top_p
        # Keep the first token that crosses the threshold
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        scaled = torch.empty_like(scaled).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    picked = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return picked.reshape(probs.shape[:-1])


def sample_batched(logits: torch.Tensor, params_per_row: list[SamplingParams]) -> torch.Tensor:
    """Sample one token per row with per-row params. logits: [N, vocab] -> [N].

    Rows sharing identical params are sampled in ONE call, so cost is O(distinct configs),
    not O(rows). The old per-row Python loop launched a full softmax/top-k/top-p over a
    [1, 262144] slice N times per decode step — a direct O(N) term in TPOT.
    """
    if not params_per_row:
        return torch.empty(0, dtype=torch.long, device=logits.device)
    first = params_per_row[0]
    if all(p == first for p in params_per_row):
        return sample(logits, first)
    groups: dict[SamplingParams, list[int]] = {}
    for i, p in enumerate(params_per_row):
        groups.setdefault(p, []).append(i)
    out = torch.empty(len(params_per_row), dtype=torch.long, device=logits.device)
    for p, idx in groups.items():
        rows = torch.tensor(idx, device=logits.device)
        out[rows] = sample(logits.index_select(0, rows), p)
    return out
