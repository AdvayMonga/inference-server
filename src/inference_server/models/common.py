"""Pieces shared by the model forward and the backend: parameter-init skipping, the contiguous
KV cache the plain (unpaged) forward uses, and HF-layout rotary embeddings."""

from __future__ import annotations

import contextlib
import threading

import torch
import torch.nn as nn


# A plain Lock, not an RLock, on purpose: same-thread nesting of the patch window does not happen
# in any loader. If a change ever introduces it, Lock deadlocks loudly at the call site; RLock
# would permit it silently and hold the process-wide patch open across work that has no business
# being inside it.
_INIT_PATCH_LOCK = threading.Lock()


@contextlib.contextmanager
def _skip_param_init():
    """Allocate parameters without randomly filling them — the loader overwrites every one.

    Kaiming-uniform over billions of parameters is tens of seconds of pure waste when the next
    step is a full copy from the checkpoint. Buffers are untouched, so the RoPE tables still
    compute exactly as before.

    What the lock does and does not protect. The patch is PROCESS-WIDE — nn.Linear and
    nn.Embedding are shared by every module in the process, ours or not. The lock serialises
    concurrent callers of this context manager, so two loads cannot restore each other's saved
    functions in the wrong order. It cannot protect an unrelated `nn.Linear(...)` built by
    another thread inside the window: that module comes back unfilled, which is silent garbage
    rather than a crash. Accepted because the window is small and happens once, at startup,
    before the server serves anything — `CustomTorchBackend` awaits its single
    `run_in_executor` load before accepting traffic.
    """
    with _INIT_PATCH_LOCK:
        saved = {c: c.reset_parameters for c in (nn.Linear, nn.Embedding)}
        for c in saved:
            c.reset_parameters = lambda self: None
        try:
            yield
        finally:
            for c, fn in saved.items():
                c.reset_parameters = fn


class KVCache:
    """Per-layer (K, V) tensors stored as contiguous [B, H, S, D]. Append concatenates along S."""

    def __init__(self, num_layers: int):
        self.k: list[torch.Tensor | None] = [None] * num_layers
        self.v: list[torch.Tensor | None] = [None] * num_layers

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.k[layer_idx] is None:
            return None
        return self.k[layer_idx], self.v[layer_idx]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)

    @property
    def seq_len(self) -> int:
        """Length of cached sequence. Any populated layer suffices."""
        for k in self.k:
            if k is not None:
                return k.shape[2]
        return 0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF-style: [-x[..., d/2:], x[..., :d/2]]. Differs from original RoPE's interleaved pairs."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """Apply rotary embeddings: (x * cos) + (rotate_half(x) * sin). cos/sin broadcast over heads."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """(cos, sin) tables for `head_dim`, full rotation, in HF's rotate_half layout."""

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", freqs, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, S] → cos/sin: [B, S, head_dim]
        freqs = position_ids[..., None].float() * self.inv_freq[None, None, :]  # [B, S, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, head_dim] — matches rotate_half layout
        return emb.cos(), emb.sin()
