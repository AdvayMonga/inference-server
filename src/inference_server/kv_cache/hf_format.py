"""Bridge between HF DynamicCache (4D) and CacheManager block storage (3D)."""

import torch
from transformers.cache_utils import DynamicCache

from inference_server.kv_cache.block import Block


def dynamic_cache_to_per_layer_3d(
    dynamic_cache: DynamicCache,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer (K, V) tensors and squeeze batch dim. HF: [1, H, S, D] → [H, S, D]."""
    out = []
    for layer in dynamic_cache.layers:
        k = layer.keys.squeeze(0)
        v = layer.values.squeeze(0)
        out.append((k, v))
    return out


def blocks_to_dynamic_cache(blocks: list[Block]) -> tuple[DynamicCache | None, int]:
    """Reconstruct a DynamicCache from a leading run of blocks with KV tensors.

    Returns (cache, num_tokens). Stops at the first block missing a KV tensor —
    the returned cache and token count are always consistent.
    """
    valid: list[Block] = []
    for b in blocks:
        if b.k_tensor is None:
            break
        valid.append(b)

    if not valid:
        return None, 0

    num_tokens = sum(b.num_tokens_stored for b in valid)
    num_layers = len(valid[0].k_tensor)
    dc = DynamicCache()
    for layer in range(num_layers):
        k_parts = [b.k_tensor[layer] for b in valid]
        v_parts = [b.v_tensor[layer] for b in valid]
        k = torch.cat(k_parts, dim=1).unsqueeze(0)  # [H, S, D] → [1, H, S, D]
        v = torch.cat(v_parts, dim=1).unsqueeze(0)
        dc.update(k, v, layer)
    return dc, num_tokens
