"""Prefill attention over the paged K/V with FlashInfer: ragged queries, paged keys, one call per layer for the whole
wave, no padding and no K/V gather.

The SDPA paths (prefill_attention.py) gather every row's cached prefix out of the blocks and run a masked attention
over the right-padded wave, or one flash call per row: the 2026-10-16 session found each in turn as the step that
capped TTFT at concurrency 128 (a 116-row step of ~17 tokens each took 3.1 s). FlashInfer's batch prefill reads the
pool directly ("HND" layout = our [blocks, Hkv, block_size, D]); `causal=True` with a query shorter than its keys
aligns the query to the end, which is exactly "prefix keys, then the causal suffix". Each layer has its own block
ids in principle, but every layer allocates and releases in the same order, so the wrapper is planned once per
wave (host work, ~0.1-0.3 ms per plan) and re-planned only for a layer whose block ids differ.
Opt-in with CUSTOM_BACKEND_PREFILL_ATTN=flashinfer; falls back to sdpa when flashinfer is not importable."""

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

WORKSPACE_BYTES = 128 * 1024 * 1024


def available() -> bool:
    try:
        import flashinfer  # noqa: F401
    except Exception as e:  # noqa: BLE001  (an import error of any kind means the sdpa path)
        logger.warning("flashinfer not usable (%s): prefill attention stays on sdpa", e)
        return False
    return True


# Over fp8 K/V pools, q goes in as fp8 too (unit scale, like the pools) and the whole attention runs on FA3's fp8
# kernel: FlashInfer 0.7.2's fa3 backend does not build for bf16 q over fp8 K/V, and its auto backend then runs FA2,
# 3x slower per call on the H200 (2026-11-01: a 2,048-token causal chunk 208 -> 71 us, 2,048 over 6,144 965 -> 326).
FP8_Q = os.environ.get("CUSTOM_BACKEND_PREFILL_FP8_Q", "1") == "1"
FP8 = torch.float8_e4m3fn


@functools.lru_cache(maxsize=None)
def _wrapper(device: torch.device, backend: str | None = None):
    import flashinfer
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    # auto | fa2 | fa3 | cutlass | trtllm-gen. fa3 measured long_prompt 14.2 req/s once and 6.25 once against auto's
    # 12.0-12.5 (H200 2026-10-18; one-off ~2 s stalls on tiny steps in both fa3 runs, unexplained), so it stays opt-in
    # for bf16 q; fp8 q needs it (`backend`).
    backend = backend or os.environ.get("CUSTOM_BACKEND_FLASHINFER_BACKEND", "auto")
    try:
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="HND", backend=backend)
    except Exception as e:  # noqa: BLE001
        logger.warning("flashinfer backend %r unavailable (%s): using auto", backend, e)
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="HND", backend="auto")


_PLAN_DONE: dict = {}           # wrapper id -> event recorded after its last plan's host -> device copies


def plan(w, *args, **kwargs) -> None:
    """`w.plan`, once the GPU has read the previous plan: FlashInfer stages each plan in one pinned host buffer and copies
    it asynchronously, so a host running ahead of the GPU (prefill graphs) overwrote a plan not yet copied, and attention
    read another step's block indices (illegal memory access under long prompts, H200 2026-11-03)."""
    ev = _PLAN_DONE.get(id(w))
    if ev is not None:
        ev.synchronize()
    w.plan(*args, **kwargs)
    ev = _PLAN_DONE[id(w)] = torch.cuda.Event()
    ev.record()


def plan_inputs(prefix: list[int], suffix: list[int], block_size: int):
    """The wave's row structure for FlashInfer, on the host (plan reads indptrs there, so no device sync):
    (qo_indptr, kv_indptr, kv_last_page_len) int32, and each row's block count."""
    lens = [p + s for p, s in zip(prefix, suffix)]
    nb = [(n + block_size - 1) // block_size for n in lens]
    qo, kv = [0], [0]
    for s, n in zip(suffix, nb):
        qo.append(qo[-1] + s)
        kv.append(kv[-1] + n)
    last = [(n - 1) % block_size + 1 for n in lens]
    i32 = torch.int32
    return torch.tensor(qo, dtype=i32), torch.tensor(kv, dtype=i32), torch.tensor(last, dtype=i32), nb


def kv_indices(blocks: list[list[int]], nb: list[int], device) -> torch.Tensor:
    """Each row's first nb[r] block ids, concatenated (int32 on `device`). From the host's block lists: a boolean
    mask over the device table waits for every kernel queued before it."""
    from inference_server.models.paged_kv_cache import to_device
    return to_device([b for row, n in zip(blocks, nb) for b in row[:n]], device, torch.int32)


def _same_blocks(ctx, layer_idx: int, other: int) -> bool:
    """Whether every row holds the same block ids in `layer_idx` as in `other` (host lists, no device work). Every
    layer allocates and releases in the same order, so this holds for all of Qwen3's layers; one plan serves them."""
    caches = getattr(ctx, "caches", None)
    return caches is not None and all(c.block_tables[layer_idx] == c.block_tables[other] for c in caches)


def flashinfer_prefill(q: torch.Tensor, ctx, layer_idx: int, scale: float) -> torch.Tensor:
    """q [B, Hq, S, D] right-padded; the suffix K/V already appended to the pool. Returns [B, S, Hq, D] in q's dtype,
    padding rows zero. `ctx` is the backend's prefill context (`_prefix`/`_suffix`, `tokens`, `block_table_tensor`)."""
    B, Hq, S, D = q.shape
    pool = ctx.pools[layer_idx]
    Hkv, bs = pool.k.shape[1], pool.k.shape[2]
    st = getattr(ctx, "_fi", None)
    fp8_q = FP8_Q and pool.k.dtype == FP8
    w = _wrapper(q.device, "fa3" if fp8_q else None)
    if st is None:                                   # once per wave: the row structure, and the plan for this layer
        tokens = getattr(ctx, "tokens", None)        # the real positions of a ragged wave (None when none is padded)
        if tokens is None and sum(ctx._suffix) != B * S:
            valid = torch.arange(S, device=q.device)[None, :] < torch.tensor(ctx._suffix, device=q.device)[:, None]
            tokens = valid.reshape(-1).nonzero().squeeze(1)
        st = ctx._fi = (*plan_inputs(ctx._prefix, ctx._suffix, bs), tokens, layer_idx)
    qo_indptr, kv_indptr, last, nb, tokens, planned = st
    if layer_idx != planned and not _same_blocks(ctx, layer_idx, planned):
        planned = layer_idx                          # a layer with its own block ids: its own plan
    if planned == layer_idx:
        getattr(ctx, "_ensure_blocks", lambda: ctx.block_table_tensor(layer_idx))()     # every row's blocks allocated
        _PLANNED.pop(id(w), None)                    # `attend`'s plan is this wrapper's no longer
        plan(w, qo_indptr, kv_indptr, kv_indices([c.blocks for c in ctx.caches], nb, q.device), last, Hq, Hkv, D, bs,
               causal=True, sm_scale=scale, q_data_type=FP8 if fp8_q else q.dtype, kv_data_type=pool.k.dtype,
               o_data_type=q.dtype)
    qp = q.permute(0, 2, 1, 3).reshape(B * S, Hq, D)
    qr = qp[tokens] if tokens is not None else qp
    if fp8_q:
        from inference_server.models import fused
        qr = fused.cast_fp8(qr)
    o = torch.empty(qr.shape, dtype=q.dtype, device=q.device)       # [total real tokens, Hq, D]; every row is written,
    w.run(qr, (pool.k, pool.v), out=o)                              # so not FlashInfer's default zero-filled buffer
    if tokens is None:
        return o.view(B, S, Hq, D)
    return torch.zeros(B * S, Hq, D, dtype=o.dtype, device=o.device).index_copy_(0, tokens, o).view(B, S, Hq, D)


# --- the same attention from tensors, for the prefill stations (models/stations.py) ---

_PLANNED: dict = {}             # wrapper id -> (the kv_indices tensor it was planned for, the rest of the plan's key)


def wave(ctx, q: torch.Tensor) -> dict:
    """The wave's FlashInfer inputs as tensors, made once per wave (every layer shares one block layout) after the
    first layer's K/V append: what the prefill stations' attention reads."""
    st = getattr(ctx, "_fi_wave", None)
    if st is None:
        B, Hq, S, D = q.shape
        tokens = getattr(ctx, "tokens", None)
        if tokens is None and sum(ctx._suffix) != B * S:
            valid = torch.arange(S, device=q.device)[None, :] < torch.tensor(ctx._suffix, device=q.device)[:, None]
            tokens = valid.reshape(-1).nonzero().squeeze(1)
        qo, kv, last, nb = plan_inputs(ctx._prefix, ctx._suffix, ctx.pools[0].k.shape[2])
        getattr(ctx, "_ensure_blocks", lambda: ctx.block_table_tensor(0))()
        st = ctx._fi_wave = {"fi_qo_indptr": qo, "fi_kv_indptr": kv, "fi_last_page_len": last, "fi_tokens": tokens,
                             "fi_kv_indices": kv_indices([c.blocks for c in ctx.caches], nb, q.device)}
    return st


def attend(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor, qo_indptr: torch.Tensor,
           kv_indptr: torch.Tensor, kv_indices_: torch.Tensor, last: torch.Tensor, tokens: torch.Tensor | None,
           scale: float) -> torch.Tensor:
    """`flashinfer_prefill` from the wave's tensors (`wave`): q [B, Hq, S, D] -> [B, S, Hq, D] in q's dtype, padding
    rows zero. Plans only when the wave (its kv_indices tensor) or the shapes change."""
    B, Hq, S, D = q.shape
    Hkv, bs = k_pool.shape[1], k_pool.shape[2]
    fp8_q = FP8_Q and k_pool.dtype == FP8
    w = _wrapper(q.device, "fa3" if fp8_q else None)
    key = (Hq, Hkv, D, bs, scale, q.dtype, k_pool.dtype)
    done = _PLANNED.get(id(w))
    if done is None or done[0] is not kv_indices_ or done[1] != key:
        plan(w, qo_indptr, kv_indptr, kv_indices_, last, Hq, Hkv, D, bs, causal=True, sm_scale=scale,
               q_data_type=FP8 if fp8_q else q.dtype, kv_data_type=k_pool.dtype, o_data_type=q.dtype)
        _PLANNED[id(w)] = (kv_indices_, key)     # holding the tensor keeps its memory from being reused by another
    qp = q.permute(0, 2, 1, 3).reshape(B * S, Hq, D)
    qr = qp[tokens] if tokens is not None else qp
    if fp8_q:
        from inference_server.models import fused
        qr = fused.cast_fp8(qr)
    o = torch.empty(qr.shape, dtype=q.dtype, device=q.device)
    w.run(qr, (k_pool, v_pool), out=o)
    if tokens is None:
        return o.view(B, S, Hq, D)
    return torch.zeros(B * S, Hq, D, dtype=o.dtype, device=o.device).index_copy_(0, tokens, o).view(B, S, Hq, D)
