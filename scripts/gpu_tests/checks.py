"""The CUDA correctness checks, as plain functions any CUDA box can run.

Both the `*_modal.py` gate scripts and `cuda_gate.py` (the RunPod instrument) call these, so
the gate is one set of checks with two launchers rather than two copies that drift. Each check
returns `(passed, detail)` and imports torch lazily: this module must import on a laptop with
no CUDA and no triton, so the gate can be dry-run and the modal scripts stay importable.
"""

from __future__ import annotations

import time
from typing import Callable

TOL = 5e-2


def _reference(q, k_pool, v_pool, block_tables, seq_lens, scale, window=1 << 30):
    """Pure-torch paged attention: gather each seq's real K/V, softmax over the last
    `window` keys (window >= L → full attention)."""
    import torch

    N, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    bs = k_pool.shape[2]
    group = Hq // num_kv_heads
    out = torch.empty(N, Hq, D, dtype=torch.float32, device=q.device)
    for i in range(N):
        L = int(seq_lens[i].item())
        lo = max(0, L - window)                      # window lower bound (key pos >= lo)
        nb = (L + bs - 1) // bs
        ks, vs = [], []
        for b in range(nb):
            blk = int(block_tables[i, b].item())
            ks.append(k_pool[blk])  # [num_kv_heads, bs, D]
            vs.append(v_pool[blk])
        K = torch.cat(ks, dim=1)[:, lo:L, :].float()  # [num_kv_heads, win, D]
        V = torch.cat(vs, dim=1)[:, lo:L, :].float()
        K = K.repeat_interleave(group, dim=0)        # [Hq, win, D]
        V = V.repeat_interleave(group, dim=0)
        qi = q[i].float()                            # [Hq, D]
        scores = (qi[:, None, :] * K).sum(-1) * scale  # [Hq, win]
        attn = torch.softmax(scores, dim=-1)
        out[i] = (attn[:, :, None] * V).sum(dim=1)   # [Hq, D]
    return out


def paged_decode_parity() -> tuple[bool, str]:
    """Decode kernel vs the torch reference: sliding (256) and full (512) head_dim, GQA 8:1."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_decode_attention

    torch.manual_seed(0)
    results = []
    for D, num_kv_heads, Hq in [(256, 1, 8), (512, 1, 8), (256, 2, 8)]:
        N, num_blocks, bs = 5, 64, 16
        max_blocks = 8
        seq_lens = torch.tensor([1, 16, 17, 40, 100][:N], dtype=torch.int32, device="cuda")
        seq_lens = torch.clamp(seq_lens, max=max_blocks * bs)
        k_pool = torch.randn(num_blocks, num_kv_heads, bs, D, dtype=torch.bfloat16, device="cuda")
        v_pool = torch.randn_like(k_pool)
        # distinct physical blocks per (seq, logical block)
        bt = torch.arange(N * max_blocks, dtype=torch.int32, device="cuda").reshape(N, max_blocks) % num_blocks
        q = torch.randn(N, Hq, D, dtype=torch.bfloat16, device="cuda")

        got = paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        ref = _reference(q, k_pool, v_pool, bt, seq_lens, scale=1.0).to(got.dtype)
        diff = (got.float() - ref.float()).abs().max().item()
        results.append((D, num_kv_heads, Hq, diff))
    ok = all(diff < TOL for *_, diff in results)
    return ok, " ".join(f"D={D} kvh={kvh} Hq={Hq} max|diff|={d:.2e}" for D, kvh, Hq, d in results)


def paged_decode_window() -> tuple[bool, str]:
    """Sliding-window probe: window < seq_len must match a windowed reference (last W keys)."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_decode_attention

    torch.manual_seed(0)
    results = []
    for D in [256, 512]:
        N, num_blocks, bs, W = 4, 256, 16, 50
        seq_lens = torch.tensor([10, 50, 80, 200], dtype=torch.int32, device="cuda")  # mix < and > W
        max_blocks = (int(seq_lens.max()) + bs - 1) // bs
        k_pool = torch.randn(num_blocks, 1, bs, D, dtype=torch.bfloat16, device="cuda")
        v_pool = torch.randn_like(k_pool)
        bt = torch.arange(N * max_blocks, dtype=torch.int32, device="cuda").reshape(N, max_blocks) % num_blocks
        q = torch.randn(N, 8, D, dtype=torch.bfloat16, device="cuda")
        got = paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0, window=W)
        ref = _reference(q, k_pool, v_pool, bt, seq_lens, scale=1.0, window=W).to(got.dtype)
        diff = (got.float() - ref.float()).abs().max().item()
        results.append((D, W, diff))
    ok = all(diff < TOL for *_, diff in results)
    return ok, " ".join(f"D={D} W={W} max|diff|={d:.2e}" for D, W, d in results)


def paged_decode_no_recompile() -> tuple[bool, str]:
    """Vary block-count widely: with the runtime loop bound only the first call per head_dim
    compiles. A constexpr bound would recompile per distinct count → ~1s spikes → the
    mixed-sweep stall."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_decode_attention

    torch.manual_seed(0)
    D, num_kv_heads, Hq, bs = 256, 1, 8, 16
    num_blocks = 4096
    k_pool = torch.randn(num_blocks, num_kv_heads, bs, D, dtype=torch.bfloat16, device="cuda")
    v_pool = torch.randn_like(k_pool)
    timings, slow = [], []
    for nb in [2, 8, 20, 40, 64, 96, 128]:  # distinct block-counts (would each recompile if constexpr)
        max_blocks = nb
        seq_lens = torch.tensor([nb * bs], dtype=torch.int32, device="cuda")
        bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").reshape(1, max_blocks) % num_blocks
        q = torch.randn(1, Hq, D, dtype=torch.bfloat16, device="cuda")
        paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        timings.append(f"{nb}:{dt:.1f}ms")
        if dt > 100:
            slow.append((nb, dt))
    return not slow, "block_count:call " + " ".join(timings)


def paged_prefill_parity() -> tuple[bool, str]:
    """Prefill kernel (multi-query, no gather) vs a gather + per-query causal-softmax reference:
    varied prefix/suffix lengths incl. a cold row and a 1-token cache-hit row, GQA 8:1."""
    import torch

    from inference_server.models.paged_attention_kernel import paged_prefill_attention

    torch.manual_seed(0)
    dev = "cuda"
    Hq, num_kv, D, bs = 8, 1, 64, 16        # GQA 8:1, head_dim 64 (kernel handles any D)
    prefix_lens = [20, 5, 33, 0]            # incl. a cold row (no prefix)
    suffix_lens = [4, 1, 7, 6]              # incl. a 1-token (cache-hit) row
    N = len(prefix_lens)
    Sq = max(suffix_lens)
    totals = [p + s for p, s in zip(prefix_lens, suffix_lens)]
    max_blocks = (max(totals) + bs - 1) // bs

    num_blocks = N * max_blocks + 1
    k_pool = torch.randn(num_blocks, num_kv, bs, D, device=dev)
    v_pool = torch.randn(num_blocks, num_kv, bs, D, device=dev)
    bt = torch.zeros(N, max_blocks, dtype=torch.int32, device=dev)

    # Per-seq full KV (prefix+suffix), scattered into blocks i*max_blocks + b.
    full_k = [torch.randn(totals[i], num_kv, D, device=dev) for i in range(N)]
    full_v = [torch.randn(totals[i], num_kv, D, device=dev) for i in range(N)]
    for i in range(N):
        for b in range((totals[i] + bs - 1) // bs):
            blk = i * max_blocks + b
            bt[i, b] = blk
            for s in range(bs):
                pos = b * bs + s
                if pos < totals[i]:
                    k_pool[blk, :, s, :] = full_k[i][pos]
                    v_pool[blk, :, s, :] = full_v[i][pos]

    q = torch.randn(N, Sq, Hq, D, device=dev)
    pl = torch.tensor(prefix_lens, dtype=torch.int32, device=dev)
    sl = torch.tensor(suffix_lens, dtype=torch.int32, device=dev)
    out = paged_prefill_attention(q, k_pool, v_pool, bt, pl, sl, scale=1.0)

    maxdiff = 0.0
    for i in range(N):
        fk, fv = full_k[i][:, 0, :], full_v[i][:, 0, :]   # [total, D] (num_kv=1)
        for j in range(suffix_lens[i]):
            qpos = prefix_lens[i] + j
            scores = q[i, j] @ fk[:qpos + 1].T            # [Hq, qpos+1]
            ref = torch.softmax(scores.float(), dim=-1) @ fv[:qpos + 1].float()
            maxdiff = max(maxdiff, (out[i, j].float() - ref).abs().max().item())
    return maxdiff < 1e-2, (f"max|kernel - ref|={maxdiff:.2e} "
                            f"(N={N}, prefix={prefix_lens}, suffix={suffix_lens})")


# Name → check. The gate runs all of them; each modal script runs its own subset.
CHECKS: dict[str, Callable[[], tuple[bool, str]]] = {
    "paged_decode_parity": paged_decode_parity,
    "paged_decode_window": paged_decode_window,
    "paged_decode_no_recompile": paged_decode_no_recompile,
    "paged_prefill_parity": paged_prefill_parity,
}
