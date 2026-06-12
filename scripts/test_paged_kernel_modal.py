"""Isolated GPU parity test for the Triton paged-decode kernel (M2.3).

Validates the kernel against a pure-torch reference on synthetic data — no model — so
kernel correctness is proven before it's wired into the forward. Run:

    venv/bin/modal run scripts/test_paged_kernel_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)

app = modal.App("paged-kernel-test", image=image)


def _reference(q, k_pool, v_pool, block_tables, seq_lens, scale):
    """Pure-torch paged attention: gather each seq's real K/V, full softmax."""
    import torch

    N, Hq, D = q.shape
    num_kv_heads = k_pool.shape[1]
    bs = k_pool.shape[2]
    group = Hq // num_kv_heads
    out = torch.empty(N, Hq, D, dtype=torch.float32, device=q.device)
    for i in range(N):
        L = int(seq_lens[i].item())
        nb = (L + bs - 1) // bs
        ks, vs = [], []
        for b in range(nb):
            blk = int(block_tables[i, b].item())
            ks.append(k_pool[blk])  # [num_kv_heads, bs, D]
            vs.append(v_pool[blk])
        K = torch.cat(ks, dim=1)[:, :L, :].float()  # [num_kv_heads, L, D]
        V = torch.cat(vs, dim=1)[:, :L, :].float()
        K = K.repeat_interleave(group, dim=0)        # [Hq, L, D]
        V = V.repeat_interleave(group, dim=0)
        qi = q[i].float()                            # [Hq, D]
        scores = (qi[:, None, :] * K).sum(-1) * scale  # [Hq, L]
        attn = torch.softmax(scores, dim=-1)
        out[i] = (attn[:, :, None] * V).sum(dim=1)   # [Hq, D]
    return out


@app.function(gpu="A10G")
def run():
    import torch

    from inference_server.models.paged_attention_kernel import paged_decode_attention

    torch.manual_seed(0)
    results = []
    # (head_dim, num_kv_heads, Hq) — sliding (256) and full (512), GQA 8:1
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
        print(f"D={D} kvh={num_kv_heads} Hq={Hq}  max|diff|={diff:.4e}")

    ok = all(diff < 5e-2 for *_, diff in results)
    print("PARITY", "OK" if ok else "FAIL", results)

    # Recompilation probe: vary block-count widely. With the runtime loop bound, only the
    # first call per head_dim compiles; the rest must be fast (a constexpr bound would
    # recompile per distinct block-count → ~1s spikes → the mixed-sweep stall).
    import time as _time
    D, num_kv_heads, Hq, bs = 256, 1, 8, 16
    num_blocks = 4096
    k_pool = torch.randn(num_blocks, num_kv_heads, bs, D, dtype=torch.bfloat16, device="cuda")
    v_pool = torch.randn_like(k_pool)
    slow = []
    for nb in [2, 8, 20, 40, 64, 96, 128]:  # distinct block-counts (would each recompile if constexpr)
        max_blocks = nb
        seq_lens = torch.tensor([nb * bs], dtype=torch.int32, device="cuda")
        bt = torch.arange(max_blocks, dtype=torch.int32, device="cuda").reshape(1, max_blocks) % num_blocks
        q = torch.randn(1, Hq, D, dtype=torch.bfloat16, device="cuda")
        paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        paged_decode_attention(q, k_pool, v_pool, bt, seq_lens, scale=1.0)
        torch.cuda.synchronize()
        dt = (_time.perf_counter() - t0) * 1e3
        print(f"  block_count={nb:4d}  call={dt:.1f} ms")
        if dt > 100:
            slow.append((nb, dt))
    no_recompile = not slow
    print("NO_RECOMPILE", "OK" if no_recompile else f"FAIL {slow}")
    return ok and no_recompile


@app.local_entrypoint()
def main():
    print("kernel parity:", run.remote())
