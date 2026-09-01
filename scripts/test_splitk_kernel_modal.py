"""Split-K decode attention: correctness vs the single-program kernel, then latency.

Correctness gate: split-K must match the non-split kernel on the SAME inputs. They compute the
same quantity by a different reduction order, so the bar is tight-but-not-bitwise. Edge cases
that matter: sequences SHORTER than the split count (some splits get no blocks and must
contribute a clean zero via m=-inf), and sliding-window rows (the split range starts at the
window's first live block, not block 0).

    PYTHONPATH=$PWD/src venv/bin/modal run --detach scripts/test_splitk_kernel_modal.py
"""

import os

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("splitk-kernel-test", image=image)

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
GROUP = 4          # Hq = 8
D = 256


@app.function(gpu=os.environ.get("BENCH_GPU", "A100-80GB"), timeout=1800)
def run():
    import time

    import torch

    from inference_server.models.paged_attention_kernel import (
        _paged_decode_kernel,
        decode_splits,
        paged_decode_attention,
    )

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    Hq = NUM_KV_HEADS * GROUP
    num_blocks = 4096
    k_pool = torch.randn(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, D, device=dev, dtype=torch.bfloat16)
    v_pool = torch.randn(num_blocks, NUM_KV_HEADS, BLOCK_SIZE, D, device=dev, dtype=torch.bfloat16)

    def no_split(q, bt, sl, window):
        """Force the original single-program path for the reference."""
        N = q.shape[0]
        out = torch.empty(N, Hq, D, dtype=torch.float32, device=dev)
        qc = q.contiguous()
        _paged_decode_kernel[(N, Hq)](
            qc, k_pool, v_pool, bt, sl, out,
            qc.stride(0), qc.stride(1),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            out.stride(0), out.stride(1),
            bt.stride(0), 1.0, window,
            GROUP=GROUP, BLOCK_SIZE=BLOCK_SIZE, D=D,
        )
        return out.to(q.dtype)

    def make(N, lens):
        q = torch.randn(N, Hq, D, device=dev, dtype=torch.bfloat16)
        maxb = (max(lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
        bt = torch.randint(0, num_blocks, (N, maxb), device=dev, dtype=torch.int32)
        sl = torch.tensor(lens, device=dev, dtype=torch.int32)
        return q, bt, sl

    print("=== correctness: split-K vs single-program kernel ===", flush=True)
    ok = True
    cases = [
        ("n=1 long",        1,  [2048],                     1 << 30),
        ("n=1 short",       1,  [3],                        1 << 30),   # fewer keys than splits
        ("n=2 mixed",       2,  [17, 1024],                 1 << 30),
        ("n=4 mixed",       4,  [1, 33, 512, 2047],         1 << 30),
        ("n=4 window=512",  4,  [64, 700, 1500, 2048],      512),
        ("n=8 window=512",  8,  [7, 16, 129, 640, 900, 1200, 1800, 2048], 512),
        ("n=1 exactly1blk", 1,  [16],                       1 << 30),
        ("n=2 win>len",     2,  [5, 9],                     512),
    ]
    for name, N, lens, window in cases:
        q, bt, sl = make(N, lens)
        ref = no_split(q, bt, sl, window)
        got = paged_decode_attention(q, k_pool, v_pool, bt, sl, scale=1.0, window=window)
        sp = decode_splits(N, Hq)
        diff = float((ref.float() - got.float()).abs().max())
        rel = diff / max(float(ref.float().abs().max()), 1e-6)
        good = rel < 2e-2 and torch.isfinite(got.float()).all()
        ok &= good
        print(f"  {name:<18} splits={sp}  maxdiff={diff:.6f}  rel={rel:.2e}  "
              f"{'OK' if good else 'FAIL'}", flush=True)
    print(f"correctness: {'PASS' if ok else 'FAIL'}\n", flush=True)

    print("=== latency: ms per decode-attention call (L=2048, full attention) ===", flush=True)
    print(f"{'N':>5} {'splits':>7} {'split-K':>9} {'single':>9} {'speedup':>8}", flush=True)
    for N in (1, 2, 4, 8, 16, 32, 64):
        q, bt, sl = make(N, [2048] * N)

        def timeit(fn):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 100 * 1000

        a = timeit(lambda: paged_decode_attention(q, k_pool, v_pool, bt, sl, 1.0, 1 << 30))
        b = timeit(lambda: no_split(q, bt, sl, 1 << 30))
        print(f"{N:>5} {decode_splits(N, Hq):>7} {a:>9.4f} {b:>9.4f} {b / a:>7.2f}x", flush=True)
    return bool(ok)
