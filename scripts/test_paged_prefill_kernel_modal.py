"""Isolated correctness test for the paged PREFILL kernel (multi-query, no gather).

Builds random per-sequence KV in scattered blocks (varied prefix/suffix lengths + GQA), runs
paged_prefill_attention, and compares against a gather + per-query causal-softmax reference.

    venv/bin/modal run scripts/test_paged_prefill_kernel_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)
app = modal.App("paged-prefill-kernel-test", image=image)


@app.function(gpu="A10G", timeout=600)
def run():
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

    print(f"\nmax|kernel - ref| = {maxdiff:.2e}  (N={N}, prefix={prefix_lens}, suffix={suffix_lens})")
    print("PAGED PREFILL KERNEL", "OK" if maxdiff < 1e-2 else "FAIL")


@app.local_entrypoint()
def main():
    run.remote()
