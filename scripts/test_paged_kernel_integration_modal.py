"""End-to-end GPU integration test for the paged-attention kernel decode path (M2.3).

Loads the real Gemma 4 model on an A10G and checks that scheduler-style batched decode
via the Triton kernel (CUDA path) produces the SAME generated tokens as the portable
gather+SDPA path, for a multi-row batch of different prompt lengths. Run:

    venv/bin/modal run scripts/test_paged_kernel_integration_modal.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("inference_server")
)

app = modal.App("paged-kernel-integration", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL = "google/gemma-4-E2B-it"
PROMPTS = [
    [2, 651, 6037, 576, 6081, 603, 8, 235248],  # 8 tokens
    [2, 1841, 603, 573],                          # 4 tokens
    [2, 23274, 1432, 573, 2778, 692, 6],          # 7 tokens
]
STEPS = 12


@app.function(gpu="A10G", volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=600)
def run():
    import torch

    from inference_server.models.gemma4 import GemmaForCausalLM
    from inference_server.models.paged_kv_cache import (
        BatchedPagedKVCache, PagedKVCache, make_pools_for_gemma,
    )

    model = GemmaForCausalLM.from_hf(MODEL, dtype=torch.bfloat16).to("cuda").eval()

    # --- Single-step logit diagnostic: identical inputs, one decode step, no cascade. ---
    # Distinguishes a real bug (large logit diff) from a greedy-argmax flip (tiny diff).
    def prefill_caches():
        pools = make_pools_for_gemma(model, num_blocks_per_pool=256, block_size=16)
        rows, cur = [], []
        with torch.no_grad():
            for p in PROMPTS:
                c = PagedKVCache(pools=pools)
                logits = model(torch.tensor([p], device="cuda"), kv_cache=c)
                rows.append(c)
                cur.append(int(logits[:, -1, :].argmax(-1).item()))
        return pools, rows, cur

    with torch.no_grad():
        _, rk, cur = prefill_caches()
        _, rg, _ = prefill_caches()
        tokens = torch.tensor([[t] for t in cur], device="cuda")
        posk = torch.tensor([[r.seq_len] for r in rk], device="cuda")
        posg = torch.tensor([[r.seq_len] for r in rg], device="cuda")
        lk = model(tokens, position_ids=posk, paged_ctx=BatchedPagedKVCache(rk))
        seq_lens = [r.seq_len for r in rg]
        lmax = max(seq_lens)
        mask = torch.zeros(len(rg), 1, 1, lmax + 1, dtype=torch.bool, device="cuda")
        for i, L in enumerate(seq_lens):
            mask[i, 0, 0, lmax - L:] = True
        lg = model(tokens, position_ids=posg, kv_cache=BatchedPagedKVCache(rg), attn_mask=mask)
        for i in range(len(PROMPTS)):
            d = (lk[i, -1].float() - lg[i, -1].float()).abs().max().item()
            print(f"row {i} (len {len(PROMPTS[i])}): max|logit diff|={d:.4e}  "
                  f"argmax kernel={int(lk[i,-1].argmax())} gather={int(lg[i,-1].argmax())}")
        for c in rk + rg:
            c.free_all()

    def decode_sequence(use_kernel: bool):
        """Prefill all prompts, then greedily decode STEPS tokens for each row in lockstep,
        using either the kernel path or the gather+SDPA path for every decode step."""
        pools = make_pools_for_gemma(model, num_blocks_per_pool=256, block_size=16)
        rows, cur = [], []
        with torch.no_grad():
            for p in PROMPTS:
                c = PagedKVCache(pools=pools)
                logits = model(torch.tensor([p], device="cuda"), kv_cache=c)
                rows.append(c)
                cur.append(int(logits[:, -1, :].argmax(-1).item()))
            gen = [[t] for t in cur]
            for _ in range(STEPS - 1):
                ctx = BatchedPagedKVCache(rows)
                tokens = torch.tensor([[t] for t in cur], device="cuda")
                pos = torch.tensor([[r.seq_len] for r in rows], device="cuda")
                if use_kernel:
                    logits = model(tokens, position_ids=pos, paged_ctx=ctx)
                else:
                    seq_lens = [r.seq_len for r in rows]
                    lmax = max(seq_lens)
                    mask = torch.zeros(len(rows), 1, 1, lmax + 1, dtype=torch.bool, device="cuda")
                    for i, L in enumerate(seq_lens):
                        mask[i, 0, 0, lmax - L:] = True
                    logits = model(tokens, position_ids=pos, kv_cache=ctx, attn_mask=mask)
                cur = [int(logits[i, -1, :].argmax(-1).item()) for i in range(len(rows))]
                for i, t in enumerate(cur):
                    gen[i].append(t)
        for c in rows:
            c.free_all()
        return gen

    kernel_gen = decode_sequence(use_kernel=True)
    ref_gen = decode_sequence(use_kernel=False)

    match = kernel_gen == ref_gen
    for i, (k, r) in enumerate(zip(kernel_gen, ref_gen)):
        print(f"row {i}: kernel={k}")
        print(f"row {i}: gather={r}  {'OK' if k == r else 'DIFF'}")
    print("INTEGRATION", "OK" if match else "FAIL")
    return match


@app.local_entrypoint()
def main():
    print("kernel integration:", run.remote())
