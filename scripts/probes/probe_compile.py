"""Feasibility probe: does torch.compile trace our custom Gemma 4 forward cleanly?

Graph breaks are device-independent (Dynamo traces Python), so CPU results transfer
to CUDA. We DON'T measure perf here (CPU Inductor ≠ Triton) — only:
  (a) how many graph breaks each path has + why,
  (b) numerical agreement compiled-vs-eager.

Paths probed:
  1. square prefill  model(input_ids)                       — parity path, no cache
  2. paged prefill   model(suffix, kv_cache=PagedKVCache)   — real prefill
  3. CPU batched decode  model(tok, kv_cache=BatchedPagedKVCache, attn_mask)
     (the CUDA decode path uses paged_ctx + Triton kernel — can't run on CPU, reasoned separately)
"""
from __future__ import annotations
import torch
from inference_server.models.gemma4 import GemmaForCausalLM
from inference_server.models.paged_kv_cache import make_pools_for_gemma, PagedKVCache, BatchedPagedKVCache

DEV = torch.device("cpu")
torch.manual_seed(0)

print("loading model (cpu, bf16)…")
m = GemmaForCausalLM.from_hf("google/gemma-4-E2B-it", dtype=torch.bfloat16).to(DEV).eval()

def explain(tag, fn):
    print(f"\n===== {tag} =====")
    try:
        exp = torch._dynamo.explain(fn)()
        print(f"graph_count={exp.graph_count}  graph_break_count={exp.graph_break_count}  op_count={exp.op_count}")
        for i, b in enumerate(exp.break_reasons):
            print(f"  break[{i}]: {b.reason}")
    except Exception as e:
        print(f"EXPLAIN FAILED: {type(e).__name__}: {e}")

@torch.no_grad()
def run():
    ids = torch.tensor([[2, 100, 200, 300, 400]], device=DEV)

    explain("1. square prefill  model(ids)", lambda: m(ids))

    def paged_prefill():
        c = PagedKVCache(pools=make_pools_for_gemma(m, num_blocks_per_pool=64, block_size=16))
        return m(ids, kv_cache=c)
    explain("2. paged prefill", paged_prefill)

    def cpu_decode():
        pools = make_pools_for_gemma(m, num_blocks_per_pool=64, block_size=16)
        rows = [PagedKVCache(pools=pools), PagedKVCache(pools=pools)]
        for c in rows:
            m(ids, kv_cache=c)              # seed each row
        ctx = BatchedPagedKVCache(rows)
        n = len(rows)
        lmax = max(r.seq_len for r in rows)
        mask = torch.ones(n, 1, 1, lmax + 1, dtype=torch.bool, device=DEV)
        tok = torch.tensor([[1], [1]], device=DEV)
        pos = torch.tensor([[ids.shape[1]], [ids.shape[1]]], device=DEV)
        return m(tok, position_ids=pos, kv_cache=ctx, attn_mask=mask)
    explain("3. cpu batched decode", cpu_decode)

run()
print("\nprobe done.")
