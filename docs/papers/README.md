# papers/ — reading notes

Notes on the papers this engine borrows from, each with the idea, what it buys, and what (if
anything) it changed here.

| note | idea | status in this engine |
|---|---|---|
| `vllm-paged-attention.md` | block-paged KV, PagedAttention | implemented (`models/paged_kv_cache.py`, Triton kernels) |
| `flashattention.md` | IO-aware tiled attention | tiled prefill kernel built, measured, rejected end-to-end (head_dim 512) |
| `batchllm.md` | prefix-sharing-aware batching | radix prefix cache; wave grouping measured inert |
| `elis.md` | length-predictive scheduling | not pursued — TTFT tail is prefill compute, not queueing |
| `speculative-decoding.md` | draft-and-verify decode | not pursued yet |
| `loki.md` | low-rank keys for sparse attention | not pursued |
| `tokenweave.md` | compute/communication overlap | single-GPU engine; not applicable yet |
