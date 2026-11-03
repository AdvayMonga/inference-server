"""Piecewise CUDA graphs of the packed prefill forward (`prefill_chunk_batch`), at token buckets.

An eager packed prefill launches ~600 kernels through Python (Triton's launcher, torch modules, FlashInfer's wrapper),
~0.7 ms of host time per layer: every step of 129..2,048 tokens cost the same ~36 ms while the GPU waited (H200,
2026-11-02: 256 tokens 34.9 ms, 2,048 tokens 36.7 ms). Here each layer boundary is one graph (the previous layer's
o_proj, residual norm and MoE, then this layer's input norm, QKV, head norm + rotary and K/V store) and FlashInfer's
attention, which needs a host plan, runs eagerly between them. A step pads its tokens to the bucket: the padding's K/V
lands in the scratch block and its rows are never read. The kernels are the eager path's own, in the same order."""
from __future__ import annotations

import logging
import os
import time

import torch

from inference_server.models import fused

logger = logging.getLogger(__name__)

BUCKETS = (256, 384, 512, 768, 1024, 1536, 2048)
ENABLED = os.environ.get("CUSTOM_BACKEND_PREFILL_PIECEWISE", "1") == "1"


class PrefillGraphs:
    """One set of layer-boundary graphs per bucket, captured on first use, all buckets in one graph memory pool."""

    def __init__(self, model, pools, device: torch.device, scratch: int):
        self.lm, self.m = model, model.model
        self.pools, self.dev, self.scratch = pools, device, scratch
        self.n_layers = len(self.m.layers)
        at = self.m.layers[0].self_attn
        self.hq, self.hkv, self.d, self.scale = at.num_q_heads, at.num_kv_heads, at.head_dim, at.scale
        self.fp8_q = pools[0].k.dtype == torch.float8_e4m3fn
        self.sets: dict[int, dict] = {}
        self.failed: set[int] = set()
        self._mempool = None

    @staticmethod
    def bucket(n: int) -> int | None:
        return next((b for b in BUCKETS if b >= n), None)

    # --- the pieces (each is the eager layer's code, in its order) ---

    def _attn_in(self, L: int, x: torch.Tensor, cos, sin, st: dict) -> None:
        at, N = self.m.layers[L].self_attn, x.shape[1]
        q, k, v = at._qkv(x)
        q = fused.rmsnorm_rope(q.view(1, N, self.hq, self.d), at.q_norm.weight, cos, sin, at.q_norm.eps)
        k = fused.rmsnorm_rope(k.view(1, N, self.hkv, self.d), at.k_norm.weight, cos, sin, at.k_norm.eps)
        v = v.view(1, N, self.hkv, self.d)
        pool = self.pools[L]
        fused.kv_store_rows(pool.k, pool.v, st["bids"], st["slots"], k[0], v[0])
        st["q"].copy_(fused.cast_fp8(q[0]) if self.fp8_q else q[0])

    def _attn_out(self, L: int, h: torch.Tensor, st: dict) -> torch.Tensor:
        layer, N = self.m.layers[L], h.shape[1]
        a = layer.self_attn.o_proj(st["o"].view(1, N, -1))
        pn = layer.post_attention_layernorm
        h, n = fused.add_rmsnorm(h, a, pn.weight, pn.eps)
        return layer.mlp(n, None, residual=h)

    def _piece(self, i: int, st: dict) -> None:
        """Piece i: i = 0 embeds and enters layer 0; 0 < i < n_layers leaves layer i-1 and enters layer i; the last
        leaves the last layer. The residual stream stays in st["h"]."""
        if i == 0:
            h, cos, sin = fused.embed_rope(st["ids"], self.m.embed_tokens.weight, self.m.rope.inv_freq, st["pos"])
            st["cos"], st["sin"] = cos, sin
        else:
            h = self._attn_out(i - 1, st["h"], st)
        st["h"].copy_(h)
        if i < self.n_layers:
            self._attn_in(i, self.m.layers[i].input_layernorm(st["h"]), st["cos"], st["sin"], st)

    # --- capture ---

    def _capture(self, N: int) -> dict:
        t0 = time.perf_counter()
        dev, H = self.dev, self.m.hidden_size
        st = {"ids": torch.zeros(1, N, dtype=torch.long, device=dev),
              "pos": torch.zeros(1, N, dtype=torch.long, device=dev),
              "bids": torch.full((N,), self.scratch, dtype=torch.long, device=dev),
              "slots": torch.zeros(N, dtype=torch.long, device=dev),
              "q": torch.zeros(N, self.hq, self.d, dtype=torch.float8_e4m3fn if self.fp8_q else torch.bfloat16,
                               device=dev),
              "o": torch.zeros(N, self.hq, self.d, dtype=torch.bfloat16, device=dev),
              "h": torch.zeros(1, N, H, dtype=torch.bfloat16, device=dev)}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                       # warm every kernel (Triton JIT, cuBLAS handles) first
            for i in range(self.n_layers + 1):
                self._piece(i, st)
        torch.cuda.current_stream().wait_stream(side)
        if self._mempool is None:
            self._mempool = torch.cuda.graph_pool_handle()
        graphs = []
        for i in range(self.n_layers + 1):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._mempool):
                self._piece(i, st)
            graphs.append(g)
        st["graphs"] = graphs
        logger.info("prefill piecewise graphs: bucket %d captured in %.1f s", N, time.perf_counter() - t0)
        return st

    def get(self, N: int) -> dict | None:
        if N in self.failed:
            return None
        st = self.sets.get(N)
        if st is None:
            if not self.sets:                               # all buckets at once, largest first (one shared pool)
                for b in sorted(BUCKETS, reverse=True):
                    try:
                        self.sets[b] = self._capture(b)
                    except Exception:
                        logger.exception("prefill piecewise capture failed (bucket %d): that bucket stays eager", b)
                        self.failed.add(b)
                        torch.cuda.synchronize()
            st = self.sets.get(N)
        return st

    # --- a step ---

    def forward(self, st: dict, ctx, ids: torch.Tensor, pos: torch.Tensor, logits_index: torch.Tensor) -> torch.Tensor:
        """The packed wave in `ctx` (its blocks allocated here) through the bucket's graphs: logits [1, n, V] at
        `logits_index`, as the eager forward returns them."""
        from inference_server.models import prefill_flashinfer as fi
        from inference_server.models.paged_kv_cache import to_device
        T = ids.numel()
        ctx._ensure_blocks()
        starts = [0]
        for n in ctx._suffix[:-1]:
            starts.append(starts[-1] + n)
        bs = self.pools[0].block_size
        bids, slots = fused.packed_slots(ctx.prefix_lens, ctx.suffix_lens, to_device(starts, self.dev, torch.int32),
                                         ctx._table(), bs, T, max(ctx._suffix))
        st["ids"][0, :T].copy_(ids.view(-1))
        st["pos"][0, :T].copy_(pos.view(-1))
        st["bids"][:T].copy_(bids)
        st["bids"][T:].fill_(self.scratch)
        st["slots"][:T].copy_(slots)
        w = fi._wrapper(self.dev, "fa3" if self.fp8_q else None)
        qo, kv, last, nb = fi.plan_inputs(ctx._prefix, ctx._suffix, bs)
        fi._PLANNED.pop(id(w), None)                        # `attend`'s plan is this wrapper's no longer
        fi.plan(w, qo, kv, fi.kv_indices([c.blocks for c in ctx.caches], nb, self.dev), last, self.hq, self.hkv, self.d,
               bs, causal=True, sm_scale=self.scale, q_data_type=st["q"].dtype, kv_data_type=self.pools[0].k.dtype,
               o_data_type=torch.bfloat16)
        q, o = st["q"][:T], st["o"][:T]
        for i, g in enumerate(st["graphs"]):
            g.replay()
            if i < self.n_layers:
                L = i
                w.run(q, (self.pools[L].k, self.pools[L].v), out=o)
        for cache, n in zip(ctx.caches, ctx._suffix):       # what each layer's append records on the host
            for L, p in enumerate(self.pools):
                if p is not None:
                    cache.seq_lens[L] += n
        norm = self.m.norm
        return self.lm.lm_head(fused.rmsnorm_rows(st["h"][:, :T], logits_index, norm.weight, norm.eps))
