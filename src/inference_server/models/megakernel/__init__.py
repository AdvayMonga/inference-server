"""The decode-layer megakernel (decode_layer.cu): one persistent CUDA kernel per decoder layer for batch <= 16,
JIT-built with torch's cpp_extension on first use (nvcc, sm_90a). `available()` is False without CUDA or nvcc.
MEGAKERNEL=1 makes the backend use it for decode steps of up to MEGAKERNEL_MAX_ROWS rows (default 2: measured on
the H200, 68 us a layer at batch 1 against 122 for the torch path, 4.39 ms single-stream TPOT p99 against 5.87;
from batch 4 on both paths stream ~30 distinct experts' weights and the CUDA graphs win); the Triton/torch path
stays for prefill, wider batches and everything else."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MEGAKERNEL", "0") == "1"
BMAX = 16


@functools.lru_cache(maxsize=None)
def _ext():
    import sys
    from torch.utils.cpp_extension import load
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")   # ninja lives there
    src = Path(__file__).with_name("decode_layer.cu")
    return load(name="qwen3_decode_layer", sources=[str(src)],
                # no --use_fast_math: the rope's cosf/sinf must be the precise ones (torch's are), the arguments
                # reach position x inv_freq ~ 3e4 rad; fast exp is spelled __expf where it is wanted
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                verbose=False)


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ext()
    except Exception as e:  # noqa: BLE001
        logger.warning("megakernel not built (%s)", str(e)[:300])
        return False
    return True


class LayerWeights:
    """The layer's tensors the kernel reads, checked once: contiguous bf16, the fused q/k/v weight, bf16 experts."""

    def __init__(self, layer, rope_inv_freq: torch.Tensor):
        attn, mlp = layer.self_attn, layer.mlp
        assert hasattr(attn, "qkv_weight"), "fuse_qkv() first"
        assert not mlp.experts.fp8, "bf16 experts (fp8 experts: a later stage)"
        ts = dict(in_norm=layer.input_layernorm.weight, wqkv=attn.qkv_weight, q_norm=attn.q_norm.weight,
                  k_norm=attn.k_norm.weight, wo=attn.o_proj.weight, post_norm=layer.post_attention_layernorm.weight,
                  wr=mlp.gate.weight, gate_up=mlp.experts.gate_up_proj, down=mlp.experts.down_proj)
        for name, t in ts.items():
            assert t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous(), name
        self.t = ts
        self.inv_freq = rope_inv_freq.to(torch.float32).contiguous()
        self.scale = attn.scale


class Scratch:
    """Device scratch for one step (sized for BMAX rows), shared by every layer."""

    def __init__(self, device):
        e = _ext()
        self.f32 = torch.empty(e.scratch_f32_elems(), dtype=torch.float32, device=device)
        self.bf16 = torch.empty(e.scratch_bf16_elems(), dtype=torch.bfloat16, device=device)
        self.i32 = torch.empty(e.scratch_i32_elems(), dtype=torch.int32, device=device)


_NO_TIMES = None


def decode_layer(h: torch.Tensor, pos: torch.Tensor, block_table: torch.Tensor, pool_k: torch.Tensor,
                 pool_v: torch.Tensor, w: LayerWeights, scratch: Scratch, times: torch.Tensor | None = None) -> torch.Tensor:
    """h [B, 2048] bf16 (the residual stream), pos [B] int32 (each row's length so far = the new token's position),
    block_table [B, cols] int32 for this layer's pool. Writes the new token's K/V into the pool and returns the
    layer's output [B, 2048] bf16."""
    t = w.t
    global _NO_TIMES
    if times is None:
        if _NO_TIMES is None:
            _NO_TIMES = torch.empty(0, dtype=torch.int64, device=h.device)
        times = _NO_TIMES
    return _ext().decode_layer(h.contiguous(), pos, block_table.contiguous(), pool_k, pool_v, t["in_norm"], t["wqkv"],
                               t["q_norm"], t["k_norm"], t["wo"], t["post_norm"], t["wr"], t["gate_up"], t["down"],
                               w.inv_freq, scratch.f32, scratch.bf16, scratch.i32, float(w.scale), times)
