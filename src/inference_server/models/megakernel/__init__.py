"""The decode-step megakernel (decode_step.cu): one persistent CUDA kernel for a whole decode step of batch <= 16
(embedding, 48 layers, final norm, lm_head), JIT-built with torch's cpp_extension on first use (nvcc, sm_90a).
`available()` is False without CUDA or nvcc. MEGAKERNEL=1 makes the backend use it for decode steps of up to
MEGAKERNEL_MAX_ROWS rows (default 2); MEGAKERNEL_FP8=1 streams the expert weights as fp8 e4m3 with one fp32 scale
per output channel (weight-only: activations stay bf16/fp32), copies made at load (~29 GB) when the engine's experts
are bf16; the Triton/torch path stays for prefill, wider batches and everything else."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MEGAKERNEL", "1") == "1"      # on where it builds (CUDA + nvcc); MEGAKERNEL=0 for the graphs
FP8 = os.environ.get("MEGAKERNEL_FP8", "0") == "1"      # measured 2026-10-18: no gain at batch 1 (the phases are latency-bound)
BMAX = 16
H, NQ, NKV, D, I, E, K = 2048, 32, 4, 128, 768, 128, 8   # the kernel's compile-time shape (Qwen3-30B-A3B)


@functools.lru_cache(maxsize=None)
def _ext():
    import sys
    from torch.utils.cpp_extension import load
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")   # ninja lives there
    src = Path(__file__).with_name("decode_step.cu")
    return load(name="qwen3_decode_step", sources=[str(src)],
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


def _check(t: torch.Tensor, name: str, shape: tuple, dtype=torch.bfloat16) -> torch.Tensor:
    assert t.is_cuda and t.dtype == dtype and t.is_contiguous() and tuple(t.shape) == shape, \
        f"{name}: want {dtype} {shape} contiguous on CUDA, got {t.dtype} {tuple(t.shape)}"
    return t


class Model:
    """Everything the kernel reads, checked and packed once: the per-layer pointer table (13 pointers a layer, the
    kernel's `Layer`), fp8 copies of the experts when asked for and the engine's are bf16, the embedding, final norm
    and lm_head, device scratch. Build it before the KV pools are sized (the fp8 copies take ~29 GB), then
    `bind_pools`."""

    def __init__(self, model, device, fp8: bool = FP8):
        from inference_server.models.qwen3_moe import quantize_rowwise
        m = model.model
        fp8 = fp8 or bool(m.layers[0].mlp.experts.fp8)        # fp8 experts (CUSTOM_BACKEND_QUANT=fp8) are used as they are
        self.device, self.fp8 = device, fp8
        self.n_layers = len(m.layers)
        self._keep: list[torch.Tensor] = []                  # tensors the table points to that the model does not own
        self._rows: list[list[int]] = []
        for layer in m.layers:
            attn, mlp, ex = layer.self_attn, layer.mlp, layer.mlp.experts
            assert hasattr(attn, "qkv_weight"), "fuse_qkv() first"
            if fp8:
                if ex.fp8:
                    gu, gus, dn, dns = ex.gate_up_proj_fp8, ex.gate_up_proj_scale, ex.down_proj_fp8, ex.down_proj_scale
                else:
                    gu, gus = quantize_rowwise(ex.gate_up_proj.data)
                    dn, dns = quantize_rowwise(ex.down_proj.data)
                    gus, dns = gus.contiguous(), dns.contiguous()
                    self._keep += [gu, gus, dn, dns]
                _check(gu, "gate_up fp8", (E, 2 * I, H), torch.float8_e4m3fn)
                _check(gus, "gate_up scale", (E, 2 * I), torch.float32)
                _check(dn, "down fp8", (E, H, I), torch.float8_e4m3fn)
                _check(dns, "down scale", (E, H), torch.float32)
            else:
                assert not ex.fp8, "bf16 experts for the bf16 kernel (MEGAKERNEL_FP8=0 with CUSTOM_BACKEND_QUANT=fp8)"
                gu, dn = _check(ex.gate_up_proj, "gate_up", (E, 2 * I, H)), _check(ex.down_proj, "down", (E, H, I))
                gus = dns = None
            ts = [_check(layer.input_layernorm.weight, "in_norm", (H,)), _check(attn.qkv_weight, "wqkv", (NQ * D + 2 * NKV * D, H)),
                  _check(attn.q_norm.weight, "q_norm", (D,)), _check(attn.k_norm.weight, "k_norm", (D,)),
                  _check(attn.o_proj.weight, "wo", (H, NQ * D)), _check(layer.post_attention_layernorm.weight, "post_norm", (H,)),
                  _check(mlp.gate.weight, "wr", (E, H)), gu, gus, dn, dns]
            self._rows.append([t.data_ptr() if t is not None else 0 for t in ts] + [0, 0])
            self.scale = attn.scale
        if self._keep:
            torch.cuda.empty_cache()
        self.embed = m.embed_tokens.weight
        self.final_norm = _check(m.norm.weight, "final_norm", (H,))
        self.lm_head = model.lm_head.weight
        for name, t in (("embed", self.embed), ("lm_head", self.lm_head)):
            assert t.is_cuda and t.dtype == torch.bfloat16 and t.shape[1] == H and t.is_contiguous(), name
        self.inv_freq = m.rope.inv_freq.to(torch.float32).contiguous()
        e = _ext()
        self.f32 = torch.empty(e.scratch_f32_elems(), dtype=torch.float32, device=device)
        self.bf16 = torch.empty(e.scratch_bf16_elems(), dtype=torch.bfloat16, device=device)
        self.i32 = torch.empty(e.scratch_i32_elems(), dtype=torch.int32, device=device)
        self.table: torch.Tensor | None = None

    def bind_pools(self, pools) -> None:
        """The paged K/V pools, one per layer ([blocks, NKV, 16, D] bf16 each), into the table."""
        assert len(pools) == self.n_layers and all(p is not None for p in pools), "one live pool per layer"
        for row, p in zip(self._rows, pools):
            assert p.k.dtype == torch.bfloat16 and tuple(p.k.shape[1:]) == (NKV, 16, D) and p.k.is_contiguous() and p.v.is_contiguous()
            row[11], row[12] = p.k.data_ptr(), p.v.data_ptr()
        self.table = torch.tensor(self._rows, dtype=torch.int64, device=self.device)


_NO_TIMES = None


def decode_step(mdl: Model, tokens: torch.Tensor, pos: torch.Tensor, bt: torch.Tensor,
                times: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens [B] int64, pos [B] int64 (each row's length so far = the new token's position), bt [layers, B, cols]
    int64 block tables, as the batched state keeps them. Writes every layer's K/V for the new token into the pools and
    returns (logits [B, V] bf16, greedy next token [B] int64: the argmax, lowest index on ties, like torch's).
    `times` [16] int64: globaltimer stamps (layer 0's phases, after the layers, the end), debug."""
    assert mdl.table is not None, "bind_pools() first"
    global _NO_TIMES
    if times is None:
        if _NO_TIMES is None:
            _NO_TIMES = torch.empty(0, dtype=torch.int64, device=tokens.device)
        times = _NO_TIMES
    return _ext().decode_step(tokens.contiguous(), pos.contiguous(), bt.contiguous(), mdl.table, mdl.embed, mdl.final_norm, mdl.lm_head, mdl.inv_freq,
                              mdl.f32, mdl.bf16, mdl.i32, float(mdl.scale), mdl.fp8, times)
