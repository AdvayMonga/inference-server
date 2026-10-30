"""The decode-step megakernel (decode_step.cu): one persistent CUDA kernel for a whole decode step of batch <= 16
(embedding, 48 layers, final norm, lm_head), JIT-built with torch's cpp_extension on first use (nvcc, sm_90a).
`available()` is False without CUDA or nvcc. MEGAKERNEL=1 makes the backend use it for decode steps of up to
MEGAKERNEL_MAX_ROWS rows (default 2); MEGAKERNEL_FP8=1 streams the expert weights as fp8 e4m3 with one fp32 scale
per output channel (weight-only: activations stay bf16/fp32), copies made at load (~29 GB) when the engine's experts
are bf16; the Triton/torch path stays for prefill, wider batches and everything else. A module of the region overrides
package (`models/regions/`) may define `STEP(tokens, pos, bt, k_pools, v_pools, weights) -> (logits, next)`, a whole
step (the kernel layer's `region:embed:argmax` winner, K/V written in place, `weights` named as `Model.weights`):
`decode_step` then runs it in place of the extension."""

from __future__ import annotations

import functools
import importlib
import logging
import os
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MEGAKERNEL", "1") == "1"      # on where it builds (CUDA + nvcc); MEGAKERNEL=0 for the graphs
FP8 = os.environ.get("MEGAKERNEL_FP8", "0") == "1"      # measured 2026-10-18: no gain at batch 1 (the phases are latency-bound)
TMA = os.environ.get("MEGAKERNEL_TMA", "1") == "1"      # weights through a shared-memory ring (batch <= 2): 4.48 -> 3.71 ms a step
BPS = int(os.environ.get("MEGAKERNEL_BPS", "1"))        # blocks per SM where the shared memory allows (never with TMA)
NST = int(os.environ.get("MEGAKERNEL_NST", "3"))        # ring stages (3 x 32 KB, 4 x 24 KB, 6 x 16 KB, 12 x 8 KB)
CHUNK = int(os.environ.get("MEGAKERNEL_CHUNK", "128"))  # attention positions per item (256 -> 128, 2026-10-25)
PRIME = os.environ.get("MEGAKERNEL_PRIME", "1") == "1"  # a phase's first ring stages issued before its barrier
# Defaults from the 2026-10-25 lab A/B (researcher-20261025-151422-487eb4 vs the control researcher-20261025-144137-53382d):
# 16 warps, pre-issued stages and 128-position chunks cut the B=1 step 3.67 -> 3.58 ms (ctx 37), 4.22 -> 3.90 (ctx 2000)
# and single_stream TPOT p50 4.33 -> 4.10 ms.
THREADS = int(os.environ.get("MEGAKERNEL_THREADS", "512"))   # 256 or 512 threads a block (8 or 16 warps)
RING_KB = int(os.environ.get("MEGAKERNEL_RING_KB", "96"))      # the weight ring's size (up to 192 at batch <= 2)
NOMATH = os.environ.get("MEGAKERNEL_NOMATH", "0") == "1"       # timing only: the ring phases skip their dot products
BMAX = 16
H, NQ, NKV, D, I, E, K = 2048, 32, 4, 128, 768, 128, 8   # the kernel's compile-time shape (Qwen3-30B-A3B)
ROW = ("w.input_norm", "w.qkv", "w.q_norm", "w.k_norm", "w.o_proj", "w.post_norm", "w.router", "w.gate_up",
       "w.gate_up_scale", "w.down", "w.down_scale")      # the kernel's Layer, then the pools; stations' "w.*" names


def source(name: str = "decode_step.cu") -> str:
    """A kernel's source with common.cuh inlined: one self-contained file (load_inline, the kernel layer's start)."""
    d = Path(__file__).parent
    return (d / name).read_text().replace('#include "common.cuh"', (d / "common.cuh").read_text().replace("#pragma once", ""), 1)


@functools.lru_cache(maxsize=None)
def step_override(package: str = "inference_server.models.regions"):
    """The one `STEP` defined by a module of `package` (stations.overrides' package), or None."""
    pkg = importlib.import_module(package)
    found = [(m.name, getattr(mod, "STEP")) for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name)
             if hasattr(mod := importlib.import_module(f"{package}.{m.name}"), "STEP")]
    if len(found) > 1:
        raise ValueError(f"STEP defined by more than one module of {package}: {[n for n, _ in found]}")
    return found[0][1] if found else None


def _build(src: str, tag: str, kv8: bool, tma: bool, bps: int, include: list[str] = ()):
    """`src` built with the megakernel's build-time flags (decode_step.cu and the interpreter, interp.cu, alike)."""
    import sys
    from torch.utils.cpp_extension import load
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")   # ninja lives there
    return load(name=f"{tag}_kv8{int(kv8)}_tma{int(tma)}_bps{bps}_nst{NST}_c{CHUNK}_p{int(PRIME)}_t{THREADS}_r{RING_KB}_nm{int(NOMATH)}",
                sources=[str(Path(__file__).with_name(src))], extra_include_paths=list(include),
                # no --use_fast_math: the rope's cosf/sinf must be the precise ones (torch's are), the arguments
                # reach position x inv_freq ~ 3e4 rad; fast exp is spelled __expf where it is wanted
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo",
                                   f"-DMEGA_KV8={int(kv8)}", f"-DMEGA_TMA={int(tma)}", f"-DMEGA_BPS={bps}",
                                   f"-DMEGA_NST={NST}", f"-DMEGA_CHUNK={CHUNK}", f"-DMEGA_PRIME={int(PRIME)}",
                                   f"-DMEGA_THREADS={THREADS}", f"-DMEGA_RING_KB={RING_KB}", f"-DMEGA_NOMATH={int(NOMATH)}"],
                verbose=False)


@functools.lru_cache(maxsize=None)
def _ext(kv8: bool = False, tma: bool = False):
    """The extension built for fp8 (`kv8`) or bf16 KV pools and with (`tma`) or without the weight ring: build-time
    flags, so a kernel carries only its own path (both as runtime flags cost the base path 16%, 2026-10-20)."""
    return _build("decode_step.cu", "qwen3_decode_step", kv8, tma, 1 if tma else BPS)


def available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ext(os.environ.get("KV_CACHE_DTYPE", "").lower() == "fp8", TMA)     # the build the server will use
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
        self.weights: dict = {}                              # what a STEP override gets, BlameGraph kernels/step.py's names
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
            self.weights.update({f"{len(self._rows) - 1}.{n}": t for n, t in zip(ROW, ts)})
            self.scale = attn.scale
        if self._keep:
            torch.cuda.empty_cache()
        self.embed = m.embed_tokens.weight
        self.final_norm = _check(m.norm.weight, "final_norm", (H,))
        self.lm_head = model.lm_head.weight
        for name, t in (("embed", self.embed), ("lm_head", self.lm_head)):
            assert t.is_cuda and t.dtype == torch.bfloat16 and t.shape[1] == H and t.is_contiguous(), name
        self.inv_freq = m.rope.inv_freq.to(torch.float32).contiguous()
        from inference_server.models import stations
        self.weights.update(embed=self.embed, final_norm=self.final_norm, lm_head=self.lm_head, inv_freq=self.inv_freq,
                            cfg=stations.layer_config(model), n_layers=self.n_layers)
        self._scratch()

    @classmethod
    def from_weights(cls, w: dict, k_pools, v_pools) -> "Model":
        """A Model over a STEP's `weights` and pools, as `Model.weights` names them (no checks, no copies)."""
        self = cls.__new__(cls)
        self.weights, self.n_layers, self.device = w, w["n_layers"], w["embed"].device
        self.fp8, self.scale = w["0.w.gate_up_scale"] is not None, w["cfg"].scale
        self.embed, self.final_norm, self.lm_head, self.inv_freq = w["embed"], w["final_norm"], w["lm_head"], w["inv_freq"]
        self._rows = [[w[f"{i}.{n}"].data_ptr() if w[f"{i}.{n}"] is not None else 0 for n in ROW] + [0, 0]
                      for i in range(self.n_layers)]
        self._scratch()
        self.bind_pools([SimpleNamespace(k=k, v=v) for k, v in zip(k_pools, v_pools)])
        return self

    def _scratch(self) -> None:
        e = _ext(os.environ.get("KV_CACHE_DTYPE", "").lower() == "fp8", TMA)     # scratch sizes: any build; this one is the server's
        self.f32 = torch.empty(e.scratch_f32_elems(), dtype=torch.float32, device=self.device)
        self.bf16 = torch.empty(e.scratch_bf16_elems(), dtype=torch.bfloat16, device=self.device)
        self.i32 = torch.empty(e.scratch_i32_elems(), dtype=torch.int32, device=self.device)
        self.table: torch.Tensor | None = None

    def bind_pools(self, pools) -> None:
        """The paged K/V pools, one per layer ([blocks, NKV, 16, D] bf16 each), into the table."""
        assert len(pools) == self.n_layers and all(p is not None for p in pools), "one live pool per layer"
        self.kv8 = pools[0].k.dtype == torch.float8_e4m3fn            # KV_CACHE_DTYPE=fp8
        for row, p in zip(self._rows, pools):
            assert p.k.dtype in (torch.bfloat16, torch.float8_e4m3fn) and p.k.dtype == pools[0].k.dtype
            assert tuple(p.k.shape[1:]) == (NKV, 16, D) and p.k.is_contiguous() and p.v.is_contiguous()
            row[11], row[12] = p.k.data_ptr(), p.v.data_ptr()
        self.k_pools, self.v_pools = [p.k for p in pools], [p.v for p in pools]
        self.table = torch.tensor(self._rows, dtype=torch.int64, device=self.device)


_NO_TIMES = None


def decode_step(mdl: Model, tokens: torch.Tensor, pos: torch.Tensor, bt: torch.Tensor,
                times: torch.Tensor | None = None, stamp_layer: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """tokens [B] int64, pos [B] int64 (each row's length so far = the new token's position), bt [B, cols] int64
    block table (one for every layer), as the batched state keeps it. Writes every layer's K/V for the new token into the pools and
    returns (logits [B, V] bf16, greedy next token [B] int64: the argmax, lowest index on ties, like torch's).
    `times` [16] int64: globaltimer stamps (layer `stamp_layer`'s phases, after the layers, the end), debug; given
    [blocks, 16] (blocks = the grid, torch.cuda SM count), every block stamps its own row: the phases' skew. A STEP
    override (models/regions/) runs in place of the extension for a Model without stamps; a caller passing its own
    namespace (the kernel layer's baseline) gets the extension."""
    assert mdl.table is not None, "bind_pools() first"
    step = step_override()
    if step is not None and times is None and isinstance(mdl, Model):
        return step(tokens, pos, bt, mdl.k_pools, mdl.v_pools, mdl.weights)
    global _NO_TIMES
    if times is None:
        if _NO_TIMES is None:
            _NO_TIMES = torch.empty(0, dtype=torch.int64, device=tokens.device)
        times = _NO_TIMES
    ext = _ext(bool(mdl.kv8), TMA)                   # the ring is built in; the kernel uses it at batch <= 2
    return ext.decode_step(tokens.contiguous(), pos.contiguous(), bt.contiguous(), mdl.table, mdl.embed, mdl.final_norm, mdl.lm_head, mdl.inv_freq,
                           mdl.f32, mdl.bf16, mdl.i32, float(mdl.scale), mdl.fp8, times, stamp_layer, mdl.kv8, TMA)
