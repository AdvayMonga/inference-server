"""The decode step as an instruction interpreter (interp.cu): one persistent kernel whose blocks each run a queue of
instructions, ordered by dependency counters instead of grid-wide barriers. Same inputs and outputs as
`megakernel.decode_step`; taken only where the routing table (routing.py) names the path "framework".

A program is a schedule plus its instructions' device code: a module with `schedule(B, blocks, n_layers, vocab, fp8)
-> Schedule` (Python, run at load for every batch size) and `OPS`, the header interp.cu includes (it defines
`run_op` and `program_smem`). `program/` is the default, today's megakernel phases as instructions; a search passes
its own module to `Framework`. Weights are the `megakernel.Model`'s device tensors: nothing is loaded here."""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

FIELDS = 16          # ints per instruction (interp.cu's Instr): op, layer, wait0, need0, wait1, need1, signal, arg[9]
ARGS = 9


@dataclass
class Schedule:
    instrs: torch.Tensor       # [N, FIELDS] int32; queue j = instrs[offsets[j]:offsets[j + 1]]
    offsets: torch.Tensor      # [blocks + 1] int32
    n_counters: int

    def to(self, device) -> "Schedule":
        return Schedule(self.instrs.to(device), self.offsets.to(device), self.n_counters)

    def queues(self) -> list[list[list[int]]]:
        """Each block's instructions as lists of FIELDS ints."""
        rows, off = self.instrs.tolist(), self.offsets.tolist()
        return [rows[off[j]:off[j + 1]] for j in range(len(off) - 1)]


class Builder:
    """Per-block queues of instructions over named counters. A wait is on a counter reaching the number of
    instructions that signal it, counted when the schedule is built."""

    def __init__(self, blocks: int):
        self.queues: list[list[tuple]] = [[] for _ in range(blocks)]
        self.ids: dict[str, int] = {}
        self.signals: list[int] = []

    def counter(self, name: str) -> int:
        if name not in self.ids:
            self.ids[name] = len(self.signals)
            self.signals.append(0)
        return self.ids[name]

    def add(self, block: int, op: int, layer: int, args=(), waits=(), signal: str | None = None) -> None:
        """Appends an instruction to `block`'s queue that waits on the counters named in `waits` (at most 2)."""
        assert len(waits) <= 2 and len(args) <= ARGS, (waits, args)
        sig = -1 if signal is None else self.counter(signal)
        if sig >= 0:
            self.signals[sig] += 1
        self.queues[block].append((op, layer, tuple(self.counter(w) for w in waits), sig, tuple(args)))

    def build(self) -> Schedule:
        flat, offsets = [], [0]
        for q in self.queues:
            for op, layer, waits, sig, args in q:
                w = [x for c in waits for x in (c, self.signals[c])] + [-1, 0] * (2 - len(waits))
                flat += [op, layer, *w, sig, *args] + [0] * (ARGS - len(args))
            offsets.append(offsets[-1] + len(q))
        return Schedule(torch.tensor(flat, dtype=torch.int32).view(-1, FIELDS),
                        torch.tensor(offsets, dtype=torch.int32), len(self.signals))


@functools.lru_cache(maxsize=None)
def _ext(ops: Path, kv8: bool, tma: bool):
    """interp.cu built around the program's header `ops` (named by its path and contents: each program its own build)."""
    from inference_server.models import megakernel as mk
    tag = hashlib.sha1(str(ops.resolve()).encode() + ops.read_bytes()).hexdigest()[:12]
    return mk._build("interp.cu", f"qwen3_interp_{tag}", kv8, tma, 1, [str(ops.parent)])


class Framework:
    """A program's schedules for batch 1..BMAX on this GPU's SMs, its counters and scratch, over a `megakernel.Model`
    (its weights, KV pools and build flags). Build it before the KV pools are sized (~70 MB of scratch)."""

    def __init__(self, mdl, program=None):
        from inference_server.models import megakernel as mk
        from inference_server.models.megakernel import program as default
        self.mdl, self.program = mdl, program or default
        blocks = torch.cuda.get_device_properties(mdl.device).multi_processor_count
        vocab = mdl.lm_head.shape[0]
        self.schedules = {B: self.program.schedule(B, blocks, mdl.n_layers, vocab, mdl.fp8).to(mdl.device)
                          for B in range(1, mk.BMAX + 1)}
        self.counters = torch.zeros(max(s.n_counters for s in self.schedules.values()), dtype=torch.int32, device=mdl.device)
        self.f32 = torch.empty_like(mdl.f32)                    # its own scratch, laid out as the megakernel's
        self.bf16 = torch.empty_like(mdl.bf16)
        self.i32 = torch.zeros_like(mdl.i32)                    # the argmax keys start at 0; ARGMAX zeroes them after use


def decode_step(fw: Framework, tokens: torch.Tensor, pos: torch.Tensor, bt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`megakernel.decode_step` through the interpreter: (logits [B, V] bf16, greedy next token [B] int64), every
    layer's K/V for the new token written into the pools."""
    from inference_server.models import megakernel as mk
    mdl = fw.mdl
    assert mdl.table is not None, "bind_pools() first"
    s = fw.schedules[tokens.shape[0]]
    ext = _ext(Path(fw.program.OPS), bool(mdl.kv8), mk.TMA)
    return ext.decode_step(tokens.contiguous(), pos.contiguous(), bt.contiguous(), mdl.table, mdl.embed, mdl.final_norm,
                           mdl.lm_head, mdl.inv_freq, fw.f32, fw.bf16, fw.i32, float(mdl.scale), mdl.fp8,
                           s.instrs, s.offsets, fw.counters)
