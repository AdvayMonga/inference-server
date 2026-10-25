"""CAPTURE_REGION=first:last records, during decode steps, what that stretch of the decode stations (models/stations.py)
reads and writes, in the layout of BlameGraph's lab/capture.py: one JSON line per layer call in
<CAPTURE_REGION_DIR>/calls.jsonl, and the first call per (layer, shape signature) whole in samples/<n>.pt as
{"reads": {name: tensor}, "writes": {name: tensor}} on the CPU. Weights are named and described, never saved. A
sample's K/V pool is only the blocks the step touches, renumbered in the order of its `kv_blocks` (their ids in the
real pool), its block tables renumbered to match. CAPTURE_REGION_LAYERS=0,23,47 limits the layers. Reads tensors
back, so it is for a capture run; with CUDA graphs on (CUSTOM_BACKEND_CUDA_GRAPH) only the eager calls are seen."""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from pathlib import Path

import torch

SMALL_INT = 4096                 # 1-D integer tensors up to this many elements are logged by value
SAMPLE_BYTES = 256 * 2**20       # a call whose reads and writes are larger is logged, not sampled
KV = ("k_cache", "v_cache")
TABLES = ("block_tables", "attn_block_tables")


def _describe(x) -> dict:
    if isinstance(x, torch.Tensor):
        d = {"shape": list(x.shape), "dtype": str(x.dtype).removeprefix("torch."), "contiguous": x.is_contiguous()}
        if not x.is_floating_point() and x.dim() == 1 and x.numel() <= SMALL_INT:
            d["values"] = x.tolist()
        return d
    if dataclasses.is_dataclass(x):
        return {"value": dataclasses.asdict(x)}
    if isinstance(x, (int, float, bool, str)) or x is None:
        return {"value": x}
    return {"type": type(x).__name__}


@functools.lru_cache(maxsize=None)
def active() -> "Capture | None":
    spec = os.environ.get("CAPTURE_REGION")
    if not spec:
        return None
    first, _, last = spec.partition(":")
    out = os.environ.get("CAPTURE_REGION_DIR")
    if not out:
        raise ValueError("CAPTURE_REGION needs CAPTURE_REGION_DIR")
    layers = os.environ.get("CAPTURE_REGION_LAYERS")
    return Capture(first, last or first, {int(x) for x in layers.split(",")} if layers else None, Path(out))


class Capture:
    def __init__(self, first: str, last: str, layers: set[int] | None, out: Path):
        from inference_server.models import stations
        self.first, self.last, self.layers, self.out = first, last, layers, out
        self.inputs, _, self.outputs = stations.interface(stations.DECODE, first, last)
        (out / "samples").mkdir(parents=True, exist_ok=True)
        self._log = open(out / "calls.jsonl", "a", buffering=1)
        self._seen: set[str] = set()
        self._layer, self._call, self._sample, self._blocks = 0, None, None, None

    def wants(self, layer: int) -> bool:
        return self.layers is None or layer in self.layers

    def steps(self, steps: tuple, layer: int) -> tuple:
        """`steps` (a resolved plan) with a recording step before the stretch's first station and after its last."""
        starts, ends = [s[0] for s in steps], [s[1] for s in steps]
        if self.first not in starts or self.last not in ends or starts.index(self.first) > ends.index(self.last):
            raise ValueError(f"CAPTURE_REGION {self.first}:{self.last} does not start and end on the plan's steps "
                             f"{[s[:2] for s in steps]}")
        i, j = starts.index(self.first), ends.index(self.last) + 1
        self._layer = layer
        return (*steps[:i], ("", "", self._before), *steps[i:j], ("", "", self._after), *steps[j:])

    def _before(self, c: dict) -> dict:
        self._call = self._sample = None
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return {}
        reads = {n: c[n] for n in self.inputs}
        call = {"fn": f"region:{self.first}:{self.last}", "layer": self._layer,
                "reads": {n: _describe(v) for n, v in reads.items()}}
        sig = json.dumps([self._layer] + [[n, d.get("shape"), d.get("dtype"), d.get("value")]
                                          for n, d in call["reads"].items()])
        if sig not in self._seen:
            self._seen.add(sig)
            self._blocks = _touched(c) if any(k in reads for k in KV) else None
            self._sample = {"reads": self._cpu(reads)}
            if self._blocks is not None:
                self._sample["reads"]["kv_blocks"] = self._blocks.cpu()
        self._call = call
        return {}

    def _after(self, c: dict) -> dict:
        if self._call is None:
            return {}
        writes = {n: c[n] for n in self.outputs}
        self._call["writes"] = {n: _describe(v) for n, v in writes.items()}
        if self._sample is not None:
            self._sample["writes"] = self._cpu(writes)
            size = sum(t.numel() * t.element_size() for part in self._sample.values() for t in part.values()
                       if isinstance(t, torch.Tensor))
            if size <= SAMPLE_BYTES:
                name = f"{len(self._seen)}.pt"
                torch.save(self._sample, self.out / "samples" / name)
                self._call["sample"] = name
            else:
                self._call["sample_skipped_bytes"] = size
        self._log.write(json.dumps(self._call) + "\n")
        self._call = self._sample = None
        return {}

    def _cpu(self, entries: dict) -> dict:
        out = {}
        for n, v in entries.items():
            if n.startswith("w."):
                continue
            if n in KV and self._blocks is not None:
                v = v[self._blocks]
            elif n in TABLES and self._blocks is not None:
                v = _renumber(v, self._blocks)
            if dataclasses.is_dataclass(v):
                v = dataclasses.asdict(v)              # fields, so a sample loads with torch.load(weights_only=True)
            out[n] = v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
        return out


def _touched(c: dict) -> torch.Tensor:
    """The pool blocks this step's rows hold (the new token's included), sorted."""
    bt, sl, bs = c["attn_block_tables"], c["attn_seq_lens"], c["k_cache"].shape[2]
    used = torch.arange(bt.shape[1], device=bt.device)[None, :] < ((sl + bs - 1) // bs)[:, None]
    return bt[used].long().unique()


def _renumber(table: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Block ids as positions in `blocks`; an id not in it (a column past the row's end) becomes 0."""
    pos = torch.searchsorted(blocks, table.long().clamp_min(0)).clamp_max(blocks.numel() - 1)
    return torch.where(blocks[pos] == table, pos, 0).to(table.dtype)
