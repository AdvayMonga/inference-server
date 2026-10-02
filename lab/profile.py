"""profile: run the engine in process under torch.profiler on a workload and write a bundle.

    python -m lab.profile --backend custom-mps --requests 8 --max-tokens 32 --out lab/runs
    python -m lab.profile --backend custom-mps --prompts prompts.json   # a JSON list of strings

White box by design: the profiler must live in the process it traces. A profile run is never a
benchmark run, since profiling perturbs timing; `bench` is the separate, black-box tool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity

from inference_server.backends.base import InferenceBackend
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.timeline import Timeline
from lab import bundle, gpu


def synthetic_prompts(n: int, length: int, vocab: int = 1000, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, vocab) for _ in range(length)] for _ in range(n)]


def _profiler(device: str) -> torch.profiler.profile:
    """All threads: by default the profiler records only the thread that opened it, and the scheduler has its own."""
    acts = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    return torch.profiler.profile(activities=acts,
                                  experimental_config=_ExperimentalConfig(profile_all_threads=True))


async def _fire(sched: ContinuousBatchScheduler, prompts: list[list[int]], max_tokens: int,
                session: str) -> list:
    loop = asyncio.get_running_loop()
    reqs = [ScheduledRequest(token_ids=p, max_tokens=max_tokens, session_id=f"{session}-{i}",
                             future=loop.create_future()) for i, p in enumerate(prompts)]
    return await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)


async def run(backend: InferenceBackend, prompts: list[list[int]], max_tokens: int, out: Path,
              warmup: int = 1, max_batch_size: int = 16, **scheduler_kw) -> Path:
    """Profile `prompts` through a fresh scheduler on `backend`; return the bundle directory."""
    out = bundle.new_dir(out)
    device = backend.device_str
    timeline = Timeline(out)
    sched = ContinuousBatchScheduler(backend, max_batch_size=max_batch_size, timeline=timeline,
                                     **scheduler_kw)
    sched.start()
    sampler = gpu.Sampler(out / "gpu.csv")
    try:
        if warmup:
            await _fire(sched, prompts[:warmup], max_tokens, "warmup")
        timeline.event("profile_window", state="begin")
        sampler.start()
        t0 = time.time()
        with _profiler(device) as prof:
            results = await _fire(sched, prompts, max_tokens, "profile")
        t1 = time.time()
        sampler.stop()
        timeline.event("profile_window", state="end")
    finally:
        await sched.stop()
    prof.export_chrome_trace(str(out / "trace.json"))
    bundle.write_json(out / "memory.json", bundle.device_memory(device))
    bundle.write_json(out / "stats.json", sched.stats())
    failed = sum(isinstance(r, BaseException) for r in results)
    bundle.write_json(out / "meta.json", bundle.meta(device, prompts, max_tokens, t0, t1,
                                                     failed=failed, warmup=warmup))
    return out


def _build_backend(name: str, model: str, num_blocks: int, block_size: int) -> InferenceBackend:
    from inference_server.backends import create_backend
    from inference_server.kv_cache.cache_manager import CacheManager
    backend = create_backend(name)
    backend.load_model(model)
    layer_shapes = backend.kv_shape_per_layer() if hasattr(backend, "kv_shape_per_layer") else None
    backend.set_cache_adapter(CacheManager(
        num_blocks=num_blocks, block_size=block_size, layer_shapes=layer_shapes,
        device=str(getattr(backend, "device", "cpu")), dtype=getattr(backend, "kv_dtype", None)))
    return backend


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="custom-mps")
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--prompts", help="JSON list of prompt strings; synthetic token ids when absent")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--kv-blocks", type=int, default=256)
    ap.add_argument("--kv-block-size", type=int, default=16)
    ap.add_argument("--out", default="lab/runs")
    args = ap.parse_args(argv)

    backend = _build_backend(args.backend, args.model, args.kv_blocks, args.kv_block_size)
    if args.prompts:
        from inference_server.tokenizer import Tokenizer
        tok = Tokenizer(args.model, 8192)
        prompts = [tok.encode_chat(t) for t in json.loads(Path(args.prompts).read_text())]
    else:
        prompts = synthetic_prompts(args.requests, args.prompt_len)
    out = asyncio.run(run(backend, prompts, args.max_tokens, Path(args.out), warmup=args.warmup,
                          max_batch_size=args.max_batch_size))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
