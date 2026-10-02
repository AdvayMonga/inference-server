"""profile: run the served engine in process under torch.profiler on a workload and write a raw bundle (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import random
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity

from inference_server.backends.base import InferenceBackend
from inference_server.config import Settings, load_settings
from inference_server.scheduler import ContinuousBatchScheduler, ScheduledRequest
from inference_server.server import build_backend, build_scheduler
from inference_server.timeline import Timeline
from lab import bundle, gpu

logger = logging.getLogger(__name__)


def synthetic_prompts(n: int, length: int, vocab: int = 1000, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, vocab) for _ in range(length)] for _ in range(n)]


def _profiler(device: str) -> torch.profiler.profile:
    """All threads: by default the profiler records only the thread that opened it, and the scheduler has its own."""
    acts = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    try:
        from torch._C._profiler import _ExperimentalConfig
        config = _ExperimentalConfig(profile_all_threads=True)
    except (ImportError, TypeError):   # older torch: the trace will miss the scheduler thread
        logger.warning("torch %s cannot profile all threads; trace covers the main thread only",
                       torch.__version__)
        return torch.profiler.profile(activities=acts)
    return torch.profiler.profile(activities=acts, experimental_config=config)


async def _fire(sched: ContinuousBatchScheduler, prompts: list[list[int]], max_tokens: int,
                session: str) -> list:
    loop = asyncio.get_running_loop()
    reqs = [ScheduledRequest(token_ids=p, max_tokens=max_tokens, session_id=f"{session}-{i}",
                             future=loop.create_future()) for i, p in enumerate(prompts)]
    return await asyncio.gather(*(sched.submit(r) for r in reqs), return_exceptions=True)


async def run(backend: InferenceBackend, settings: Settings, prompts: list[list[int]],
              max_tokens: int, out: Path, warmup: int = 1) -> Path:
    """Profile `prompts` through the served scheduler config on `backend`; return the bundle directory."""
    out = bundle.new_dir(out)
    device = backend.device_str
    timeline = Timeline(out)
    sched = build_scheduler(backend, settings, timeline=timeline)
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
        timeline.event("profile_window", state="end")
    finally:
        sampler.stop()
        await sched.stop()
    prof.export_chrome_trace(str(out / "trace.json"))
    bundle.write_json(out / "memory.json", bundle.device_memory(device))
    bundle.write_json(out / "stats.json", sched.stats())
    failed = sum(isinstance(r, BaseException) for r in results)
    bundle.write_json(out / "meta.json", bundle.meta(device, settings, prompts, max_tokens, t0, t1,
                                                     failed=failed, warmup=warmup))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("(")[0])
    ap.add_argument("--backend", help="override BACKEND, e.g. custom-mps")
    ap.add_argument("--model", help="override MODEL_NAME")
    ap.add_argument("--prompts", help="JSON list of prompt strings; synthetic token ids when absent")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="lab/runs")
    args = ap.parse_args(argv)

    # The served configuration, from the same env the server reads; only the workload is ours.
    settings = load_settings()
    overrides = {k: v for k, v in (("backend_name", args.backend), ("model_name", args.model)) if v}
    settings = dataclasses.replace(settings, **overrides)
    backend, _ = build_backend(settings)
    if args.prompts:
        from inference_server.tokenizer import Tokenizer
        tok = Tokenizer(settings.model_name, settings.context_window)
        prompts = [tok.encode_chat(t) for t in json.loads(Path(args.prompts).read_text())]
    else:
        prompts = synthetic_prompts(args.requests, args.prompt_len)
    out = asyncio.run(run(backend, settings, prompts, args.max_tokens, Path(args.out),
                          warmup=args.warmup))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
