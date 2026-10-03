"""Deliberate regressions the eval harness must flag, applied from outside the engine (usage: lab/README.md)."""

from __future__ import annotations

import argparse
import functools
import time

NAMES = ("slow_decode", "slow_start", "kv_leak", "admit_fewer")
SLOW_DECODE_S = 0.003      # per decode step: worse TPOT in every decode-heavy regime
SLOW_START_S = 20.0        # after the model loads: worse cold start


def parse(spec: str) -> frozenset[str]:
    names = {n.strip() for n in spec.split(",") if n.strip()}
    unknown = names - set(NAMES)
    if unknown:
        raise ValueError(f"unknown canary {sorted(unknown)}; one of {NAMES}")
    return frozenset(names)


def apply(names: frozenset[str]) -> None:
    """Monkeypatch the engine in this process. The engine's own tree never changes, so the player cannot edit a canary away."""
    from inference_server.backends.custom_torch_backend import CustomTorchBackend as B
    from inference_server.scheduler import ContinuousBatchScheduler as S

    if "slow_start" in names:
        load = B.load_model

        @functools.wraps(load)
        def slow_load(self, *a, **k):
            load(self, *a, **k)
            time.sleep(SLOW_START_S)
        B.load_model = slow_load
    if "slow_decode" in names:
        step = B.decode_step_batched

        @functools.wraps(step)
        def slow_step(self, *a, **k):
            time.sleep(SLOW_DECODE_S)
            return step(self, *a, **k)
        B.decode_step_batched = slow_step
    if "kv_leak" in names:
        B.kv_release = lambda self, prompt_len, max_tokens: None        # reservations are never freed
    if "admit_fewer" in names:
        admit = S._admit_pending

        @functools.wraps(admit)
        def admit_fewer(self, device):
            cap = self.max_batch_size            # one fewer row admitted per pass; stats still report the config
            self.max_batch_size = max(0, cap - 1)
            try:
                admit(self, device)
            finally:
                self.max_batch_size = cap
        S._admit_pending = admit_fewer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.canary",
                                 description="serve the engine with deliberate regressions applied")
    ap.add_argument("names", help="comma-separated: " + ", ".join(NAMES))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    apply(parse(a.names))
    import uvicorn
    uvicorn.run("inference_server.server:app", host=a.host, port=a.port)   # same process: patches hold
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
