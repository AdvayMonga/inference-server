"""Deliberate regressions that prove the eval harness covers the stack. Off unless
INFERENCE_SERVER_CANARY names one or more (comma-separated). Each is a realistic bug the harness must
flag; a canary nothing catches is a blind spot. Read once at import (`reload()` for tests).

  slow_decode   +3 ms per decode step            -> worse TPOT in every decode-heavy regime
  slow_start    +20 s after the model loads       -> worse cold start
  kv_leak       KV reservations are never freed   -> admissions stall, failures under long/overload load
  admit_fewer   one fewer row admitted per batch  -> lower goodput when saturated
"""

import os

NAMES = ("slow_decode", "slow_start", "kv_leak", "admit_fewer")
_active: frozenset[str] = frozenset()


def reload() -> frozenset[str]:
    global _active
    names = {n.strip() for n in os.environ.get("INFERENCE_SERVER_CANARY", "").split(",") if n.strip()}
    unknown = names - set(NAMES)
    if unknown:
        raise ValueError(f"unknown canary {sorted(unknown)}; one of {NAMES}")
    _active = frozenset(names)
    return _active


def active(name: str) -> bool:
    return name in _active


reload()
