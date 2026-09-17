"""What the GPU was actually doing while we measured it.

Two identical A100-80GB draws differed by **2.31x on byte-identical config** (LOOP.md, step 4).
A difference that large is not code, so it is the machine — and the two candidate mechanisms are
(a) boost and thermal drift, because nothing pinned the clocks, and (b) two physically different
hosts behind one SKU label. Neither was recorded, so neither could be ruled out.

This module records both, and locks the clocks where that is permitted:

    query_device(shell)  -> DeviceState      what the GPU is, and how fast it is allowed to run
    lock_clocks(shell)   -> (locked, why)    pin the SM clock, or say plainly that we may not
    unlock_clocks(shell)                     put it back

**Locking requires root.** `nvidia-smi -pm`, `-lgc`, `-ac` and `-pl` are all privileged, so no
container-based venue (RunPod Pods, Vast's Docker, Modal) can lock anything, whatever the run
asks for. That is why a refusal here is an outcome and not an error: the run continues, the panel
carries `clocks_locked=false`, and `compare.py` refuses to put it beside a locked one. The
read-only `--query-gpu` side works unprivileged everywhere, so the *recording* half always works.

stdlib only, and no torch: `research/` must not import the engine, and this runs on a box where
the only thing installed yet may be nvidia-smi.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Callable

Shell = Callable[[list[str]], subprocess.CompletedProcess]

# One query, one line, in this order. `nounits` so "300.00 W" arrives as "300.00".
_QUERY_FIELDS = (
    "name", "uuid", "driver_version", "clocks.sm", "clocks.mem", "clocks.max.sm",
    "power.limit", "persistence_mode", "ecc.mode.current", "clocks_throttle_reasons.active",
)

# nvidia-smi writes these when a field is unsupported on the part or unreadable by this user.
_MISSING = ("", "n/a", "[n/a]", "[not supported]", "not supported", "unknown",
            "[unknown error]", "[insufficient permissions]")


@dataclass
class DeviceState:
    """The machine a panel was measured on, as far as an unprivileged process can see it."""

    gpu_name: str | None = None
    gpu_uuid: str | None = None
    driver_version: str | None = None
    # nvidia-smi's CSV query has no CUDA-version field; an instrument that imports torch can fill
    # this from torch.version.cuda before stamping its panel. None here is normal, not a failure.
    cuda_version: str | None = None
    sm_clock_mhz: int | None = None
    mem_clock_mhz: int | None = None
    max_sm_clock_mhz: int | None = None
    power_limit_w: float | None = None
    persistence_mode: str | None = None
    ecc_mode: str | None = None
    # Two different facts, deliberately separate. `clocks_locked=False` alone conflates "we
    # never tried" (PodSpec.lock_clocks=False) with "we tried and were refused root", and the
    # only way to tell them apart used to be string-matching `lock_error`. A reader should
    # never have to parse prose to know whether a control was attempted.
    lock_attempted: bool = False
    clocks_locked: bool = False
    lock_error: str | None = None
    throttle_reasons: str | None = None
    host_id: str | None = None          # the provider's machine id, when it exposes one

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(shell: Shell, cmd: list[str]) -> tuple[int, str, str]:
    """(rc, stdout, stderr). A missing nvidia-smi is rc=127 with a message, never an exception."""
    try:
        r = shell(cmd)
    except Exception as e:                      # noqa: BLE001 — no GPU is a state, not a crash
        return 127, "", f"{type(e).__name__}: {e}"
    return r.returncode, r.stdout or "", r.stderr or ""


def _clean(value: str) -> str | None:
    v = value.strip()
    return None if v.lower() in _MISSING else v


def _int(value: str | None) -> int | None:
    try:
        return int(float(value))    # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float(value: str | None) -> float | None:
    try:
        return float(value)         # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def query_device(shell: Shell) -> DeviceState:
    """Read the device. Never raises, and never partially fails: an unparseable field is None.

    Defensive on purpose — this runs on whatever box the venue handed us, and a `[N/A]` in one
    column must not cost us the other nine. No GPU at all yields an all-None state, which is the
    correct record for a laptop run.
    """
    rc, out, _ = _run(shell, ["nvidia-smi", f"--query-gpu={','.join(_QUERY_FIELDS)}",
                              "--format=csv,noheader,nounits"])
    if rc != 0 or not out.strip():
        return DeviceState()
    # First GPU only: every venue we use rents one, and a multi-GPU box would need a per-device
    # panel to say anything honest anyway.
    cells = [_clean(c) for c in out.strip().splitlines()[0].split(",")]
    cells += [None] * (len(_QUERY_FIELDS) - len(cells))     # a short row loses fields, not all
    return DeviceState(
        gpu_name=cells[0], gpu_uuid=cells[1], driver_version=cells[2],
        sm_clock_mhz=_int(cells[3]), mem_clock_mhz=_int(cells[4]),
        max_sm_clock_mhz=_int(cells[5]), power_limit_w=_float(cells[6]),
        persistence_mode=cells[7], ecc_mode=cells[8], throttle_reasons=cells[9],
    )


NO_ROOT = "requires root; clocks left at default"

# How far the read-back SM clock may sit from the one we asked for and still count as pinned.
# `-lgc X` sets a floor and a ceiling, so a healthy card reads back exactly X; one boost bin is
# ~15 MHz, so this allows one bin of rounding and nothing like a card that ignored us.
LOCK_TOLERANCE_MHZ = 30

# Permissions ONLY. "not supported" is deliberately absent: nvidia-smi says that when the part
# or the mode cannot do it at all (MIG, most GeForce), and recording "requires root" there would
# put a false reason in the panel — the one thing this module exists not to do.
_PERMISSION_MARKERS = ("insufficient permission", "permission denied",
                       "requires root", "must be run as", "operation not permitted")


def _refusal(rc: int, out: str, err: str) -> str:
    """Why a privileged nvidia-smi call failed, in the caller's words when we recognise it."""
    msg = (err or out).strip().replace("\n", " ")[:200]
    if any(m in msg.lower() for m in _PERMISSION_MARKERS):
        return NO_ROOT
    return f"nvidia-smi exited {rc}: {msg}" if msg else f"nvidia-smi exited {rc}"


def lock_clocks(shell: Shell, sm_clock: int | None = None) -> tuple[bool, str]:
    """Pin persistence mode and the SM clock. Returns (locked, reason). Never raises.

    `(False, reason)` is an ordinary outcome: every container venue lands there, because the
    privileged nvidia-smi verbs need real root. The caller runs anyway and records the refusal —
    an unlocked run is still usable evidence, it just cannot be compared against a locked one.
    """
    rc, out, err = _run(shell, ["nvidia-smi", "-pm", "1"])
    if rc != 0:
        return False, _refusal(rc, out, err)

    if sm_clock is None:
        sm_clock = query_device(shell).max_sm_clock_mhz
        if sm_clock is None:
            return False, "could not read the maximum SM clock; clocks left at default"

    rc, out, err = _run(shell, ["nvidia-smi", "-lgc", str(sm_clock)])
    if rc != 0:
        return False, _refusal(rc, out, err)

    # `-lgc` exiting 0 is NOT proof the clock moved: the driver can accept the request and
    # leave the card where it was (thermal state, an already-active lock, driver quirk).
    # Believing the exit code would write a wrong-but-plausible `clocks_locked=true` into the
    # validity block, and this whole methodology rests on that block being true. So read it
    # back. A refusal we know about is worth far more than a lock we only think we took.
    readback = query_device(shell).sm_clock_mhz
    if readback is None:
        return False, "lock reported success but the SM clock could not be read back"
    if abs(readback - sm_clock) > LOCK_TOLERANCE_MHZ:
        return False, (f"lock reported success but clocks read back at {readback} MHz, "
                       f"not the {sm_clock} MHz asked for")
    return True, f"SM clock locked to {sm_clock} MHz (read back {readback} MHz)"


def unlock_clocks(shell: Shell) -> None:
    """Best effort `-rgc`. Correct if a machine is ever reused, harmless if it is torn down."""
    _run(shell, ["nvidia-smi", "-rgc"])
