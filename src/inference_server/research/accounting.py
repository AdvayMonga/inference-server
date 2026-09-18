"""Total accounting: what the whole run cost, not only what happened after the first request.

The spec is blunt about why this exists (notes/03, "Total accounting"):

    The measured window must include everything, or cost hides outside it. [...]
    If it is not in the accounting, the loop will eventually move cost into it.

It is an anti-hacking measure, not a nicety. notes/07 lists two of the six ways this loop would
reward hack as "move cost off the measured window" (do the work at init, or after the timer
stops) and "burn an unmeasured resource" (host RAM, disk, startup time). Both are blocked by
accounting from process start rather than from first successful request — and only by that.

Four terms, and the honesty about which of them this box can measure:

| term | how | measurable here |
|---|---|---|
| wall clock from process start | the instrument stamps the launch, not the first request | yes |
| peak host RSS | `resource.getrusage().ru_maxrss` | yes |
| peak device memory | the INSTRUMENT measures it and passes it in (see the seam below) | MPS: sampled; CUDA: exact |
| storage read bytes during load | `/proc/self/io` | Linux only; see `storage_read_bytes` |

**The torch seam.** `research/` may not import torch — the loop CI lane installs pytest and
ruff and nothing else, and that seam is what keeps the judging layer independent of the engine.
So device memory is measured by the instrument (which owns torch, and on a server run owns the
subprocess that holds the device) and handed to `Accounting` as a number. This module never
learns what a GPU is.

**Idle time is a term, not a rounding error.** "GPU-seconds allocated, including idle time in
warm pools" is what stops warm pooling being free: a replica that holds a device for 90s to
serve 11s of traffic has spent 90 GPU-seconds. `wall_s_from_process_start` and `serving_wall_s`
are recorded separately so the ratio is derivable at read time rather than baked in by the
instrument — same reason `sweep_headline` derives late from rate-points.
"""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# Stamped when this module is first imported, which every instrument does before it does any
# work. It is an approximation of process start: the interpreter startup and the imports that
# ran before this one are NOT in it. An instrument that can stamp the real launch — one that
# starts the measured process itself, like replay_local.py — should pass that instead.
PROCESS_START_MONOTONIC = time.monotonic()



def process_uptime_s() -> float:
    """Seconds since this module was imported. See PROCESS_START_MONOTONIC for what that misses."""
    return time.monotonic() - PROCESS_START_MONOTONIC


def peak_rss_gb(*, children: bool = False) -> float:
    """Peak resident set size in GB, for this process or for its reaped children.

    `children=True` reads RUSAGE_CHILDREN, which is a high-water mark over every child this
    process has already waited on — not a per-child figure. For an instrument that launches one
    server per run it therefore reports the largest server so far, which is an upper bound for
    any later run. An upper bound is the conservative direction for a cost gate, so it is
    reported rather than approximated away.
    """
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    # ru_maxrss is BYTES on Darwin and KIBIBYTES on Linux. Getting this wrong is a 1024x error,
    # in the direction that makes a memory regression invisible. (coldstart_load.py has the same
    # two lines inline; it predates this module and research/ cannot import scripts/.)
    divisor = 1e9 if sys.platform == "darwin" else 1e6
    return round(resource.getrusage(who).ru_maxrss / divisor, 3)


def storage_read_bytes() -> tuple[int | None, str]:
    """Bytes read from storage by this process, and why the answer is None when it is.

    Linux exposes this as `read_bytes` in `/proc/self/io`. macOS does not: `ru_inblock` stays 0
    across a 2.3GB safetensors read (measured), and the number that would work —
    `proc_pid_rusage`'s `ri_diskio_bytesread` — is a C API with no stdlib binding. Saying so is
    the point: an approximated term is worse than a named gap, because the gate would then
    compare two guesses and call it evidence.
    """
    try:
        with open("/proc/self/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1]), ""
    except OSError:
        pass
    return None, (f"storage read bytes are not readable on {sys.platform} from the stdlib "
                  f"(ru_inblock stays 0 on Darwin; proc_pid_rusage has no stdlib binding)")


@dataclass
class Accounting:
    """The whole cost of one measured run. Rides on the panel as `Vitals.accounting`.

    Every field is optional and defaults to None, because an instrument that cannot measure a
    term must say None and name the gap in `unmeasured` rather than write a zero. A zero is
    indistinguishable from "free", which is exactly the hiding place this block exists to close.
    """

    # Wall clock from the measured process's START — model load, warm-up and idle included.
    # NOT the replay window: `Vitals.wall_s` is that, and substituting it here would exclude
    # model load, which is the largest single cost on a cold replica.
    wall_s_from_process_start: float | None = None
    serving_wall_s: float | None = None          # the window that actually served traffic
    sessions_served: int | None = None           # distinct sessions inside that window
    peak_host_rss_gb: float | None = None
    peak_device_mem_gb: float | None = None
    device_mem_source: str | None = None         # how it was measured; None means it was not
    storage_read_bytes: int | None = None
    # term -> why it is missing. Read by the cost gate and printed with the primary metric, so
    # an unmeasured term travels with every number derived from this run.
    unmeasured: dict[str, str] = field(default_factory=dict)

    @property
    def idle_s(self) -> float | None:
        """Allocated but not serving. On a one-box run this IS the warm-pool term."""
        if self.wall_s_from_process_start is None or self.serving_wall_s is None:
            return None
        return round(max(0.0, self.wall_s_from_process_start - self.serving_wall_s), 3)

    @property
    def serving_fraction(self) -> float | None:
        """Share of allocated time that served traffic. 0.12 means 88% of the bill was idle."""
        if self.idle_s is None or not self.wall_s_from_process_start:
            return None
        return round(self.serving_wall_s / self.wall_s_from_process_start, 4)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["idle_s"] = self.idle_s
        d["serving_fraction"] = self.serving_fraction
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Accounting":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def of(cls, panel: Any) -> "Accounting | None":
        """The accounting block on a panel, or None for a panel written before this existed."""
        raw = getattr(panel, "accounting", None)
        return cls.from_dict(raw) if isinstance(raw, dict) else None

    def apply(self, panel: Any) -> Any:
        """Attach to a panel, mirroring into the two resource fields that already exist.

        `peak_host_rss_gb` and `peak_gpu_mem_gb` have been in the panel since LOOP.md step 0 and
        the cost gate reads them; filling them here is what makes the gate see this run's
        resources without a second definition of the same number.
        """
        panel.accounting = self.to_dict()
        if self.peak_host_rss_gb is not None and panel.peak_host_rss_gb is None:
            panel.peak_host_rss_gb = self.peak_host_rss_gb
        if self.peak_device_mem_gb is not None and panel.peak_gpu_mem_gb is None:
            panel.peak_gpu_mem_gb = self.peak_device_mem_gb
        return panel

    def summary(self) -> str:
        """One block a human reads next to the instrument's own table."""
        def g(v, unit=""):
            return "unmeasured" if v is None else f"{v}{unit}"
        lines = [
            f"  wall from process start  {g(self.wall_s_from_process_start, 's')}",
            f"  of which serving         {g(self.serving_wall_s, 's')} "
            f"(idle {g(self.idle_s, 's')})",
            f"  sessions served          {g(self.sessions_served)}",
            f"  peak host RSS            {g(self.peak_host_rss_gb, ' GB')}",
            f"  peak device memory       {g(self.peak_device_mem_gb, ' GB')}"
            + (f"  [{self.device_mem_source}]" if self.device_mem_source else ""),
            f"  storage read bytes       {g(self.storage_read_bytes)}",
        ]
        for term, why in sorted(self.unmeasured.items()):
            lines.append(f"  NOT ACCOUNTED: {term} — {why}")
        return "\n".join(lines)
