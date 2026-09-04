"""The research loop: contract, gates and stages that drive engine improvement.

Nothing in here may import engine internals. Instruments in scripts/ are the only layer that
knows about backends, pools and kernels; they hand this package structured records.
See LOOP.md for the method this implements.
"""

from inference_server.research.schemas import (
    PANEL_VERSION,
    Experiment,
    GateResult,
    Hypothesis,
    KnowledgeEntry,
    Validity,
    Vitals,
)

__all__ = [
    "PANEL_VERSION",
    "Experiment",
    "GateResult",
    "Hypothesis",
    "KnowledgeEntry",
    "Validity",
    "Vitals",
]
