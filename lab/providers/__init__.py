"""Cloud adapters for lab.vm: each exposes get/create/start/stop/types over one provider's API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    """What lab.vm needs to know about a VM, whichever cloud it is on."""
    name: str
    state: str          # running | stopped | absent | anything else = transitional
    ip: str | None
    type: str
    detail: str = ""    # provider wording, for status output


class ProviderError(RuntimeError):
    """The provider refused; its message is here."""


def load(name: str):
    if name == "verda":
        from lab.providers import verda
        return verda.Verda()
    if name == "crusoe":
        from lab.providers import crusoe
        return crusoe.Crusoe()
    raise ProviderError(f"unknown provider {name!r}; LAB_VM_PROVIDER is verda or crusoe")
