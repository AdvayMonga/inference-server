"""Cloud adapters for lab.vm: each exposes get/create/start/stop/types over one provider's API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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


def env_field(name: str, default: str):
    """A dataclass default read from env at construction, not import."""
    return field(default_factory=lambda: os.environ.get(name, default))


def as_list(payload, what: str) -> list:
    """An API answer that must be a list, or a ProviderError naming the endpoint."""
    if not isinstance(payload, list):
        raise ProviderError(f"{what}: expected a list, got {type(payload).__name__}: {str(payload)[:200]}")
    return payload


def load(name: str):
    if name == "verda":
        from lab.providers import verda
        return verda.Verda()
    if name == "crusoe":
        from lab.providers import crusoe
        return crusoe.Crusoe()
    if name == "nebius":
        from lab.providers import nebius
        return nebius.Nebius()
    raise ProviderError(f"unknown provider {name!r}; LAB_VM_PROVIDER is verda, nebius or crusoe")
