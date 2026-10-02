"""The OS jail untrusted code runs in: sandbox-runtime (`srt`) settings and command wrapper."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


class JailMissing(RuntimeError):
    """No sandbox-runtime on this machine, and LAB_NO_JAIL is not set."""


def available() -> bool:
    return shutil.which("srt") is not None


def required() -> bool:
    """The referee runs agent code jailed unless LAB_NO_JAIL=1 (tests, a dev box you trust)."""
    return os.environ.get("LAB_NO_JAIL") != "1"


def settings(writable: list[Path], venv: Path, domains: list[str],
             readonly: tuple[Path, ...] | list[Path] = ()) -> dict:
    """Write only `writable`; read nothing under home but it, `readonly` and `venv`; reach `domains`."""
    return {
        "network": {"allowedDomains": list(domains), "deniedDomains": [],
                    "allowUnixSockets": [], "allowAllUnixSockets": False,
                    "allowLocalBinding": False},
        "filesystem": {
            "denyRead": [str(Path.home())],
            "allowRead": [str(p) for p in [*writable, *readonly, venv]],
            "allowWrite": [str(p) for p in writable],
            "denyWrite": [],
        },
        "enableWeakerNestedSandbox": False,
        "enableWeakerNetworkIsolation": False,
        "allowAppleEvents": False,
    }


def wrap(config: dict, settings_path: Path, argv: list[str]) -> list[str]:
    """`argv` under srt with `config` (written to `settings_path`, outside the jail); bare argv when the jail is waived."""
    if not available():
        if required():
            raise JailMissing("sandbox-runtime not installed: npm install -g @anthropic-ai/sandbox-runtime "
                              "(or LAB_NO_JAIL=1 on a box you trust)")
        return argv
    settings_path.write_text(json.dumps(config, indent=2))
    return [shutil.which("srt"), "--settings", str(settings_path), *argv]
