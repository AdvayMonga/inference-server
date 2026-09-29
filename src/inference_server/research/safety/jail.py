"""The OS jail untrusted code runs in: sandbox-runtime (`srt`) settings and command wrapper."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

API_HOST = "api.anthropic.com"


def srt_settings(writable: list[Path], venv: Path, domains: list[str],
                 readonly: tuple[Path, ...] | list[Path] = ()) -> dict:
    """Write only `writable`; read nothing under home but it, `readonly` and `venv`; reach `domains`.

    Keep each run's temp dir inside `writable` and under home, where other runs cannot read it.
    """
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


def jail_command(settings: dict, settings_path: Path, argv: list[str]) -> list[str]:
    """Write `settings` to `settings_path` (keep it outside the jail) and wrap `argv` in srt."""
    srt = shutil.which("srt")
    if srt is None:
        raise RuntimeError("sandbox-runtime not installed: npm install -g @anthropic-ai/sandbox-runtime")
    settings_path.write_text(json.dumps(settings, indent=2))
    return [srt, "--settings", str(settings_path), *argv]
