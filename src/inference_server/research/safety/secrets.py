"""The loop's credentials: where they come from, and keeping them out of anything recorded."""

from __future__ import annotations

import os
import subprocess
from typing import Any

KEYCHAIN_SERVICE = "claude-loop-token"   # `claude setup-token`, stored with `security add-generic-password`
REDACTED = "[REDACTED]"


def model_token() -> str | None:
    """The subscription token for agent sessions: macOS Keychain first, then the environment."""
    try:
        out = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except FileNotFoundError:   # not macOS
        pass
    return os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")


def known_secrets() -> list[str]:
    """Every credential value the loop holds, for redaction."""
    found = [model_token()] + [os.environ.get(k) for k in
                               ("ANTHROPIC_API_KEY", "RUNPOD_API_KEY", "GITHUB_TOKEN", "GH_TOKEN")]
    return [s for s in found if s and len(s) >= 8]


def redact(value: Any, secrets: list[str]) -> Any:
    """`value` with every secret replaced, recursing through dicts and lists."""
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, REDACTED)
        return value
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value
