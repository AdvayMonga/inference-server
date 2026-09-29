"""SDK PreToolUse hook: refuse a file-tool write before it happens.

Covers the file tools only. Bash can still write; the jail bounds where, and the grader's audit
checks what changed, so a Bash write outside the surface is caught at the end instead of now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from .change_kinds import KINDS, may_write

WRITE_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"


def allowed(workspace: Path, file_path: str) -> bool:
    """A write the loop could make under some kind; the audit later checks the declared one."""
    ws = workspace.resolve()
    target = (ws / file_path).resolve()
    if not target.is_relative_to(ws):
        return False
    rel = target.relative_to(ws).as_posix()
    return any(may_write(k, rel, new_file=not target.exists()) for k in KINDS)


def write_guard(workspace: Path) -> Callable[..., Awaitable[dict[str, Any]]]:
    """A PreToolUse callback that denies writes outside every loop kind's surface."""
    async def hook(input_data: dict[str, Any], _tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
        tool_input = input_data.get("tool_input", {})
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if allowed(workspace, path):
            return {}
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"{path} is outside what this change may write",
        }}
    return hook
