"""Agent SDK PreToolUse hook: refuse a file-tool write outside the surface before it happens (Bash writes are caught by the audit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from lab.safety.surfaces import may_write

WRITE_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"


def allowed(workspace: Path, file_path: str) -> bool:
    ws = workspace.resolve()
    target = (ws / file_path).resolve()
    if not target.is_relative_to(ws):
        return False
    return may_write(target.relative_to(ws).as_posix(), new_file=not target.exists())


def write_guard(workspace: Path) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def hook(input_data: dict[str, Any], _tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
        tool_input = input_data.get("tool_input", {})
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if allowed(workspace, path):
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": f"{path} is outside what the agent may write"}}
    return hook
