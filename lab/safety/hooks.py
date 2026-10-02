"""Agent SDK PreToolUse hook: refuse a file-tool write outside the surface before it happens (Bash writes are caught by the audit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from lab.safety.surfaces import may_write

WRITE_TOOLS = "Write|Edit|MultiEdit|NotebookEdit"


def allowed(workspace: Path, base_files: frozenset[str], file_path: str) -> bool:
    """`new_file` means not in the base tree: a test the agent added stays editable by the agent."""
    ws = workspace.resolve()
    target = (ws / file_path).resolve()
    if not target.is_relative_to(ws):
        return False
    rel = target.relative_to(ws).as_posix()
    return may_write(rel, new_file=rel not in base_files)


def write_guard(workspace: Path, base_files: frozenset[str]) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def hook(input_data: dict[str, Any], _tool_use_id: str | None, _ctx: Any) -> dict[str, Any]:
        tool_input = input_data.get("tool_input", {})
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if allowed(workspace, base_files, path):
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                       "permissionDecisionReason": f"{path} is outside what the agent may write"}}
    return hook
