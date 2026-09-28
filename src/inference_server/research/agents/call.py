"""The one place a model is called: the Agent SDK's CLI, run inside the srt jail with a scrubbed env."""

from __future__ import annotations

import asyncio
import shlex
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..safety.hooks import WRITE_TOOLS, write_guard
from ..safety.jail import API_HOST, jail_command, srt_settings

MODEL = "claude-fable-5-1"   # strongest available; agreed 2026-09-28
TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]
PASS_ENV = ("PATH", "ANTHROPIC_API_KEY", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_AGENT_SDK_VERSION")


@dataclass
class AgentSpec:
    node: str
    system: str
    prompt: str
    workspace: Path
    scratch: Path                  # outside the workspace: the CLI's own home, the jail settings
    output_schema: dict[str, Any]
    max_turns: int
    max_budget_usd: float
    timeout_s: float
    readonly: list[Path] = field(default_factory=list)


@dataclass
class AgentReply:
    output: dict[str, Any] | None
    cost_usd: float | None
    turns: int
    error: str | None


def _wrapper(spec: AgentSpec, cli: Path) -> Path:
    """A script the SDK runs as its CLI: wipes the env, then execs the real CLI under srt."""
    home = spec.scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.gettempdir()).resolve()
    settings = srt_settings([spec.workspace.resolve(), home, tmp], Path(sys.prefix),
                            [API_HOST], readonly=spec.readonly)
    argv = jail_command(settings, spec.scratch / "srt.json", [str(cli)])
    keep = " ".join(f'"{k}=${k}"' for k in PASS_ENV)
    env = (f"{keep} HOME={shlex.quote(str(home))} CLAUDE_CONFIG_DIR={shlex.quote(str(home))} "
           f"TMPDIR={shlex.quote(str(tmp))} "
           f"PYTHONPATH={shlex.quote(str(spec.workspace / 'src'))} PYTHONDONTWRITEBYTECODE=1")
    script = spec.scratch / "cli.sh"
    script.write_text(f"#!/bin/sh\nexec env -i {env} {shlex.join(argv)} \"$@\"\n")
    script.chmod(0o700)
    return script


async def _run(spec: AgentSpec) -> AgentReply:
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

    import claude_agent_sdk
    cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    options = ClaudeAgentOptions(
        model=MODEL, system_prompt=spec.system, tools=TOOLS, allowed_tools=TOOLS,
        permission_mode="dontAsk", setting_sources=[], cwd=str(spec.workspace),
        max_turns=spec.max_turns, max_budget_usd=spec.max_budget_usd,
        output_format={"type": "json_schema", "schema": spec.output_schema},
        cli_path=str(_wrapper(spec, cli)),
        hooks={"PreToolUse": [HookMatcher(matcher=WRITE_TOOLS, hooks=[write_guard(spec.workspace)])]},
    )
    reply = AgentReply(None, None, 0, "no result message")
    async for msg in query(prompt=spec.prompt, options=options):
        if isinstance(msg, ResultMessage):
            reply = AgentReply(msg.structured_output if not msg.is_error else None,
                               msg.total_cost_usd, msg.num_turns,
                               None if not msg.is_error else msg.subtype)
    return reply


def call_agent(spec: AgentSpec) -> AgentReply:
    """Run one fresh agent session and return its structured output; never raises on agent failure."""
    spec.scratch.mkdir(parents=True, exist_ok=True)
    try:
        return asyncio.run(asyncio.wait_for(_run(spec), spec.timeout_s))
    except TimeoutError:
        return AgentReply(None, None, 0, f"timed out after {spec.timeout_s}s")
