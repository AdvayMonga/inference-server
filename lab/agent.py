"""The one place a model is called. A Provider runs one session with the lab's tools; Claude via the Agent SDK is the first."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from lab.safety import jail
from lab.safety.grader import HF_HUB
from lab.safety.hooks import WRITE_TOOLS, write_guard

API_HOST = "api.anthropic.com"
BUILTIN_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]
PASS_ENV = ("PATH", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_ENTRYPOINT",
            "CLAUDE_AGENT_SDK_VERSION")
# What a session says when it ends. `stop` means the agent is done with this run.
OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["status", "note"],
                 "properties": {"status": {"type": "string", "enum": ["continue", "stop"]},
                                "note": {"type": ["string", "null"]}}}


@dataclass(frozen=True)
class ToolSpec:
    """A lab tool: runs in the harness process, outside the jail, with the referee's rights."""
    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[[dict[str, Any]], str]    # returns the text the agent sees


@dataclass
class AgentSpec:
    system: str
    prompt: str
    workspace: Path
    scratch: Path                           # outside the workspace: the CLI's home, jail settings
    tools: list[ToolSpec]
    base_files: frozenset[str] = frozenset()   # for the write hook: new means not in the base tree
    builtin: list[str] = field(default_factory=lambda: list(BUILTIN_TOOLS))
    max_turns: int = 200
    max_budget_usd: float = 5.0
    timeout_s: float = 3600.0
    model: str = field(default_factory=lambda: os.environ.get("LAB_MODEL", "claude-fable-5-1"))


@dataclass
class AgentReply:
    output: dict[str, Any] | None
    cost_usd: float
    turns: int
    error: str | None
    cost_estimated: bool = False     # no accounting came back; charged at the session cap


class Provider(Protocol):
    name: str

    def run(self, spec: AgentSpec) -> AgentReply: ...


def load(name: str | None = None) -> Provider:
    name = name or os.environ.get("LAB_AGENT_PROVIDER", "claude")
    if name == "claude":
        return ClaudeAgentSDK()
    raise ValueError(f"unknown agent provider {name!r}; LAB_AGENT_PROVIDER is claude")


class ClaudeAgentSDK:
    """Claude through the Agent SDK CLI, which runs inside the srt jail with a wiped env; lab tools run out here."""

    name = "claude"

    def _wrapper(self, spec: AgentSpec, cli: Path) -> Path:
        home, tmp = spec.scratch / "home", spec.scratch / "tmp"
        home.mkdir(parents=True, exist_ok=True)
        tmp.mkdir(exist_ok=True)
        ws = spec.workspace.resolve()
        weights = [HF_HUB] if HF_HUB.exists() else []      # model weights read-only; the token beside them stays hidden
        cli_tmp = Path(f"/private/tmp/claude-{os.getuid()}")   # the CLI's own per-project scratch; Bash fails without it
        cli_tmp.mkdir(parents=True, exist_ok=True)
        config = jail.settings([home.resolve(), tmp.resolve(), ws, cli_tmp.resolve()], Path(sys.prefix),
                               [API_HOST], readonly=weights)
        argv = jail.wrap(config, spec.scratch / "srt.json", [str(cli)])
        keep = " ".join(f'"{k}=${k}"' for k in PASS_ENV)
        env = (f"{keep} HOME={shlex.quote(str(home))} CLAUDE_CONFIG_DIR={shlex.quote(str(home))} "
               f"TMPDIR={shlex.quote(str(tmp))} PYTHONPATH={shlex.quote(str(ws / 'src'))} "
               f"PYTHONDONTWRITEBYTECODE=1 HF_HUB_CACHE={shlex.quote(str(HF_HUB))} HF_HUB_OFFLINE=1 "
               f"HF_HOME={shlex.quote(str(tmp / 'hf-home'))}")
        script = spec.scratch / "cli.sh"
        script.write_text(f"#!/bin/sh\nexec env -i {env} {shlex.join(argv)} \"$@\"\n")
        script.chmod(0o700)
        return script

    async def _run(self, spec: AgentSpec) -> AgentReply:
        import claude_agent_sdk
        from claude_agent_sdk import (ClaudeAgentOptions, HookMatcher, ResultMessage,
                                      create_sdk_mcp_server, query, tool)

        def adapt(t: ToolSpec):
            async def call(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    text = await asyncio.to_thread(t.fn, args)   # a long test run must not block the transport
                except Exception as e:          # the agent sees the refusal, the harness keeps going
                    text = f"{t.name} refused: {e}"
                return {"content": [{"type": "text", "text": text}]}
            return tool(t.name, t.description, t.schema)(call)

        server = create_sdk_mcp_server("lab", tools=[adapt(t) for t in spec.tools])
        cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        options = ClaudeAgentOptions(
            model=spec.model, system_prompt=spec.system,
            tools=spec.builtin, allowed_tools=spec.builtin + [f"mcp__lab__{t.name}" for t in spec.tools],
            mcp_servers={"lab": server}, permission_mode="dontAsk", setting_sources=[],
            cwd=str(spec.workspace), max_turns=spec.max_turns, max_budget_usd=spec.max_budget_usd,
            output_format={"type": "json_schema", "schema": OUTPUT_SCHEMA},
            cli_path=str(self._wrapper(spec, cli)),
            hooks={"PreToolUse": [HookMatcher(matcher=WRITE_TOOLS,
                                              hooks=[write_guard(spec.workspace, spec.base_files)])]},
        )
        reply = AgentReply(None, 0.0, 0, "no result message")
        async for msg in query(prompt=spec.prompt, options=options):
            if isinstance(msg, ResultMessage):
                reply = AgentReply(msg.structured_output if not msg.is_error else None,
                                   float(msg.total_cost_usd or 0.0), msg.num_turns,
                                   None if not msg.is_error else msg.subtype)
        return reply

    def run(self, spec: AgentSpec) -> AgentReply:
        """Never raises. Without a result message the spend is unknown and is charged at the session cap."""
        spec.scratch.mkdir(parents=True, exist_ok=True)
        try:
            return asyncio.run(asyncio.wait_for(self._run(spec), spec.timeout_s))
        except TimeoutError:
            return AgentReply(None, spec.max_budget_usd, 0, f"timed out after {spec.timeout_s:.0f}s", True)
        except Exception as e:
            return AgentReply(None, spec.max_budget_usd, 0, f"provider failed: {type(e).__name__}: {e}", True)
