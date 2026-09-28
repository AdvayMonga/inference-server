"""The context an agent node is given: its input records, knowledge, and where it is in the graph."""

from __future__ import annotations

import json
from pathlib import Path

from ..method import graph
from ..method.state import Record

PROMPTS = Path(__file__).parent / "prompts"


def knowledge_for(inputs: dict[str, Record]) -> list[dict]:
    """Knowledge-base entries for this brief. NOT DESIGNED YET: returns none until we discuss it."""
    return []


def graph_map(here: str) -> str:
    """The graph as text, with the agent's own node marked."""
    lines = []
    for (src, outcome), dst in graph.EDGES.items():
        mark = "  <- you are here" if src == here else ""
        lines.append(f"{src} --{outcome}--> {dst}{mark}")
    return "\n".join(lines)


def render(node: str, inputs: dict[str, Record], extra: dict[str, str] | None = None) -> str:
    """The user-turn brief: deterministic sections only, no other agent's reasoning."""
    parts = [f"# Where you are\n{graph_map(node)}"]
    for type_, rec in inputs.items():
        parts.append(f"# {type_} (from {rec.node}, outcome {rec.outcome})\n"
                     f"{json.dumps(rec.body, indent=2, sort_keys=True)}")
    parts.append("# Knowledge\n" + json.dumps(knowledge_for(inputs), indent=2))
    for title, text in (extra or {}).items():
        parts.append(f"# {title}\n{text}")
    return "\n\n".join(parts)


def prompt(node: str) -> str:
    """The node's instructions (system prompt)."""
    return (PROMPTS / f"{node}.md").read_text()
