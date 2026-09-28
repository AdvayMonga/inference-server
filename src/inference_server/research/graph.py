"""The research loop as a fixed graph. Agents run inside nodes; they cannot edit this file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GRAPH_VERSION = 1


@dataclass(frozen=True)
class Node:
    kind: Literal["code", "agent", "human"]
    takes: tuple[str, ...]       # record types it must read from the turn state
    gives: str | None            # record type it appends
    outcomes: tuple[str, ...]    # labels it may end with; EDGES routes on them


NODES: dict[str, Node] = {
    "question":    Node("human", (), "Question", ("ok",)),
    "measure":     Node("code", ("Question",), "Panel", ("ok",)),
    "hypothesize": Node("agent", ("Question", "Panel"), "Hypothesis", ("ok", "objection")),
    "experiment":  Node("code", ("Hypothesis", "Panel"), "Verdict",
                        ("confirmed", "rejected", "anomaly")),
    "record":      Node("code", ("Hypothesis", "Verdict"), "KBEntry", ("confirmed", "rejected")),
    "ask_human":   Node("human", (), None, ("stopped",)),
}

ENTRY = "question"
TERMINAL = {"record", "ask_human"}   # a turn ends here; the next turn starts fresh at ENTRY

EDGES: dict[tuple[str, str], str] = {
    ("question", "ok"):             "measure",
    ("measure", "ok"):              "hypothesize",
    ("hypothesize", "ok"):          "experiment",
    ("hypothesize", "objection"):   "ask_human",
    ("experiment", "confirmed"):    "record",
    ("experiment", "rejected"):     "record",
    ("experiment", "anomaly"):      "ask_human",
}


def next_node(node: str, outcome: str) -> str | None:
    """Where the graph goes after `node` ends with `outcome`; None when `node` is terminal."""
    if node in TERMINAL:
        return None
    return EDGES[(node, outcome)]
