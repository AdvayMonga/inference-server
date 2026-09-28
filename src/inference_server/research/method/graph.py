"""The research loop as a fixed graph. Agents run inside nodes; they cannot edit this file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GRAPH_VERSION = 3


@dataclass(frozen=True)
class Node:
    kind: Literal["code", "agent", "human"]
    takes: tuple[str, ...]       # record types it must read from the turn state
    gives: str | None            # record type it appends
    outcomes: tuple[str, ...]    # labels it may end with; EDGES routes on them
    max_visits: int | None = None   # per turn; past it the turn goes to GIVE_UP
    optional: tuple[str, ...] = ()  # record types it also reads when the turn has them


NODES: dict[str, Node] = {
    "question":    Node("human", (), "Question", ("ok",)),
    "measure":     Node("code", ("Question",), "Panel", ("ok",)),
    "hypothesize": Node("agent", ("Question", "Panel"), "Hypothesis", ("ok", "objection")),
    # code-edit subgraph
    "build":       Node("agent", ("Hypothesis", "Panel"), "Change", ("ok", "objection"),
                        max_visits=3, optional=("CheckReport", "Review")),
    "check":       Node("code", ("Change",), "CheckReport", ("pass", "fail")),
    "review_1":    Node("agent", ("Hypothesis", "Change", "CheckReport"), "Review",
                        ("ok", "changes", "objection")),
    "experiment":  Node("code", ("Hypothesis", "Change"), "Results", ("done", "anomaly")),
    "review_2":    Node("agent", ("Hypothesis", "Change", "Results"), "Judgement",
                        ("accept", "reject", "objection")),
    "publish":     Node("code", ("Change", "Judgement"), "PullRequest", ("ok",)),
    # end of turn
    "record":      Node("code", ("Hypothesis",), "KBEntry", ("ok",)),
    "ask_human":   Node("human", (), None, ("stopped",)),
}

ENTRY = "question"
TERMINAL = {"record", "ask_human"}   # a turn ends here; the next turn starts fresh at ENTRY
GIVE_UP = "record"                   # where the turn goes when a node is past its max_visits

EDGES: dict[tuple[str, str], str] = {
    ("question", "ok"):             "measure",
    ("measure", "ok"):              "hypothesize",
    ("hypothesize", "ok"):          "build",
    ("hypothesize", "objection"):   "ask_human",
    ("build", "ok"):                "check",
    ("build", "objection"):         "ask_human",
    ("check", "pass"):              "review_1",
    ("check", "fail"):              "build",
    ("review_1", "ok"):             "experiment",
    ("review_1", "changes"):        "build",
    ("review_1", "objection"):      "ask_human",
    ("experiment", "done"):         "review_2",
    ("experiment", "anomaly"):      "ask_human",
    ("review_2", "accept"):         "publish",
    ("review_2", "reject"):         "record",
    ("review_2", "objection"):      "ask_human",
    ("publish", "ok"):              "record",
}


def next_node(node: str, outcome: str) -> str | None:
    """Where the graph goes after `node` ends with `outcome`; None when `node` is terminal."""
    if node in TERMINAL:
        return None
    return EDGES[(node, outcome)]
