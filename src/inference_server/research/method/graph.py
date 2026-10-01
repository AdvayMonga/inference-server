"""The research loop as a fixed graph. Agents run inside nodes; they cannot edit this file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GRAPH_VERSION = 5


@dataclass(frozen=True)
class Node:
    kind: Literal["code", "agent", "human"]
    takes: tuple[str, ...]       # record types it must read from the turn state
    gives: str | None            # record type it appends
    outcomes: tuple[str, ...]    # labels it may end with; EDGES routes on them
    optional: tuple[str, ...] = ()  # record types it also reads when the turn has them


NODES: dict[str, Node] = {
    "question":    Node("human", (), "Question", ("ok",)),
    "measure":     Node("code", ("Question",), "Panel", ("ok",)),
    "hypothesize": Node("agent", ("Question", "Panel"), "Hypothesis", ("ok",)),
    # code-edit subgraph
    "build":       Node("agent", ("Hypothesis", "Panel"), "Change", ("ok", "crashed"),
                        optional=("CheckReport", "Review", "Judgement", "Profile")),
    "check":       Node("code", ("Change",), "CheckReport", ("pass", "fail", "violation")),
    "review_1":    Node("agent", ("Hypothesis", "Change", "CheckReport"), "Review",
                        ("ok", "fix", "changes", "crashed"), optional=("Review",)),
    "experiment":  Node("code", ("Hypothesis", "Change"), "Results", ("done", "anomaly")),
    "profile":     Node("code", ("Panel", "Results"), "Profile", ("ok",)),
    "review_2":    Node("agent", ("Hypothesis", "Change", "Results", "Profile"), "Judgement",
                        ("accept", "revise", "reject", "crashed")),
    "publish":     Node("code", ("Change", "Judgement"), "PullRequest", ("ok",)),
    # end of turn
    "record":      Node("code", ("Hypothesis",), "KBEntry", ("ok",)),
    "ask_human":   Node("human", (), None, ("stopped",)),
}

ENTRY = "question"
TERMINAL = {"record", "ask_human"}   # a turn ends here; the next turn starts fresh at ENTRY
GIVE_UP = "record"                   # where the turn goes when a budget below runs out

EDGES: dict[tuple[str, str], str] = {
    ("question", "ok"):             "measure",
    ("measure", "ok"):              "hypothesize",
    ("hypothesize", "ok"):          "build",
    ("build", "ok"):                "check",
    ("build", "crashed"):           "build",         # the agent failed, not the idea
    ("check", "pass"):              "review_1",
    ("check", "fail"):              "review_1",      # the reviewer diagnoses the failure
    ("check", "violation"):         "ask_human",     # touched what grades it: you look first
    ("review_1", "ok"):             "experiment",
    ("review_1", "fix"):            "build",         # the check failed: a code error, not a round
    ("review_1", "changes"):        "build",
    ("review_1", "crashed"):        "review_1",
    ("experiment", "done"):         "profile",
    ("experiment", "anomaly"):      "ask_human",     # a win too big to believe
    ("profile", "ok"):              "review_2",
    ("review_2", "accept"):         "publish",
    ("review_2", "revise"):         "build",
    ("review_2", "reject"):         "record",
    ("review_2", "crashed"):        "review_2",
    ("publish", "ok"):              "record",
}

# Budgets per turn, enforced by TurnState. A round is a send-back the reviewers chose.
ROUNDS = {("review_1", "changes"), ("review_2", "revise")}
MAX_ROUNDS = 6
MAX_EXPERIMENTS = 2        # each re-measurement is another chance for noise to look like a win
MAX_NODE_VISITS = 40       # PLACEHOLDER backstop for free loops until the dollar budget exists


def next_node(node: str, outcome: str) -> str | None:
    """Where the graph goes after `node` ends with `outcome`; None when `node` is terminal."""
    if node in TERMINAL:
        return None
    return EDGES[(node, outcome)]
