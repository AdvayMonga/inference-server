"""Control plane: the routing decision and the global prefix index. CPU only, no model.

The numbers in the tradeoff tests are worked through the formula in `router.py`'s docstring on
purpose — if the formula changes, these fail loudly rather than drifting.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from control_plane.prefix_index import PrefixIndex
from control_plane.router import ReplicaView, RequestView, Router, replica_score, route

BS = 4                                    # block size used throughout
P = list(range(8))                        # a 2-block prompt
SRC = Path(__file__).resolve().parents[1] / "src" / "control_plane"


def idx(*inserts: tuple[str, list[int]], block_size: int = BS) -> PrefixIndex:
    """An index with (replica, prompt) entries already confirmed at tick 0."""
    index = PrefixIndex(block_size)
    for replica_id, prompt in inserts:
        index.insert(replica_id, prompt)
    return index


# ------------------------------------------------------------------ the index

def test_lookup_is_longest_match_first():
    index = idx(("a", P), ("b", P[:4]))
    assert [(m.replica_id, m.matched_blocks) for m in index.lookup(P)] == [("a", 2), ("b", 1)]


def test_partial_block_is_not_counted():
    index = idx(("a", P[:4]))
    # 6 tokens = one full block plus two: only the full block can match.
    m = index.lookup(P[:6])
    assert [(x.replica_id, x.matched_blocks) for x in m] == [("a", 1)]
    assert index.n_blocks(P[:6]) == 1
    # A prompt shorter than one block has nothing shareable at all.
    assert index.lookup(P[:3]) == []
    assert index.n_blocks(P[:3]) == 0


def test_insert_shorter_than_a_block_stores_nothing():
    index = PrefixIndex(BS)
    index.insert("a", P[:3])
    assert index.lookup(P) == []


def test_insert_respects_n_blocks_held():
    index = PrefixIndex(BS)
    index.insert("a", P, n_blocks=1)       # replica holds only the first block
    assert [m.matched_blocks for m in index.lookup(P)] == [1]


def test_entries_record_when_they_were_confirmed():
    index = PrefixIndex(BS)
    index.insert("a", P)                   # at tick 0
    index.advance(5)
    index.insert("b", P, as_of=3)          # a report that was true two ticks ago
    ages = {m.replica_id: index.tick - m.confirmed_at for m in index.lookup(P)}
    assert ages == {"a": 5, "b": 2}


def test_lookup_does_not_rebuild_every_aligned_prefix():
    """Complexity pin. Building a key per block boundary costs 157 ms on this prompt; the
    `_lengths` skip PrefixCache uses brings it to ~0.2 ms, so 20 ms is a 100x margin."""
    ids = list(range(32768))
    index = PrefixIndex(16)
    index.insert("a", ids[:16384])
    start = time.perf_counter()
    assert [m.matched_blocks for m in index.lookup(ids)] == [1024]
    elapsed = time.perf_counter() - start
    assert elapsed < 0.02, f"one 32k-token lookup took {elapsed * 1000:.0f} ms"


def test_length_bookkeeping_survives_evict_and_reinsert():
    """The skip must never skip a length that is still live, or a real hit reads as a miss."""
    index = idx(("a", P), ("b", P))
    index.evict("a", P)
    assert [m.replica_id for m in index.lookup(P)] == ["b"]
    index.evict("b", P)
    assert index.lookup(P) == []
    index.insert("a", P)
    assert [m.replica_id for m in index.lookup(P)] == ["a"]


def test_evict_forgets_one_entry():
    index = idx(("a", P), ("b", P))
    index.evict("a", P)
    assert [m.replica_id for m in index.lookup(P)] == ["b"]


def test_tokenize_is_the_callers():
    with pytest.raises(TypeError):
        PrefixIndex(BS).lookup("a prompt")
    index = PrefixIndex(2, tokenize=lambda s: [ord(c) for c in s])
    index.insert("a", "abcd")
    assert [(m.replica_id, m.matched_blocks) for m in index.lookup("abcdef")] == [("a", 2)]


# ------------------------------------------------------------------ the endpoints of the knob

def test_weight_zero_picks_the_shortest_queue_whatever_the_locality():
    replicas = [ReplicaView("a", queue_depth=10, sessions=frozenset({"s"})),
                ReplicaView("b", queue_depth=1)]
    req = RequestView("s", P)
    assert route(req, replicas, idx(("a", P)), weight=0.0) == "b"


def test_weight_one_picks_the_best_prefix_match_whatever_the_load():
    replicas = [ReplicaView("a", queue_depth=100), ReplicaView("b", queue_depth=0)]
    index = idx(("a", P), ("b", P[:4]))
    assert route(RequestView("s", P), replicas, index, weight=1.0) == "a"


def test_mid_weight_trades_off_as_documented():
    # a: full prefix (locality 0.5), load 10 of 10 (load term 0). b: nothing, load 0 (term 1).
    # a wins once w*0.5 > (1-w)*1, i.e. above w = 2/3.
    replicas = [ReplicaView("a", queue_depth=10), ReplicaView("b", queue_depth=0)]
    index = idx(("a", P))
    req = RequestView("s", P)
    assert route(req, replicas, index, weight=0.5) == "b"
    assert route(req, replicas, index, weight=0.8) == "a"
    assert replica_score(replicas[0], "s", 2, 2, 10, 0.8) == pytest.approx(0.4)
    assert replica_score(replicas[1], "s", 0, 2, 10, 0.8) == pytest.approx(0.2)


# ------------------------------------------------------------------ session affinity

def test_affinity_beats_a_marginally_better_prefix_match_elsewhere():
    long_prompt = list(range(16))                      # 4 blocks
    replicas = [ReplicaView("a", sessions=frozenset({"s"})), ReplicaView("b")]
    index = idx(("b", long_prompt[:12]))               # b has 3 of 4 blocks, a is not indexed
    assert route(RequestView("s", long_prompt), replicas, index, weight=1.0) == "a"


def test_affinity_weight_zero_removes_affinity_entirely():
    long_prompt = list(range(16))
    replicas = [ReplicaView("a", sessions=frozenset({"s"})), ReplicaView("b")]
    index = idx(("b", long_prompt[:12]))
    req = RequestView("s", long_prompt)
    assert route(req, replicas, index, weight=1.0, affinity_weight=0.0) == "b"


# ------------------------------------------------------------------ determinism and liveness

def test_ties_break_on_replica_id_whatever_the_input_order():
    index = idx()
    views = [ReplicaView(r) for r in ("r2", "r1", "r3")]
    req = RequestView("s", P)
    assert route(req, views, index, weight=0.5) == "r1"
    assert route(req, list(reversed(views)), index, weight=0.5) == "r1"


def test_a_full_replica_is_skipped_while_another_has_room():
    replicas = [ReplicaView("a", free_blocks=0, sessions=frozenset({"s"})),
                ReplicaView("b", queue_depth=50)]
    req = RequestView("s", P)
    assert route(req, replicas, idx(("a", P)), weight=1.0) == "b"
    # ...but with nowhere else to go, a full replica is still a decision, not an error.
    assert route(req, replicas[:1], idx(("a", P)), weight=1.0) == "a"


def test_no_replicas_is_none():
    assert route(RequestView("s", P), [], idx(), weight=0.5) is None


# ------------------------------------------------------------------ the failure mode

def test_a_stale_index_routes_to_a_replica_that_no_longer_holds_the_prefix():
    """The point of the staleness curve: the index can be confidently wrong, and does not know."""
    index = idx(("a", P), ("b", P[:4]))    # a holds both blocks, b only the first
    replicas = [ReplicaView("a", queue_depth=5), ReplicaView("b")]
    req = RequestView("s", P)
    assert route(req, replicas, index, weight=1.0) == "a"

    index.advance(10)                      # 'a' dropped the blocks ten ticks ago; nobody told us
    assert route(req, replicas, index, weight=1.0) == "a"
    assert {m.replica_id: index.tick - m.confirmed_at for m in index.lookup(P)} == {"a": 10,
                                                                                    "b": 10}

    index.evict("a", P)                    # the late report finally lands
    assert route(req, replicas, index, weight=1.0) == "b"


def test_drop_replica_removes_it_from_every_future_decision():
    router = Router(BS, weight=1.0)
    router.on_replica_state(ReplicaView("a", sessions=frozenset({"s"})))
    router.on_replica_state(ReplicaView("b", queue_depth=99))
    router.index.insert("a", P)
    req = RequestView("s", P)
    assert router.on_request(req) == "a"

    router.on_replica_gone("a")
    assert router.on_request(req) == "b"
    assert router.index.lookup(P) == []


def test_router_shell_adds_no_policy_of_its_own():
    router = Router(BS, weight=0.3, affinity_weight=0.7)
    views = [ReplicaView("a", queue_depth=4), ReplicaView("b", sessions=frozenset({"s"}))]
    for v in views:
        router.on_replica_state(v)
    router.index.insert("a", P)
    req = RequestView("s", P)
    assert router.on_request(req) == route(req, views, router.index, 0.3, 0.7)


# ------------------------------------------------------------------ isolation

def test_control_plane_does_not_import_the_engine():
    """Mirrors the CI loop lane: this package must run with no torch anywhere on the machine."""
    allowed = {"inference_server.scheduling_policy"}
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            offenders += [f"{path.name}: {n}" for n in names
                          if n.split(".")[0] == "inference_server" and n not in allowed]
    assert offenders == []
