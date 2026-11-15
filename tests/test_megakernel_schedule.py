"""The instruction interpreter's schedules (megakernel/framework.py) from the default program (megakernel/program):
every phase's work covered exactly once, every queue runnable to its end under the counters, each instruction's
inputs complete and their earlier readers done before it starts whatever the interleaving (vector clocks over the
counters), and each block's expert instructions written by its own router first. CPU only."""

import pytest

from inference_server.models.megakernel import program as P
from inference_server.models.megakernel.framework import Builder

ORDER = (P.QKV, P.ATTN, P.COMBINE, P.OPROJ, P.POST, P.TOPK, P.GATEUP, P.DOWN, P.LMHEAD, P.ARGMAX)
ACCESS = {   # op -> (scratch it reads, scratch it writes), mega_ops.cuh; a layer's K/V pool is its ATTN's alone
    P.QKV: ({"h1", "out2"}, {"qkv", "h"}), P.ATTN: ({"qkv"}, {"attn", "part"}), P.COMBINE: ({"part"}, {"attn"}),
    P.OPROJ: ({"attn"}, {"o"}), P.POST: ({"h", "o"}, {"h1", "n2", "out2", "rlogits"}), P.TOPK: ({"rlogits"}, {"ids"}),
    P.GATEUP: ({"n2"}, {"act"}), P.DOWN: ({"act", "out2"}, {"out2"}), P.LMHEAD: ({"h1", "out2"}, {"logits", "keys"}),
    P.ARGMAX: ({"keys"}, {"keys", "tokens"}),
}
CASES = [(B, blocks) for B in (1, 2, 3, 8, 16) for blocks in (132, 140)] + [(1, 20)]
LAYERS, VOCAB = 3, 1000


def _run(s, skip=()):
    """Every queue run under the counters (ops in `skip` wait on nothing, as COMBINE does when every row fits one
    chunk); each instruction's vector clock at its start: per block, how many of its instructions are done."""
    queues = s.queues()
    G = len(queues)
    count, full = [0] * s.n_counters, [[0] * G for _ in range(s.n_counters)]
    pos, clock, start = [0] * G, [[0] * G for _ in range(G)], {}
    progress = True
    while progress:
        progress = False
        for j in range(G):
            while pos[j] < len(queues[j]):
                r = queues[j][pos[j]]
                waits = [] if r[0] in skip else [(c, n) for c, n in ((r[2], r[3]), (r[4], r[5])) if c >= 0]
                if any(count[c] < n for c, n in waits):
                    break
                vc = clock[j]
                for c, _ in waits:
                    vc = [max(x, y) for x, y in zip(vc, full[c])]
                start[j, pos[j]] = vc
                clock[j] = vc[:j] + [pos[j] + 1] + vc[j + 1:]
                if r[6] >= 0:
                    count[r[6]] += 1
                    full[r[6]] = [max(x, y) for x, y in zip(full[r[6]], clock[j])]
                pos[j] += 1
                progress = True
    assert pos == [len(q) for q in queues], "deadlock: some queue cannot reach its end"
    return queues, start


def _ranges(queues, op, layer, a=0):
    return sorted((r[7 + a], r[8 + a]) for q in queues for r in q if r[0] == op and r[1] == layer and r[7 + a] < r[8 + a])


def _partition(ranges, n):
    assert ranges[0][0] == 0 and ranges[-1][1] == n and all(x[1] == y[0] for x, y in zip(ranges, ranges[1:])), ranges


@pytest.mark.parametrize("B,blocks", CASES)
@pytest.mark.parametrize("fp8", [False, True])
def test_every_phase_covered_once(B, blocks, fp8):
    queues = P.schedule(B, blocks, LAYERS, VOCAB, fp8).queues()
    pairs = B * P.K
    for L in range(LAYERS):
        _partition(_ranges(queues, P.QKV, L), P.QKV_ROWS)
        _partition(_ranges(queues, P.OPROJ, L), P.H)
        _partition(_ranges(queues, P.POST, L, 1), B * P.E)
        for op, rows in ((P.GATEUP, P.I), (P.DOWN, P.H)):
            _partition(_ranges(queues, op, L), pairs * rows)
            for q in queues:
                for r in q:
                    if r[0] == op and r[1] == L:
                        assert r[9] == r[7] // rows and (r[8] - 1) // rows <= r[9] + 1, "at most two pairs, the first in arg2"
                        assert not fp8 or op == P.DOWN or r[7] % 2 == r[8] % 2 == 0, "whole fp8 warp items"
        for op, a in ((P.QKV, 2), (P.POST, 0)):
            assert sorted(r[7 + a] for q in queues for r in q if r[0] == op and r[1] == L and r[7 + a] >= 0) == list(range(B))
        for op in (P.ATTN, P.COMBINE):
            assert sorted((r[7], r[8]) for q in queues for r in q if r[0] == op and r[1] == L) == [(j, blocks) for j in range(blocks)]
    _partition(_ranges(queues, P.LMHEAD, LAYERS), VOCAB)
    assert [j for j, q in enumerate(queues) for r in q if r[0] == P.ARGMAX] == [0]


@pytest.mark.parametrize("B,blocks", CASES)
@pytest.mark.parametrize("skip", [(), (P.COMBINE,)])
def test_no_deadlock_and_no_race(B, blocks, skip):
    queues, start = _run(P.schedule(B, blocks, LAYERS, VOCAB, True), skip)
    groups: dict[tuple, dict[int, int]] = {}          # (layer, op) -> {block: last position}
    for j, q in enumerate(queues):
        for p, r in enumerate(q):
            groups.setdefault((r[1], r[0]), {})[j] = p
    access = {g: ((set(), set()) if g[1] in skip else ACCESS[g[1]]) for g in groups}
    required: dict[tuple, set] = {g: set() for g in groups}
    last_write, readers = {}, {}
    for g in sorted(groups, key=lambda g: (g[0], ORDER.index(g[1]))):   # program order
        reads, writes = access[g]
        for x in reads | writes:
            if x in last_write:
                required[g].add(last_write[x])
            if x in writes:
                required[g] |= set(readers.get(x, ()))
        for x in reads - writes:
            readers.setdefault(x, []).append(g)
        for x in writes:
            last_write[x], readers[x] = g, []
    for j, q in enumerate(queues):
        for p, r in enumerate(q):
            vc = start[j, p]
            for need in required[(r[1], r[0])]:
                late = [k for k, last in groups[need].items() if last >= vc[k]]
                assert not late, f"block {j} op {r[0]} layer {r[1]} may start before {need} is done in blocks {late[:5]}"


@pytest.mark.parametrize("B,blocks", CASES)
def test_each_router_fills_its_own_expert_instructions_first(B, blocks):
    for q in P.schedule(B, blocks, LAYERS, VOCAB, False).queues():
        filled = set()
        for p, r in enumerate(q):
            if r[0] == P.TOPK:
                filled |= {(r[1], p + r[7] + s) for s in range(r[8])}
            if r[0] in (P.GATEUP, P.DOWN):
                assert (r[1], p) in filled, (p, r)
        assert sum(r[0] in (P.GATEUP, P.DOWN) for r in q) == len(filled)


def test_builder_waits_count_every_signaller():
    b = Builder(3)
    for j in range(3):
        b.add(j, 1, 0, (j,), [], "x")
    b.add(1, 2, 0, (7, 8), ["x"], "y")
    b.add(2, 3, 0, (), ["x", "y"])
    s = b.build()
    assert s.offsets.tolist() == [0, 1, 3, 5] and s.n_counters == 2 and s.instrs.shape == (5, 16)
    assert s.instrs[2].tolist()[:9] == [2, 0, 0, 3, -1, 0, 1, 7, 8]
    assert s.instrs[4].tolist()[:7] == [3, 0, 0, 3, 1, 1, -1]
