"""Correctness of BatchedDecodeState's de-Pythoned block bookkeeping (CPU; no kernel/model).

Writes position-valued K/V across multi-row decode steps that cross block boundaries, then
verifies every position reads back exactly and that removing rows returns every block. This
validates the scatter / prepare_step alloc / advance logic that replaces the per-step Python —
the part that must be right before the GPU kernel path uses it.
"""

from __future__ import annotations

import torch

from inference_server.models.paged_kv_cache import BatchedDecodeState, PagedKVCache, pool_group

DEV = torch.device("cpu")
BS, D = 4, 2  # block_size, head_dim


def _pools():
    return pool_group(2, 256, BS, 1, D, torch.float32, DEV)


def _val(pos):  # deterministic per-position K/V value
    return torch.full((1, 1, 1, D), float(pos))


def test_state_decode_bookkeeping():
    pools = _pools()
    free0 = [p.free_count for p in pools]
    state = BatchedDecodeState(pools, DEV)

    # Two rows with different prefill lengths.
    prefill_lens = [6, 11]
    for P in prefill_lens:
        pkv = PagedKVCache(pools=pools)
        for L, pool in enumerate(pools):
            for pos in range(P):
                pkv.append(L, _val(pos), _val(pos))
        state.add_row(pkv)  # adopt (do NOT free_all — state owns the blocks now)

    # Decode 12 steps in lockstep (each row's new token = its current seq_len position).
    STEPS = 12
    for _ in range(STEPS):
        state.prepare_step()
        seq = state.seq_lens.tolist()
        for L, pool in enumerate(pools):
            R = state.n_rows
            k = torch.stack([_val(seq[r])[0] for r in range(R)])  # [R,1,1,D]
            state.append(L, k, k)
        state.advance()

    # Verify: every position reads back its value.
    for r, P in enumerate(prefill_lens):
        final_len = P + STEPS
        for L, pool in enumerate(pools):
            n_alloc = int(state.n_alloc[r].item())
            for pos in range(final_len):
                lb = pos // BS
                if lb >= n_alloc:
                    continue
                bid = int(state.block_tables[L][r, lb].item())
                slot = pos % BS
                got = pool.k[bid, 0, slot, 0].item()
                assert got == float(pos), f"row{r} L{L} pos{pos}: read {got} != {pos}"

    # Remove all rows → every block returns to the pool (no leak / double-free).
    while state.n_rows:
        state.remove_row(0)
    assert [p.free_count for p in pools] == free0, "blocks leaked or double-freed"
