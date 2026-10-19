"""Short prefill chunks as decode rows (custom_torch_backend.decode_rows / commit_rows; the GPU path is
_prefill_as_decode): every token's row points at the block its position lands in, blocks are reserved once per
position in every layer, and lengths move on only on commit. The real-model check against the packed prefill is the
CUDA test at the bottom (MEGAKERNEL_TEST=1; run it in its own process: it loads the model)."""

import os

import pytest
import torch

from inference_server.backends.custom_torch_backend import commit_rows, decode_rows
from inference_server.models.paged_kv_cache import BlockPool, PagedKVCache

BS, H, D = 4, 2, 8


def _pools(n=3):
    return [BlockPool(64, BS, H, D, torch.float32, torch.device("cpu")) for _ in range(n)]


def _cache(pools, prefix):
    kv = PagedKVCache(pools=pools)
    for L in range(len(pools)):
        for _ in range(prefix):
            kv.append(L, torch.zeros(1, H, 1, D), torch.zeros(1, H, 1, D))
    return kv


def test_rows_point_at_each_positions_block_and_lengths_move_on_commit():
    pools = _pools()
    caches = [_cache(pools, 6), _cache(pools, 0), _cache(pools, 3)]
    chunks = [[11, 12, 13], [21, 22, 23, 24, 25], [31]]
    free_before = [len(p._free) for p in pools]
    tokens, pos, owner, table = decode_rows(caches, chunks, pools)
    assert tokens == [11, 12, 13, 21, 22, 23, 24, 25, 31]
    assert pos == [6, 7, 8, 0, 1, 2, 3, 4, 3]
    assert owner == [0, 0, 0, 1, 1, 1, 1, 1, 2]
    for t, (p, k) in enumerate(zip(pos, owner)):
        for i in range(len(pools)):
            assert table[k][i][p // BS] == caches[k].block_tables[i][p // BS]
    # blocks: row 0 grows 2 -> 3 (positions 6..8), row 1 0 -> 2, row 2 stays 1; once per layer
    assert [len(c.block_tables[0]) for c in caches] == [3, 2, 1]
    assert [free_before[i] - len(p._free) for i, p in enumerate(pools)] == [3, 3, 3]
    assert [c.seq_len for c in caches] == [6, 0, 3]
    commit_rows(caches, chunks, pools)
    assert [c.seq_len for c in caches] == [9, 5, 4]


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("MEGAKERNEL_TEST") != "1",
                    reason="needs CUDA and MEGAKERNEL_TEST=1 (loads the real model)")
def test_short_prefill_as_decode_matches_the_packed_prefill():
    """The same prompts with cached prefixes, suffixes prefilled both ways: the next token agrees and the suffix K/V
    written into the pools match."""
    from inference_server.backends import custom_torch_backend as ctb
    from inference_server.sampling import GREEDY
    name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
    b = ctb.CustomTorchBackend("cuda", name)
    b.load_model(name)
    text = ("The history of the printing press begins in the fifteenth century, when movable metal type made books "
            "cheap enough to spread literacy across Europe. ") * 40
    ids = b.tokenizer.encode(text)
    prompts = [ids[:300], ids[:1000], ids[:100]]                       # every prefix > 64 tokens: the packed path
    suffixes = [ids[300:312], ids[1000:1005], ids[100:120]]          # 37 tokens in all: the decode-rows path
    suffix = [len(x) for x in suffixes]

    def run(as_decode):
        ctb.PREFILL_AS_DECODE_TOKENS = 64 if as_decode else 0
        rows = [b.prefill_chunk_batch([(None, p, False, GREEDY)])[0][0] for p in prompts]       # the prefixes
        items = [(kv, x, True, GREEDY) for kv, x in zip(rows, suffixes)]
        return b.prefill_chunk_batch(items), b.last_prefill_timing

    got, t_got = run(True)
    ref, t_ref = run(False)
    assert t_got.get("as_decode") and not t_ref.get("as_decode"), (t_got, t_ref)
    assert [r[2] for r in got] == [r[2] for r in ref], ([r[2] for r in got], [r[2] for r in ref])
    assert [r[1] for r in got] == [r[1] for r in ref], ([r[1] for r in got], [r[1] for r in ref])
    # Layer 0's suffix K/V depend on the embedding alone; deeper layers can differ by far more than rounding once a
    # router near-tie picks another expert in some layer (seen 2026-10-19: next tokens equal, layer 47 K off by ~1).
    for (kg, _, n), (kr, _, _), s in zip(got, ref, suffix):
        for p in range(n - s, n):
            bg, br = kg.block_tables[0][p // 16], kr.block_tables[0][p // 16]
            torch.testing.assert_close(b.pools[0].k[bg, :, p % 16].float(), b.pools[0].k[br, :, p % 16].float(),
                                       atol=3e-2, rtol=3e-2, msg=lambda m, p=p: f"layer 0 position {p}: {m}")
            assert torch.equal(b.pools[0].v[bg, :, p % 16], b.pools[0].v[br, :, p % 16]), f"layer 0 V, position {p}"
