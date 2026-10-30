"""The interpreter's default program (megakernel/framework.py): today's megakernel (decode_step.cu) as instructions,
each block taking the share of every phase it takes there, with the same arithmetic (mega_ops.cuh), so outputs match
it. Per layer and block: QKV (residual, input norm, a range of qkv rows), ATTN (q/k norm and rope, the new token's
K/V written, strided attention items), COMBINE (chunks of rows longer than one chunk), OPROJ, POST (residual, post
norm, router logits), TOPK (the router: softmax and top-8, run by every block, which writes the experts into its own
GATEUP and DOWN instructions that follow), GATEUP (+ silu_mul) and DOWN (weighted, accumulated); then LMHEAD (final
norm, lm_head) and ARGMAX. Each phase waits on the counter of the phase before it: today's barrier order, without the
barriers."""

from __future__ import annotations

from pathlib import Path

from inference_server.models.megakernel.framework import Builder, Schedule

OPS = Path(__file__).with_name("mega_ops.cuh")
QKV, ATTN, COMBINE, OPROJ, POST, TOPK, GATEUP, DOWN, LMHEAD, ARGMAX = range(1, 11)    # mega_ops.cuh's opcodes
H, NQ, NKV, D, I, E, K = 2048, 32, 4, 128, 768, 128, 8
QKV_ROWS = (NQ + 2 * NKV) * D


def block_slice(n: int, blocks: int, j: int, unit: int = 1) -> tuple[int, int]:
    """Block j's [lo, hi) of n rows split as evenly as whole `unit`s allow (decode_step.cu's block_slice)."""
    per = -(-n // unit // blocks) * unit
    lo = min(n, j * per)
    return lo, min(n, lo + per)


def schedule(B: int, blocks: int, n_layers: int, vocab: int, fp8: bool) -> Schedule:
    b = Builder(blocks)
    pairs = B * K
    unit = 2 if fp8 else 1                                    # GATEUP's register path takes 2 fp8 rows a warp item
    prev = []                                                 # what the next layer's QKV waits on
    for L in range(n_layers):
        qkv, attn, comb, oproj, post, gu, down = (f"{n}{L}" for n in ("qkv", "attn", "combine", "oproj", "post", "gateup", "down"))
        for j in range(blocks):
            hrow = j if j < B else -1
            b.add(j, QKV, L, (*block_slice(QKV_ROWS, blocks, j), hrow), prev, qkv)
            b.add(j, ATTN, L, (j, blocks), [qkv], attn)
            b.add(j, COMBINE, L, (j, blocks), [attn], comb)
            b.add(j, OPROJ, L, block_slice(H, blocks, j), [comb], oproj)   # comb: every block past its ATTN
            lo, hi = block_slice(B * E, blocks, j)
            if lo < hi or hrow >= 0:
                b.add(j, POST, L, (hrow, lo, hi), [oproj], post)
            slots = [(op, r, rows) for op, r, rows in ((GATEUP, block_slice(pairs * I, blocks, j, unit), I),
                                                      (DOWN, block_slice(pairs * H, blocks, j), H)) if r[0] < r[1]]
            if slots:
                b.add(j, TOPK, L, (1, len(slots), int(j == 0)), [post])
            for op, (lo, hi), rows in slots:
                assert (hi - 1) // rows <= lo // rows + 1, "an expert instruction spans at most two pairs"
                b.add(j, op, L, (lo, hi, lo // rows), [gu] if op == DOWN else [], gu if op == GATEUP else down)
        prev = [down]
    for j in range(blocks):
        b.add(j, LMHEAD, n_layers, block_slice(vocab, blocks, j), prev, "lmhead")
    b.add(0, ARGMAX, n_layers, (), ["lmhead"])
    return b.build()
