"""Tier-1 (free, analytical): is prefill near its arithmetic ceiling, and does bucket padding
actually cost anything?

Falsifies two hypotheses without touching a GPU — which is the screening ladder working.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Gemma 4 E4B
LAYERS, HQ, HKV = 42, 8, 2
D_SLIDING, D_FULL = 256, 512
DENSE_PARAMS = 4.59e9          # streamed per token (measured, roofline_e4b_a100.txt)
A100_BF16_PEAK = 312e12
ACHIEVABLE = 0.50              # a well-tuned GEMM stack; generous for a hand-written kernel

# Observed workload: ShareGPT-ish lognormal(5.48, 0.75)
P50 = math.exp(5.48)
P95 = math.exp(5.48 + 1.645 * 0.75)
BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024)


def bucket_for(n: float) -> int | None:
    return next((b for b in BUCKETS if n <= b), None)


def attention_flops(s: float) -> float:
    """Causal attention: QK^T + PV over the lower triangle, per layer."""
    per_layer = 2 * (s * s / 2) * HQ * D_SLIDING * 2
    return per_layer * LAYERS


def gemm_flops(s: float) -> float:
    """Every prefill token passes the dense weights once."""
    return 2 * s * DENSE_PARAMS


def main() -> int:
    print(f"workload: p50={P50:.0f} tok, p95={P95:.0f} tok\n")

    # ---- H2: does bucket padding cost real work? ------------------------------------
    b = bucket_for(P95)
    pad = b - P95
    print("H2  'add a bucket near 896 so ~830 stops rounding into 1024'")
    print(f"    p95 {P95:.0f} tok -> bucket {b}: {pad:.0f} padded positions ({pad / b:.1%})")
    print("    BUT the prefill kernel exits padding queries immediately:")
    print("        `if qj >= suffix_len: return`  (paged_attention_kernel.py)")
    print(f"    so padding costs {pad:.0f} x {HQ} = {pad * HQ:.0f} grid slots that do no work,")
    print("    not a share of the compute. Padding is NOT the prefill cost.")
    print("    => FALSIFIED at tier 1, $0 spent\n")

    # ---- H3: how far is prefill from its ceiling? -----------------------------------
    for label, s in (("p50", P50), ("p95", P95)):
        att, gemm = attention_flops(s), gemm_flops(s)
        total = att + gemm
        floor_ms = total / (A100_BF16_PEAK * ACHIEVABLE) * 1000
        print(f"H3  {label} prompt = {s:.0f} tok")
        print(f"    attention {att / 1e12:6.2f} TFLOP   GEMM {gemm / 1e12:6.2f} TFLOP   "
              f"total {total / 1e12:6.2f} TFLOP")
        print(f"    arithmetic floor @ {ACHIEVABLE:.0%} of A100 peak: {floor_ms:6.1f} ms")
        print(f"    attention share of the work: {att / total:.1%}")
    measured_p95 = 218.4      # runs/*.json, rate 2, this repo
    floor_p95 = (attention_flops(P95) + gemm_flops(P95)) / (A100_BF16_PEAK * ACHIEVABLE) * 1000
    print(f"\n    measured prefill p95 = {measured_p95:.0f} ms vs floor {floor_p95:.0f} ms "
          f"=> {measured_p95 / floor_p95:.1f}x off the ceiling")
    print(f"    attention is only {attention_flops(P95) / (attention_flops(P95) + gemm_flops(P95)):.0%} "
          f"of the FLOPs, so 'O(S^2) attention is irreducible' does NOT explain the gap")
    print("    => NOT falsified: a real lever exists. Escalate to tier 3 to find where it goes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
