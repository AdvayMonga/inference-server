"""Analytical roofline+ for LLM decode — offline ceiling, per-step breakdown,
ranked lever menu, gap attribution. No GPU, no weights. See PLAN.md P0.

The model: per decode step you (a) stream the dense weights from HBM once for the
whole batch, (b) stream each sequence's KV cache, (c) do 2*N_params*B FLOP of math.
Ideal step time = max(compute_time, memory_time); tok/s = B / step_time. The gap
between that ceiling and what we measure is overhead (launches, data-movement).
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Hardware presets. flops = dense bf16 tensor-core FLOP/s; bw = HBM bytes/s.
# fp8_compute = has native fp8 tensor cores (A100 does NOT — quant is bytes-only there).
# ---------------------------------------------------------------------------
@dataclass
class HW:
    name: str
    flops: float          # bf16 FLOP/s
    bw: float             # HBM bytes/s
    mem: float            # HBM bytes
    fp8_compute: bool     # native fp8 tensor cores (doubles compute roof)

HARDWARE = {
    "A100-80GB": HW("A100-80GB", 312e12, 2.039e12, 80e9, fp8_compute=False),
    "A10G":      HW("A10G",      125e12, 0.600e12, 24e9, fp8_compute=False),
    "H100-80GB": HW("H100-80GB", 990e12, 3.350e12, 80e9, fp8_compute=True),
}

DTYPE_BYTES = {"bf16": 2, "fp8": 1, "int8": 1}


# ---------------------------------------------------------------------------
# Model spec. Values read from each model's HF config.json (cited inline).
# ---------------------------------------------------------------------------
@dataclass
class Model:
    name: str
    n_layers: int
    hidden: int
    intermediate: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int           # sliding-layer head_dim
    global_head_dim: int    # full-layer head_dim
    sliding_window: int
    vocab: int
    n_kv_shared: int        # trailing layers that reuse an earlier layer's KV
    hidden_per_layer: int   # PLE per-layer embedding width
    full_every: int = 6     # a full-attention layer every Nth (index % full_every == full_every-1)

    @property
    def first_shared(self) -> int:
        return self.n_layers - self.n_kv_shared

    def is_full(self, i: int) -> bool:
        return i % self.full_every == self.full_every - 1

    def stores_kv(self, i: int) -> bool:
        return i < self.first_shared  # trailing shared layers hold no KV of their own

    def layer_hd(self, i: int) -> int:
        return self.global_head_dim if self.is_full(i) else self.head_dim


# google/gemma-4-E4B-it/config.json (42L, 8:2 GQA, hd 256/512, window 512, 18 shared)
E4B = Model("E4B", 42, 2560, 10240, 8, 2, 256, 512, 512, 262144, 18, 256)
# google/gemma-4-E2B-it/config.json (hidden 1536; smaller twin — for cross-model sanity)
E2B = Model("E2B", 30, 1536, 8192, 4, 1, 256, 256, 512, 262144, 12, 256, full_every=6)


# ---------------------------------------------------------------------------
# Empirical gap attribution — isolated compiled decode, A100/E4B, N=32
# (benchmarks/tuning_log.md profiling, 2026-06-14). Fractions of achieved step time.
# ---------------------------------------------------------------------------
PROFILE = {
    "attention (paged kernels)": 0.24,
    "GEMM (weight matmuls)":     0.23,
    "small-kernel launch tail":  0.19,
    "DtoD memcpy (KV/act)":      0.16,
    "KV scatter (index_put)":    0.15,
    "norm/rope/act":             0.04,
}
# Which buckets are pure overhead the roofline says shouldn't exist:
OVERHEAD_BUCKETS = ["small-kernel launch tail", "DtoD memcpy (KV/act)", "KV scatter (index_put)"]

# Measured tok/s to overlay (A100/E4B). Closed-loop = full serving sweep; isolated = decode-only.
MEASURED_CLOSED = {16: 658, 32: 1151}   # compile-default, tuning_log anchor
MEASURED_ISOLATED = {32: 1870}          # isolated decode, profile_decode_kernels_modal


# ---------------------------------------------------------------------------
# Core cost model
# ---------------------------------------------------------------------------
def dense_params(m: Model) -> dict:
    """Params streamed from HBM every decode step, split into buckets."""
    attn = mlp = 0
    for i in range(m.n_layers):
        hd = m.layer_hd(i)
        attn += m.hidden * m.n_q_heads * hd          # q_proj
        attn += m.n_q_heads * hd * m.hidden          # o_proj
        if m.stores_kv(i):
            attn += 2 * (m.hidden * m.n_kv_heads * hd)  # k_proj + v_proj
        mlp += 3 * m.hidden * m.intermediate          # gate + up + down
    lm_head = m.vocab * m.hidden                       # tied embed, read for logits
    proj = m.hidden * m.n_layers * m.hidden_per_layer  # per-layer-embed projection
    return {"attn": attn, "mlp": mlp, "lm_head": lm_head, "proj": proj,
            "total": attn + mlp + lm_head + proj}


def ple_table_params(m: Model) -> int:
    """PLE embedding table — GATHERED per token, not streamed. The 'effective vs total' gap."""
    return m.vocab * m.n_layers * m.hidden_per_layer


def kv_bytes(m: Model, B: int, L: int, dtype: str) -> int:
    """KV bytes for B sequences at context L — sliding-window + KV-sharing aware."""
    per_seq = 0
    for i in range(m.n_layers):
        if not m.stores_kv(i):
            continue
        toks = min(L, m.sliding_window) if not m.is_full(i) else L
        per_seq += 2 * m.n_kv_heads * m.layer_hd(i) * toks  # K + V
    return per_seq * DTYPE_BYTES[dtype] * B


def attn_flops(m: Model, B: int, L: int) -> float:
    """Decode attention FLOP (1 query attends L_eff keys) — small but grows with L."""
    f = 0
    for i in range(m.n_layers):
        L_eff = min(L, m.sliding_window) if not m.is_full(i) else L
        f += 2 * 2 * m.n_q_heads * m.layer_hd(i) * L_eff  # QK^T + AV
    return f * B


def step(m: Model, hw: HW, B: int, L: int, w_dtype: str = "bf16", kv_dtype: str = "bf16") -> dict:
    """Ideal per-step timing + regime. Times in seconds."""
    dp = dense_params(m)
    w_bytes = dp["total"] * DTYPE_BYTES[w_dtype]
    kvb = kv_bytes(m, B, L, kv_dtype)
    flops = 2 * dp["total"] * B + attn_flops(m, B, L)
    # fp8 doubles the compute roof only where the GPU has fp8 tensor cores.
    peak = hw.flops * (2 if (w_dtype == "fp8" and hw.fp8_compute) else 1)
    t_compute = flops / peak
    t_weight = w_bytes / hw.bw
    t_kv = kvb / hw.bw
    t_mem = t_weight + t_kv
    t_step = max(t_compute, t_mem)
    return {
        "t_compute": t_compute, "t_weight": t_weight, "t_kv": t_kv, "t_mem": t_mem,
        "t_step": t_step, "tok_s": B / t_step,
        "regime": "compute" if t_compute > t_mem else "memory",
        "w_bytes": w_bytes, "kv_bytes": kvb, "flops": flops,
    }


def ridge_batch(m: Model, hw: HW, L: int, w_dtype: str = "bf16") -> float:
    """Batch where t_compute == t_mem (the roofline knee) — solved numerically."""
    lo, hi = 1.0, 4096.0
    for _ in range(40):
        mid = (lo + hi) / 2
        s = step(m, hw, max(1, round(mid)), L, w_dtype)
        if s["t_compute"] < s["t_mem"]:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def max_concurrent(m: Model, hw: HW, L: int, w_dtype: str = "bf16", kv_dtype: str = "bf16",
                   act_gb: float = 4.0) -> int:
    """Max sequences at context L before HBM is exhausted (weights + activations + KV)."""
    budget = hw.mem - dense_params(m)["total"] * DTYPE_BYTES[w_dtype] - act_gb * 1e9
    per_seq_kv = kv_bytes(m, 1, L, kv_dtype)
    return max(0, int(budget / per_seq_kv))


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def fmt_ms(s: float) -> str:
    return f"{s * 1e3:6.2f}ms"


def section_ceiling(m: Model, hw: HW):
    print(f"\n{'='*74}\n  CEILING SWEEP — {m.name} on {hw.name}, bf16\n{'='*74}")
    dp = dense_params(m)
    ple = ple_table_params(m)
    print(f"  dense params (streamed/step): {dp['total']/1e9:.2f}B "
          f"({dp['total']*2/1e9:.1f}GB bf16)  [attn {dp['attn']/1e9:.2f}B · "
          f"mlp {dp['mlp']/1e9:.2f}B · lm_head {dp['lm_head']/1e9:.2f}B]")
    print(f"  PLE table (gathered, NOT streamed): {ple/1e9:.2f}B  <- the 'effective-4B vs total' gap\n")
    Ls = [128, 512, 2048]
    header = "  B".ljust(6) + "".join(f"L={L:<5} regime  ".ljust(20) for L in Ls)
    print(header)
    for B in [1, 4, 8, 16, 32, 64, 128, 256]:
        row = f"  {B:<4}"
        for L in Ls:
            s = step(m, hw, B, L)
            row += f"{s['tok_s']:7.0f} tok/s {s['regime'][:3]}  ".ljust(20)
        print(row)
    print("\n  ridge batch B* (knee, memory→compute):",
          "  ".join(f"L={L}: {ridge_batch(m, hw, L):.0f}" for L in Ls))


def section_breakdown(m: Model, hw: HW, B: int, L: int):
    print(f"\n{'='*74}\n  PER-STEP BREAKDOWN — {m.name}/{hw.name}, B={B} L={L}, bf16\n{'='*74}")
    s = step(m, hw, B, L)
    ideal = s["t_step"]
    print(f"  weight-read   {fmt_ms(s['t_weight'])}   (dense weights, once/step — the floor)")
    print(f"  KV-read       {fmt_ms(s['t_kv'])}   ({s['kv_bytes']/1e6:.0f}MB for {B} seqs)")
    print(f"  compute       {fmt_ms(s['t_compute'])}   ({s['flops']/1e9:.0f} GFLOP)  [{s['regime']}-bound]")
    print(f"  {'-'*46}")
    print(f"  IDEAL step    {fmt_ms(ideal)}   -> ceiling {s['tok_s']:.0f} tok/s")

    meas = MEASURED_ISOLATED.get(B) or MEASURED_CLOSED.get(B)
    if meas:
        t_meas = B / meas
        overhead = t_meas - ideal
        eff = ideal / t_meas
        print(f"\n  MEASURED      {fmt_ms(t_meas)}   -> {meas} tok/s  ({eff*100:.0f}% of roof)")
        print(f"  OVERHEAD      {fmt_ms(overhead)}   ({overhead/t_meas*100:.0f}% of the step) <- the target")
        # Attribute the overhead to profiled causes.
        ov_frac = sum(PROFILE[b] for b in OVERHEAD_BUCKETS)
        print(f"\n  gap attribution (profile fractions of measured step):")
        for b in OVERHEAD_BUCKETS:
            print(f"    {b:28s} {PROFILE[b]*100:4.0f}%  ({fmt_ms(PROFILE[b]*t_meas)})")
        print(f"    {'= addressable overhead':28s} {ov_frac*100:4.0f}%  "
              f"(fusing these into the kernels is the #1 lever)")


def section_levers(m: Model, hw: HW, B: int, L: int):
    print(f"\n{'='*74}\n  LEVER MENU — {m.name}/{hw.name}, B={B} L={L}  (ranked by predicted tok/s)\n{'='*74}")
    # Additive-overhead model: measured_step = ideal_step + fixed_overhead. A lever that
    # shrinks the ideal (quant) or the overhead (fusion) shifts its own term only.
    base = step(m, hw, B, L)
    meas = MEASURED_ISOLATED.get(B) or MEASURED_CLOSED.get(B) or base["tok_s"]
    t_meas = B / meas
    overhead = t_meas - base["t_step"]                          # fixed tax (s)
    addressable = t_meas * sum(PROFILE[b] for b in OVERHEAD_BUCKETS)  # fusable part of it

    def predict(new_ideal_s: float, new_overhead_s: float, tokens_mult: float = 1.0) -> float:
        return tokens_mult * B / (new_ideal_s + new_overhead_s)

    levers = []
    # Kernel fusion: remove the fusable data-movement/launch overhead.
    levers.append(("kernel fusion (kill KV-move+launches)",
                   predict(base["t_step"], overhead - addressable),
                   "fold scatter+DtoD+launch tail into the kernels"))
    # int8/fp8 weights: shrinks the ideal weight-read term (A100: bytes only).
    levers.append(("int8/fp8 weights (bytes)",
                   predict(step(m, hw, B, L, w_dtype="int8")["t_step"], overhead),
                   "halve weight bytes; A100 has no fp8 TC so compute roof unchanged"))
    # fp8 KV: shrinks the ideal KV-read term.
    levers.append(("fp8 KV cache",
                   predict(step(m, hw, B, L, kv_dtype="fp8")["t_step"], overhead),
                   "halve KV bytes moved"))
    # Speculative decode: same step, ~(1+a*k) tokens out of it while memory-bound.
    a, k = 0.7, 3
    levers.append((f"speculative decode (k={k}, accept={a})",
                   predict(base["t_step"], overhead, tokens_mult=1 + a * k),
                   "produce multiple tokens per weight-read while memory-bound"))
    # Scale-out: run at target batch. Overhead held constant per step (optimistic — directional).
    for Bt in (128, 256):
        s_t = step(m, hw, Bt, L)
        levers.append((f"scale to B={Bt} (bucketed graph)",
                       Bt / (s_t["t_step"] + overhead),
                       "more sequences share each weight-read (toward B*)"))
    # Combined: fusion + scale + int8 weights (the realistic stack).
    s256 = step(m, hw, 256, L, w_dtype="int8")
    levers.append(("STACK: fusion + int8 wt + B=256",
                   256 / (s256["t_step"] + (overhead - addressable)),
                   "the compounded ceiling if we do all three"))

    print(f"  baseline (measured): {meas:.0f} tok/s   ideal {base['tok_s']:.0f} tok/s "
          f"({base['t_step']/t_meas*100:.0f}% of step)   overhead {overhead*1e3:.1f}ms/step\n")
    for name, tok, why in sorted(levers, key=lambda x: -x[1]):
        print(f"  {name:36s} {tok:7.0f} tok/s  {tok/meas:4.1f}x   {why}")
    print("\n  Additive-overhead model (overhead held fixed per step) — DIRECTIONAL. Batch-scaling")
    print("  numbers are optimistic (overhead grows with B). Confirm the top pick on real A100.")


def section_capacity(m: Model, hw: HW):
    print(f"\n{'='*74}\n  KV CAPACITY — {m.name}/{hw.name}  (max concurrent seqs before OOM)\n{'='*74}")
    for L in [128, 512, 2048, 8192]:
        n_bf16 = max_concurrent(m, hw, L, kv_dtype="bf16")
        n_fp8 = max_concurrent(m, hw, L, kv_dtype="fp8")
        print(f"  L={L:<5}  bf16 KV: {n_bf16:5d} seqs   fp8 KV: {n_fp8:5d} seqs   "
              f"({kv_bytes(m,1,L,'bf16')/1e6:.0f}MB/seq bf16)")
    print("  -> compare to the ~100-256 target: capacity-bound or not?")


def section_cross_hw(m: Model):
    print(f"\n{'='*74}\n  CROSS-HARDWARE — {m.name}, bf16, B=32 L=512  (why the bottleneck moves)\n{'='*74}")
    for hw in HARDWARE.values():
        s = step(m, hw, 32, 512)
        print(f"  {hw.name:12s} ceiling {s['tok_s']:6.0f} tok/s  [{s['regime']}-bound]  "
              f"B*={ridge_batch(m, hw, 512):.0f}   (BW {hw.bw/1e12:.1f}TB/s, {hw.flops/1e12:.0f} TFLOP)")


if __name__ == "__main__":
    m, hw = E4B, HARDWARE["A100-80GB"]
    section_ceiling(m, hw)
    section_breakdown(m, hw, B=32, L=160)
    section_levers(m, hw, B=32, L=160)
    section_capacity(m, hw)
    section_cross_hw(m)
    print()


# ---------------------------------------------------------------------------- research loop

def emit_panel(pct_of_roof: float | None = None, ridge_batch: int | None = None,
               config: dict | None = None):
    """Emit the attribution slice of the vitals panel (LOOP.md step 1 input).

    CPU-only and analytical, so this is the cheapest thing the loop can measure — tier 1 in the
    screening ladder, and it runs on every iteration before anything touches a GPU.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from inference_server.research import harness as H

    cfg = {"model": None, "gpu": None, "max_batch_size": None, "prefill_mode": None,
           "compile": None, "prefill_graph": None, "blocks": None, "sliding_blocks": None,
           "context_window": None, "rates": None, "duration": None, "pool_size": None,
           "max_queue_wait_s": None, "prefix_cache_impl": None, "wave_window_mult": None,
           "analytical": True}
    cfg.update(config or {})
    return H.emit(H.panel_from_stats(
        H.build_validity("roofline", cfg, n_samples=1, workload_regime="synthetic",
                         notes="analytical ceiling; no GPU"),
        pct_of_memory_roof=pct_of_roof, ridge_batch=ridge_batch), label="roofline")
