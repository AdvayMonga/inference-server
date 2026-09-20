"""Did the missing stop ids contaminate the June A100 E4B head-to-head? (kb-20260918-5906bc13)

That entry proved the custom backend stopped only at `<eos>` (1) while vLLM stopped at the
generation config's `[1, 106, 50]`, and proved E2B like-for-like on the June workload. E4B was
unproven: the CSVs hold aggregates and the weights are not cached locally.

This replays the exact June workload on the MODEL alone — plain `transformers`, no engine, no
tokenizer, no chat template: 8 raw token-id prompts `[50000+i] + range(100, 160)`, greedy, 100
steps, nothing stopping early. For each prompt it reports the first index of 1, 106 and 50, so
the step at which each engine would have stopped, and the steps the custom engine wasted.

    venv/bin/modal run scripts/gpu_tests/check_e4b_stop_tokens_modal.py --dry-run
    venv/bin/modal run scripts/gpu_tests/check_e4b_stop_tokens_modal.py
"""

import modal

MODEL = "google/gemma-4-E4B-it"
GPU = "L4"  # 24 GB, the cheapest Modal card that holds E4B in bf16 (~15 GB)
STEPS = 100
PROMPTS = [[50000 + i] + list(range(100, 160)) for i in range(8)]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4", "transformers>=4.51", "accelerate>=0.30", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("e4b-stop-token-check", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=3600)
def check():
    import time
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    t0 = time.time()
    gen_cfg = GenerationConfig.from_pretrained(MODEL)
    eos = gen_cfg.eos_token_id
    stop_ids = [eos] if isinstance(eos, int) else list(eos)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t0

    rows = []
    for prompt in PROMPTS:
        ids = torch.tensor([prompt], device="cuda")
        out, past = [], None
        with torch.no_grad():
            step_in = ids
            for _ in range(STEPS):  # manual greedy decode: nothing stops early
                res = model(input_ids=step_in, past_key_values=past, use_cache=True)
                past = res.past_key_values
                nxt = int(res.logits[0, -1].argmax())
                out.append(nxt)
                step_in = torch.tensor([[nxt]], device="cuda")
        first = {t: (out.index(t) if t in out else None) for t in (1, 106, 50)}
        rows.append({"prompt_head": prompt[0], "tokens": out, "first": first})

    return {"model": MODEL, "gpu": GPU, "steps": STEPS, "gen_config_eos": stop_ids,
            "load_s": load_s, "wall_s": time.time() - t0, "rows": rows}


def _report(r):
    """Print the verdict block."""
    print("\n" + "=" * 78)
    print(f"E4B STOP-TOKEN CHECK — {r['model']} on {r['gpu']}, greedy, {r['steps']} steps")
    print(f"generation_config eos_token_id = {r['gen_config_eos']}")
    print(f"load {r['load_s']:.1f}s, total {r['wall_s']:.1f}s")
    print("=" * 78)
    print(f"{'prompt':>8} {'first 1':>8} {'first 106':>10} {'first 50':>9} "
          f"{'custom':>7} {'vLLM':>6} {'wasted':>7}")
    total = 0
    for row in r["rows"]:
        f = row["first"]
        custom = f[1] + 1 if f[1] is not None else r["steps"]
        ends = [v + 1 for v in f.values() if v is not None]
        vllm = min(ends) if ends else r["steps"]
        wasted = custom - vllm
        total += wasted
        print(f"{row['prompt_head']:>8} {str(f[1]):>8} {str(f[106]):>10} {str(f[50]):>9} "
              f"{custom:>7} {vllm:>6} {wasted:>7}")
    print("-" * 78)
    print(f"total steps the custom backend ran beyond vLLM's stop: {total} "
          f"of {len(r['rows']) * r['steps']}")
    print("VERDICT:", "LIKE-FOR-LIKE — no 106/50 preceded a 1, the missing stop ids cost nothing"
          if total == 0 else f"CONTAMINATED — the custom engine did {total} extra decode steps")
    print("=" * 78)
    for row in r["rows"]:
        print(f"prompt {row['prompt_head']}: {row['tokens']}")


@app.local_entrypoint()
def main(dry_run: bool = False):
    if dry_run:
        print(f"planned: gpu={GPU}  model={MODEL}  steps={STEPS}  prompts={len(PROMPTS)}")
        print("image=debian_slim(3.11)+torch+transformers  volume=hf-cache@/root/.cache/huggingface")
        print(f"secret=huggingface-secret  prompt[0]={PROMPTS[0][:3]}...{PROMPTS[0][-2:]} "
              f"(len {len(PROMPTS[0])})")
        print("dry run OK — no GPU started")
        return
    _report(check.remote())
