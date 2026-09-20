"""Did the missing stop ids contaminate the June A100 E4B head-to-head? (kb-20260918-5906bc13)

That entry proved the custom backend stopped only at `<eos>` (1) while vLLM stopped at the
generation config's `[1, 106, 50]`, and proved E2B like-for-like on the June workload. E4B was
unproven: the CSVs hold aggregates and the weights are not cached locally.

This replays the exact June workload on the MODEL alone — plain `transformers`, no engine, no
tokenizer, no chat template: 8 raw token-id prompts `[50000+i] + range(100, 160)`, greedy, 100
steps, nothing stopping early. It records every generated id, so the step at which each engine
would have stopped is auditable rather than asserted. Library versions are pinned so a re-run is
the same experiment.

Scope: this is the model's behaviour under HF greedy decode, unbatched, on one card. Our engine's
greedy path is not byte-identical to it (`kb-20260611-030`) and is batch-width dependent
(`kb-20260901-011`), so it is strong evidence about the June run, not proof of it.

    venv/bin/modal run scripts/gpu_tests/check_e4b_stop_tokens_modal.py --dry-run
    venv/bin/modal run scripts/gpu_tests/check_e4b_stop_tokens_modal.py \
        --out knowledge/evidence/e4b-stop-tokens-20260919.json
"""

import json
import pathlib

import modal

MODEL = "google/gemma-4-E4B-it"
GPU = "L4"  # 24 GB, the cheapest Modal card that holds E4B in bf16 (~15 GB)
STEPS = 100
CUSTOM_STOP_IDS = (1,)  # what the custom backend used before commit bb85b10
PROMPTS = [[50000 + i] + list(range(100, 160)) for i in range(8)]

# Pinned: this instrument closes a question permanently, so a re-run must be the same experiment.
PINS = ["torch==2.14.0", "transformers==5.17.0", "accelerate==1.15.0"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*PINS)
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
app = modal.App("e4b-stop-token-check", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[hf_secret], timeout=3600)
def check():
    import time

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, GenerationConfig

    t0 = time.time()
    eos = GenerationConfig.from_pretrained(MODEL).eos_token_id
    gen_config_eos = [eos] if isinstance(eos, int) else list(eos)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t0

    rows = []
    for prompt in PROMPTS:
        out, past = [], None
        with torch.no_grad():
            step_in = torch.tensor([prompt], device="cuda")
            for _ in range(STEPS):  # manual greedy decode: nothing stops early
                res = model(input_ids=step_in, past_key_values=past, use_cache=True)
                past = res.past_key_values
                nxt = int(res.logits[0, -1].argmax())
                out.append(nxt)
                step_in = torch.tensor([[nxt]], device="cuda")
        rows.append({"prompt_lead": prompt[0], "prompt_len": len(prompt), "tokens": out})

    return {
        "model": MODEL, "gpu": GPU, "steps": STEPS, "prompts": PROMPTS,
        "gen_config_eos": gen_config_eos, "custom_backend_stop_ids": list(CUSTOM_STOP_IDS),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "decode": "greedy argmax, unbatched, bfloat16, no stopping criteria",
        "load_s": round(load_s, 1), "wall_s": round(time.time() - t0, 1), "rows": rows,
    }


def summarise(r):
    """Per-prompt stop indices and wasted steps, derived from the raw ids in `r`."""
    steps, out = r["steps"], []
    for row in r["rows"]:
        toks = row["tokens"]
        first = {t: (toks.index(t) if t in toks else None) for t in r["gen_config_eos"]}
        custom = min((first[t] + 1 for t in r["custom_backend_stop_ids"]
                      if first.get(t) is not None), default=steps)
        vllm = min((v + 1 for v in first.values() if v is not None), default=steps)
        out.append({"prompt_lead": row["prompt_lead"], "first": first,
                    "custom_stop": custom, "vllm_stop": vllm, "wasted": custom - vllm})
    return out


def _report(r):
    """Print the verdict block, derived from the raw ids so it cannot drift from them."""
    rows = summarise(r)
    ids = r["gen_config_eos"]
    print("\n" + "=" * 78)
    print(f"E4B STOP-TOKEN CHECK — {r['model']} on {r['gpu']}, {r['decode']}, {r['steps']} steps")
    print(f"generation_config eos_token_id = {ids}   custom backend had "
          f"{r['custom_backend_stop_ids']}")
    print(f"torch {r['versions']['torch']}, transformers {r['versions']['transformers']}; "
          f"load {r['load_s']}s, total {r['wall_s']}s")
    print("=" * 78)
    head = "".join(f"{'first ' + str(t):>10}" for t in ids)
    print(f"{'prompt':>8}{head}{'custom':>8}{'vLLM':>7}{'wasted':>8}")
    total = 0
    for row in rows:
        total += row["wasted"]
        cells = "".join(f"{str(row['first'][t]):>10}" for t in ids)
        print(f"{row['prompt_lead']:>8}{cells}{row['custom_stop']:>8}"
              f"{row['vllm_stop']:>7}{row['wasted']:>8}")
    print("-" * 78)
    print(f"total steps the custom backend ran beyond vLLM's stop: {total} "
          f"of {len(rows) * r['steps']}")
    print("VERDICT:", "LIKE-FOR-LIKE under HF greedy decode — no 106/50 preceded a 1"
          if total == 0 else f"CONTAMINATED — the custom engine did {total} extra decode steps")
    print("=" * 78)
    for row in r["rows"]:
        print(f"prompt {row['prompt_lead']}: distinct ids emitted {sorted(set(row['tokens']))}")


@app.local_entrypoint()
def main(dry_run: bool = False, out: str = ""):
    if dry_run:
        print(f"planned: gpu={GPU}  model={MODEL}  steps={STEPS}  prompts={len(PROMPTS)}")
        print(f"image=debian_slim(3.11)+{' '.join(PINS)}")
        print("volume=hf-cache@/root/.cache/huggingface  secret=huggingface-secret")
        print(f"prompt[0]={PROMPTS[0][:3]}...{PROMPTS[0][-2:]} (len {len(PROMPTS[0])})")
        print("dry run OK — no GPU started")
        return
    r = check.remote()
    _report(r)
    if out:
        p = pathlib.Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=2) + "\n")
        print(f"\nraw result written to {out}")
