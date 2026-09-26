"""Run isolated LLM rankers over the blind cards via the Claude CLI; raw outputs to raw/."""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from leakage_check import PATTERNS  # noqa: E402

CLAUDE = os.environ.get("CLAUDE_BIN", str(Path.home() / ".local/bin/claude"))
MODELS = ("haiku", "sonnet", "opus")
SAMPLES = 3
PROMPT = (
    "You are an LLM-inference-engine performance engineer. For each proposed change, score 0-3 "
    "how likely it is to produce a confirmed, significant end-to-end improvement on its target "
    "metric in a careful A/B with >=3 runs per arm on a single rented A100: 0 = no "
    "effect/invalid/regression, 3 = large confirmed win. Output JSON {card_id: score, ...} only, "
    "plus a one-line reason each.\n"
    'Format: one JSON object mapping each card id to {"score": <0-3>, "reason": "<one line>"}.\n'
)
FIELDS = ("change", "gap", "mechanism", "metric", "context", "tier")


def render(cards: list[dict]) -> str:
    """Card text as a ranker sees it (no category label)."""
    return "\n".join(f"[{c['id']}]\n" + "\n".join(f"{f}: {c[f]}" for f in FIELDS) + "\n"
                     for c in cards)


def build_prompt(cards: list[dict], seed: int) -> str:
    order = list(cards)
    random.Random(seed).shuffle(order)
    return PROMPT + "\nCARDS\n\n" + render(order)


def assert_isolated(prompt: str) -> None:
    """Prompts must carry no experiment ids, branch names, SHAs or dates."""
    if re.search(PATTERNS["identifiers"], prompt, re.IGNORECASE):
        raise SystemExit("identifier leaked into ranker prompt")


def parse_scores(text: str, ids: list[str]) -> dict[str, float]:
    obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    out = {k: float(v["score"] if isinstance(v, dict) else v) for k, v in obj.items()}
    missing = set(ids) - set(out)
    if missing:
        raise ValueError(f"missing cards {sorted(missing)}")
    return {k: out[k] for k in ids}


def run_one(model: str, sample: int, cards: list[dict]) -> dict:
    seed = 1000 * (MODELS.index(model) + 1) + sample
    prompt = build_prompt(cards, seed)
    assert_isolated(prompt)
    with tempfile.TemporaryDirectory(prefix="ranker-") as cwd:  # empty dir outside the repo
        proc = subprocess.run(
            [CLAUDE, "-p", "--model", model, "--tools", "", "--setting-sources", "",
             "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
             "--output-format", "json"],
            input=prompt, capture_output=True, text=True, cwd=cwd, timeout=900)
    meta = json.loads(proc.stdout)
    rec = {"model_alias": model, "sample": sample, "order_seed": seed,
           "models_used": list(meta.get("modelUsage", {})), "prompt": prompt,
           "result": meta.get("result", ""), "is_error": meta.get("is_error")}
    try:
        rec["scores"] = parse_scores(rec["result"], [c["id"] for c in cards])
    except ValueError as e:  # keep the incomplete reply on disk; the sample is re-drawn
        n = len(list((HERE / "raw").glob(f"{model}_s{sample}.failed*.json"))) + 1
        rec["error"] = str(e)
        (HERE / "raw" / f"{model}_s{sample}.failed{n}.json").write_text(json.dumps(rec, indent=2) + "\n")
        raise
    (HERE / "raw" / f"{model}_s{sample}.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def main() -> None:
    cards = json.loads((HERE / "cards.json").read_text())
    jobs = [(m, s) for m in MODELS for s in range(1, SAMPLES + 1)
            if not (HERE / "raw" / f"{m}_s{s}.json").exists()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for rec in pool.map(lambda j: run_one(*j, cards), jobs):
            print(rec["model_alias"], rec["sample"], rec["models_used"])


if __name__ == "__main__":
    main()
