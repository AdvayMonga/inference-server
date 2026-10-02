"""Download the raw inputs of the real-trace corpus into a local cache, pinned and verified.

    uv sync --extra dev --extra corpus
    .venv/bin/python scripts/corpus/fetch_traces.py [--cache ~/.cache/inference-server/traces]

Idempotent: a file already present with the pinned sha256 is not fetched again; a mismatch is
re-downloaded once and then refused. Raw data never enters git — only the built corpus does.
After download, the WildChat parquet shards are reduced to `wildchat/conversations.jsonl`: the
fields build_corpus.py reads (no IPs, locations or headers), in shard order. That step needs
pyarrow (the `corpus` extra); the build itself reads only the jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("TRACE_CACHE",
                                    Path.home() / ".cache" / "inference-server" / "traces"))

BURSTGPT = "https://github.com/HPMLL/BurstGPT/releases/download/v2.0"
AZURE = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024"
WILDCHAT_REV = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
WILDCHAT = f"https://huggingface.co/datasets/allenai/WildChat-1M/resolve/{WILDCHAT_REV}/data"
QWEN3_REV = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
QWEN3 = f"https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/{QWEN3_REV}"
WILDCHAT_SHARDS = ("train-00000-of-00014.parquet", "train-00001-of-00014.parquet")

# (cache-relative path, url, sha256)
SOURCES: list[tuple[str, str, str]] = [
    ("burstgpt/BurstGPT_3.csv", f"{BURSTGPT}/BurstGPT_3.csv",
     "2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"),
    ("azure/AzureLLMInferenceTrace_conv_1week.csv",
     f"{AZURE}/AzureLLMInferenceTrace_conv_1week.csv",
     "a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941"),
    (f"wildchat/{WILDCHAT_SHARDS[0]}", f"{WILDCHAT}/{WILDCHAT_SHARDS[0]}",
     "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c"),
    (f"wildchat/{WILDCHAT_SHARDS[1]}", f"{WILDCHAT}/{WILDCHAT_SHARDS[1]}",
     "f10fa35bb52703baad5e3177d3382d63b855fce9e9d1b9ab1fe2f0e762464f2a"),
    ("tokenizers/Qwen3-30B-A3B/tokenizer.json", f"{QWEN3}/tokenizer.json",
     "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    ("tokenizers/Qwen3-30B-A3B/tokenizer_config.json", f"{QWEN3}/tokenizer_config.json",
     "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
    ("tokenizers/Qwen3-30B-A3B/vocab.json", f"{QWEN3}/vocab.json",
     "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    ("tokenizers/Qwen3-30B-A3B/merges.txt", f"{QWEN3}/merges.txt",
     "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
]

WILDCHAT_JSONL = "wildchat/conversations.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cache: Path, rel: str, url: str, expected: str) -> None:
    dest = cache / rel
    if dest.exists() and sha256(dest) == expected:
        print(f"  ok      {rel}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetch   {rel} <- {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        raise SystemExit(f"{rel}: sha256 {got} != pinned {expected}; upstream changed, refusing")
    tmp.replace(dest)


def extract_wildchat(cache: Path) -> None:
    """Parquet shards -> one jsonl of the fields the build reads, in shard and row order."""
    import pyarrow.parquet as pq

    out = cache / WILDCHAT_JSONL
    if out.exists():
        print(f"  ok      {WILDCHAT_JSONL}")
        return
    tmp = out.with_suffix(".jsonl.part")
    cols = ["conversation_hash", "language", "toxic", "redacted", "turn", "conversation",
            "openai_moderation"]
    n = 0
    with open(tmp, "w") as f:
        for shard in WILDCHAT_SHARDS:
            pf = pq.ParquetFile(cache / "wildchat" / shard)
            for i in range(pf.num_row_groups):
                for r in pf.read_row_group(i, columns=cols).to_pylist():
                    f.write(json.dumps({
                        "hash": r["conversation_hash"], "language": r["language"],
                        "toxic": r["toxic"], "redacted": r["redacted"], "turn": r["turn"],
                        "flagged": any(m["flagged"] for m in r["openai_moderation"] or []),
                        "messages": [{"role": m["role"], "content": m["content"]}
                                     for m in r["conversation"]],
                    }, ensure_ascii=False) + "\n")
                    n += 1
    tmp.replace(out)
    print(f"  extract {WILDCHAT_JSONL} ({n} conversations)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    args = ap.parse_args()
    cache = Path(args.cache).expanduser()
    for rel, url, expected in SOURCES:
        fetch(cache, rel, url, expected)
    extract_wildchat(cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
