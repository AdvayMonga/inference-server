"""One-shot: DECISIONS.md prose -> knowledge/*.json structured records.

Run once. After this DECISIONS.md is a generated view and the JSON is the source of truth.
Preserves every entry verbatim in `summary`; adds the structure the loop needs (status, tags,
triggers) by parsing the conventions the file already used.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference_server.research.kb import KNOWLEDGE_DIR, save_entry, write_index  # noqa: E402
from inference_server.research.schemas import REPO_ROOT, KnowledgeEntry  # noqa: E402

HEADING = re.compile(r"^###\s+\[(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})(?:→[^\]]*)?\]\s+(?P<rest>.*)$")

# The file's own vocabulary -> our status enum.
STATUS_MAP = [
    ("OBSOLETE", "obsolete"),
    ("REJECTED", "rejected"), ("DO NOT BUILD", "rejected"), ("NOT VIABLE", "rejected"),
    ("NON-VIABLE", "rejected"), ("TRIED", "rejected"), ("MOOT", "rejected"),
    ("RESOLVED", "resolved"), ("SHIPPED", "resolved"), ("DONE", "resolved"),
    ("MEASURED", "resolved"), ("FIRED", "resolved"), ("ACCEPTED", "resolved"),
    ("DEFERRED", "deferred"), ("DEFER", "deferred"), ("PENDING", "deferred"),
    ("OPEN", "open"),
]

TAG_RULES = {
    "prefill": ("prefill",), "decode": ("decode",), "cache": ("cache", "prefix"),
    "kv": ("kv", "block", "pool"), "graph": ("cuda graph", "graph", "capture"),
    "compile": ("compile", "inductor", "dynamo"), "kernel": ("kernel", "triton", "attention"),
    "scheduler": ("scheduler", "admission", "wave", "queue", "fairness"),
    "backpressure": ("backpressure", "pressure", "preempt", "shed"),
    "benchmark": ("benchmark", "harness", "bench", "measure"),
    "numerics": ("parity", "numeric", "invarian", "bf16"),
    "memory": ("memory", "oom", "leak"), "modal": ("modal",),
    "quantization": ("quant", "int8", "fp8"),
}


def classify(rest: str) -> tuple[str, str]:
    title, status = rest, "open"
    if "—" in rest:
        head, _, tail = rest.rpartition("—")
        tail_u = tail.strip().upper()
        for needle, mapped in STATUS_MAP:
            if needle in tail_u:
                title, status = head.strip(), mapped
                break
    return title.strip(), status


def tags_for(text: str) -> list[str]:
    low = text.lower()
    return sorted({tag for tag, needles in TAG_RULES.items() if any(n in low for n in needles)})


def triggers_for(body: str) -> list[str]:
    out = []
    for line in body.splitlines():
        low = line.lower().lstrip("*_ -")
        if low.startswith(("trigger", "triggers", "revisit", "**trigger", "**revisit")):
            cleaned = re.sub(r"^[*_\s-]*(triggers?|revisit)[^:]*:\s*", "", line.strip(),
                             flags=re.I)
            if cleaned:
                out.append(cleaned.strip(" *_"))
    return out


def main() -> int:
    src = REPO_ROOT / "DECISIONS.md"
    text = src.read_text()
    lines = text.splitlines()

    starts = [i for i, ln in enumerate(lines) if HEADING.match(ln)]
    if not starts:
        print("no entries found — aborting rather than writing an empty knowledge base")
        return 1

    migrated = 0
    for n, i in enumerate(starts):
        m = HEADING.match(lines[i])
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        # drop trailing section headers that belong to the file, not the entry
        body = re.sub(r"\n##[^\n]*$", "", body).strip()

        title, status = classify(m.group("rest"))
        created = time.mktime(time.strptime(m.group("date"), "%Y-%m-%d"))
        entry = KnowledgeEntry(
            title=title,
            summary=body or title,
            status=status,
            tags=tags_for(title + " " + body),
            triggers=triggers_for(body),
            created_at=created,
            updated_at=created,
        )
        entry.id = f"kb-{m.group('date').replace('-', '')}-{n:03d}"
        save_entry(entry)
        migrated += 1

    print(f"migrated {migrated} entries -> {KNOWLEDGE_DIR}")
    write_index()
    print("regenerated DECISIONS.md from knowledge/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
