"""Grep blind cards (or any ranker prompt) for result leakage; exit 1 on any hit."""
import json
import re
import sys

PATTERNS = {
    "number+unit": r"\d+(\.\d+)?\s*(x|×|%|ms)\b",
    "verdict": r"\b(confirm\w*|reject\w*|noise|invalid\w*|significan\w*|retract\w*|verdict|win|wins|won|lost|loss)\b",
    "hindsight": r"\b(turned out|only|worth nothing|regress\w*|actually|honest\w*|inert|no better|wash|in fact|proved|proven|ended up|failed to|did not|didn't|was not|turns out|surprising\w*|unfortunately|in the end|nothing)\b",
    "identifiers": r"\b(exp-|hyp-|kb-|run-|grp-)|\b[0-9a-f]{7,40}\b|20\d\d-\d\d-\d\d|\b(perf|feat|fix|obs)/[\w-]+",
}


def hits(text: str) -> list[tuple[str, str]]:
    return [(name, m.group(0)) for name, pat in PATTERNS.items()
            for m in re.finditer(pat, text, re.IGNORECASE)]


def main(path: str) -> int:
    raw = open(path).read()
    units = ({c["id"]: json.dumps(c) for c in json.loads(raw)} if path.endswith(".json")
             else {path: raw})
    bad = {k: h for k, t in units.items() if (h := hits(t))}
    for k, h in bad.items():
        print(k, h)
    print(f"{len(units)} unit(s) checked, {len(bad)} with hits")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
