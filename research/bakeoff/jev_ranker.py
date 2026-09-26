"""Jev arm: score each blind card with TypeSafe System One; writes raw/jev.json. Needs TYPESAFE_API_KEY."""
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_rankers import render  # noqa: E402

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
QUESTIONS = {"payoff": {
    "type": "score",
    "instructions": ("How likely is this proposed change to produce a confirmed, significant "
                     "end-to-end improvement on its target metric in a careful A/B test?"),
    "criteria": ["No effect, invalid test, or regression", "Small or unconfirmed effect",
                 "Confirmed modest win", "Large confirmed win"],
}}


def ask(key: str, state: str) -> dict:
    body = json.dumps({"state": state, "model": MODEL, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def main() -> None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY is not set: the Jev arm cannot run, and no results are written.")
    cards = json.loads((HERE / "cards.json").read_text())
    responses, scores = {}, {}
    for c in cards:
        r = ask(key, render([c]))
        responses[c["id"]] = r
        scores[c["id"]] = float(r["answers"]["payoff"]["score"])
        print(c["id"], scores[c["id"]], r["answers"]["payoff"].get("confidence"))
    out = {"model": MODEL, "scores": scores, "responses": responses}
    (HERE / "raw" / "jev.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
