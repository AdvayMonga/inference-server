"""Score every available ranker against the answer key (Spearman + permutation p); writes results.json."""
import json
import math
import random
import re
import statistics
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_PERM = 10_000
THRESHOLD = 0.3
NAIVE = {"kernel": 2, "cuda_graph": 2, "launch_overhead": 2, "scheduling": 1}  # else 1


def midranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(midranks(a), midranks(b))


def perm_p(scores, outcome, seed=0):
    """One-sided p: share of permutations with rho >= observed."""
    obs = spearman(scores, outcome)
    if math.isnan(obs):
        return obs, float("nan")
    rng, s = random.Random(seed), list(scores)
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(s)
        ge += spearman(s, outcome) >= obs - 1e-12
    return obs, (ge + 1) / (N_PERM + 1)


def critical_rho(outcome, seed=1):
    """95th percentile of rho under the null for an untied ranker against this outcome."""
    rng, s = random.Random(seed), list(range(len(outcome)))
    dist = []
    for _ in range(N_PERM):
        rng.shuffle(s)
        dist.append(spearman(s, outcome))
    return sorted(dist)[int(0.95 * N_PERM)]


def load_rankers(cards):
    ids = [c["id"] for c in cards]
    rankers = {}
    for model in ("haiku", "sonnet", "opus"):
        files = sorted((HERE / "raw").glob(f"{model}_s*.json"))
        if files:
            recs = [json.loads(f.read_text()) for f in files]
            rankers[model] = {"samples": [r["scores"] for r in recs],
                              "models_used": sorted({m for r in recs for m in r["models_used"]})}
    jev = HERE / "raw" / "jev.json"
    if jev.exists():
        rankers["jev"] = {"samples": [json.loads(jev.read_text())["scores"]]}
    human = parse_human(HERE / "HUMAN_SHEET.md", ids)
    if human:
        rankers["human"] = {"samples": [human]}
    rankers["naive_prior"] = {"samples": [{c["id"]: float(NAIVE.get(c["category"], 1)) for c in cards}]}
    return rankers


def parse_human(path, ids):
    if not path.exists():
        return None
    got = dict(re.findall(r"^### (C\d\d)\s*$.*?^Score:[ \t]*([0-3])", path.read_text(), re.M | re.S))
    return {k: float(v) for k, v in got.items()} if set(got) == set(ids) else None


def analyse(ids, outcome, samples):
    per = [perm_p([s[i] for i in ids], outcome, seed=n) for n, s in enumerate(samples)]
    avg = [statistics.fmean(s[i] for s in samples) for i in ids]
    rho_avg, p_avg = perm_p(avg, outcome, seed=99)
    return {"per_sample": [{"rho": r, "p": p} for r, p in per],
            "mean_rho": statistics.fmean(r for r, _ in per), "rho_avg": rho_avg, "p_avg": p_avg,
            "verdict": "REJECT" if not rho_avg >= THRESHOLD else "not rejected"}


def main():
    cards = json.loads((HERE / "cards.json").read_text())
    key = json.loads((HERE / "answer_key.json").read_text())
    rankers = load_rankers(cards)
    all_ids = [c["id"] for c in cards]
    analyses = {"primary": all_ids,
                "secondary": [i for i in all_ids if key[i]["verdict"] != "invalid"]}
    res = {"n": {}, "critical_rho_one_sided_05": {}, "rankers": {}, "agreement": {}}
    for name, ids in analyses.items():
        outcome = [key[i]["outcome_score"] for i in ids]
        res["n"][name] = len(ids)
        res["critical_rho_one_sided_05"][name] = critical_rho(outcome)
        for r, spec in rankers.items():
            res["rankers"].setdefault(r, {"models_used": spec.get("models_used")})[name] = \
                analyse(ids, outcome, spec["samples"])
    avg = {r: [statistics.fmean(s[i] for s in spec["samples"]) for i in all_ids]
           for r, spec in rankers.items()}
    for a, b in combinations(avg, 2):
        res["agreement"][f"{a}~{b}"] = spearman(avg[a], avg[b])
    for r, spec in rankers.items():
        if len(spec["samples"]) > 1:
            pairs = [spearman([x[i] for i in all_ids], [y[i] for i in all_ids])
                     for x, y in combinations(spec["samples"], 2)]
            res["agreement"][f"{r} self (mean pairwise)"] = statistics.fmean(pairs)
    (HERE / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    report(res)


def report(res):
    print(f"n primary={res['n']['primary']} secondary={res['n']['secondary']}; critical rho "
          f"(one-sided .05) primary={res['critical_rho_one_sided_05']['primary']:.3f} "
          f"secondary={res['critical_rho_one_sided_05']['secondary']:.3f}")
    print("| ranker | per-sample rho (primary) | mean rho | rho(avg) primary | p | "
          "rho(avg) secondary | p | verdict (primary) |")
    print("|---|---|---|---|---|---|---|---|")
    for r, d in res["rankers"].items():
        pr, se = d["primary"], d["secondary"]
        ps = ", ".join(f"{x['rho']:.2f}" for x in pr["per_sample"])
        print(f"| {r} | {ps} | {pr['mean_rho']:.3f} | {pr['rho_avg']:.3f} | {pr['p_avg']:.3f} | "
              f"{se['rho_avg']:.3f} | {se['p_avg']:.3f} | {pr['verdict']} |")
    for k, v in res["agreement"].items():
        print(f"agreement {k}: {v:.3f}")


if __name__ == "__main__":
    main()
