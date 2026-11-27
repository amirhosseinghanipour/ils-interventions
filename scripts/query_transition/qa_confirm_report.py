"""Score confirmatory scheduler rows: denominators, behavior, cost, noise and the decision gate.

python qa_confirm_report.py PROTOCOL OUT.json LABEL=ROWS.json [LABEL=ROWS.json ...]
"""
import json
from math import comb
from pathlib import Path
import sys
import unicodedata

import numpy as np

ARMS = ("clean", "text", "full_prompt", "single_disagreement", "full_rolling", "scanned", "online",
        "self_prompt", "self_rolling")
COST_ARMS = ("full_prompt", "full_rolling", "scanned", "online", "clean", "text")
VARIANTS = ("early", "late")
B = 10000


def normalized(text):
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.casefold())
    return " ".join(x for x in text.split() if x not in ("a", "an", "the"))


def match(text, gold):
    return normalized(text) in {normalized(a) for a in gold}


def wilson(k, n):
    if n == 0:
        return dict(successes=k, n=n, rate=None, wilson95=None)
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(successes=int(k), n=int(n), rate=p, wilson95=[max(0., centre - half), min(1., centre + half)])


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def outcomes(row):
    """Per variant, per arm: (switch, keep, joint) plus clean correctness of the article."""
    gold_rel, gold_irr = row["gold"]
    cleans = [[v["queries"][q]["arms"]["clean"]["text"] for q in range(2)] for v in row["variants"].values()]
    assert all(c == cleans[0] for c in cleans), "clean output must not depend on the edit variant"
    clean_ok = match(cleans[0][0], gold_rel) and match(cleans[0][1], gold_irr)
    result = {}
    for variant, v in row["variants"].items():
        result[variant] = {}
        for arm in ARMS:
            rel = v["queries"][0]["arms"][arm]["text"]
            irr = v["queries"][1]["arms"][arm]["text"]
            switch, keep = match(rel, [v["replacement"]]), match(irr, gold_irr)
            result[variant][arm] = (switch, keep, switch and keep)
    return clean_ok, result


def bootstrap_ratio(num, den, rng):
    idx = rng.integers(len(num), size=(B, len(num)))
    ratios = num[idx].sum(1) / den[idx].sum(1)
    return dict(ratio=float(num.sum() / den.sum()), article_bootstrap95=np.quantile(ratios, [.025, .975]).tolist())


def bootstrap_difference(a, b, rng):
    """a, b: arrays [articles, variants] of 0/1; difference of mean success over variants."""
    idx = rng.integers(len(a), size=(B, len(a)))
    diff = (a[idx].sum((1, 2)) - b[idx].sum((1, 2))) / (a.shape[1] * len(a))
    delta = (a.astype(int) - b.astype(int)).ravel()
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    return dict(difference=float((a.sum() - b.sum()) / a.size), article_bootstrap95=np.quantile(diff, [.025, .975]).tolist(),
                wins=wins, losses=losses, exact_two_sided_p=sign_test(wins, losses), variants=int(a.size))


def score(rows, protocol, seed):
    rng = np.random.default_rng(seed)
    gate = protocol["decision_gate"]
    scored = [(row,) + outcomes(row) for row in rows]
    eligible = [(row, res) for row, ok, res in scored if ok]
    out = dict(denominators=dict(
        sampled_articles=len(rows),
        sampled_by_kind={k: sum(r["kind"] == k for r in rows) for k in ("numeric", "name")},
        clean_correct_articles=len(eligible),
        clean_correct_by_kind={k: sum(r["kind"] == k for r, _ in eligible) for k in ("numeric", "name")},
        clean_correct_variants=2 * len(eligible),
        text_positive_variants=sum(res[v]["text"][2] for _, res in eligible for v in VARIANTS),
        clean_relevant_correct=sum(match(r["variants"]["early"]["queries"][0]["arms"]["clean"]["text"], r["gold"][0]) for r in rows),
        clean_irrelevant_correct=sum(match(r["variants"]["early"]["queries"][1]["arms"]["clean"]["text"], r["gold"][1]) for r in rows)))
    # Behavior among clean-correct articles.
    behavior = {}
    for arm in ARMS:
        cells = {}
        for name, keep_row, variants in (
                [("all", lambda r: True, VARIANTS)] +
                [(f"{k}_{v}", (lambda k: lambda r: r["kind"] == k)(k), (v,)) for k in ("numeric", "name") for v in VARIANTS] +
                [(v, lambda r: True, (v,)) for v in VARIANTS]):
            sel = [(r, res) for r, res in eligible if keep_row(r)]
            vals = [res[v][arm] for _, res in sel for v in variants]
            cells[name] = dict(joint=wilson(sum(x[2] for x in vals), len(vals)),
                               switch=wilson(sum(x[0] for x in vals), len(vals)),
                               preservation=wilson(sum(x[1] for x in vals), len(vals)))
        tp = [res[v][arm][2] for _, res in eligible for v in VARIANTS if res[v]["text"][2]]
        cells["text_positive"] = wilson(sum(tp), len(tp))
        behavior[arm] = cells
    out["behavior_clean_correct"] = behavior
    joint = {arm: np.array([[res[v][arm][2] for v in VARIANTS] for _, res in eligible], bool).reshape(-1, 2)
             for arm in ARMS}
    comparisons = {}
    for a, b in (("online", "full_rolling"), ("scanned", "full_rolling"), ("online", "scanned"),
                 ("single_disagreement", "full_prompt"), ("full_rolling", "full_prompt"),
                 ("online", "full_prompt")):
        comparisons[f"{a}-minus-{b}"] = bootstrap_difference(joint[a], joint[b], rng) if len(eligible) else None
        if len(eligible):
            for vi, v in enumerate(VARIANTS):
                comparisons[f"{a}-minus-{b}"][v] = bootstrap_difference(
                    joint[a][:, vi:vi + 1], joint[b][:, vi:vi + 1], rng)
    out["paired_joint"] = comparisons
    # Online vs every-step: where they differ.
    inspect = dict(both_success=0, both_fail=0, online_only=0, rolling_only=0, cases=[])
    for r, res in eligible:
        for v in VARIANTS:
            o, f = res[v]["online"][2], res[v]["full_rolling"][2]
            key = "both_success" if o and f else "both_fail" if not (o or f) else "online_only" if o else "rolling_only"
            inspect[key] += 1
            q = r["variants"][v]["queries"]
            texts = {arm: [q[i]["arms"][arm]["text"] for i in range(2)] for arm in ("online", "full_rolling", "text")}
            if key in ("online_only", "rolling_only") or texts["online"] != texts["full_rolling"]:
                inspect["cases"].append(dict(article=r["article"], kind=r["kind"], variant=v, outcome=key,
                                             replacement=r["variants"][v]["replacement"], gold_irrelevant=r["gold"][1][:2],
                                             texts=texts, online_events=[q[i]["arms"]["online"]["events"] for i in range(2)]))
    out["online_vs_every_step"] = inspect
    # Cost over all sampled queries.
    def per_article(metric, arm, mask=None):
        vals = []
        for r in rows:
            total = 0.0
            for v in r["variants"].values():
                for qi, q in enumerate(v["queries"]):
                    if mask is not None and not mask(q):
                        continue
                    a = q["arms"][arm]
                    total += a["ms"] if metric == "ms" else a["forwards"] if metric == "forwards" else \
                        a["patched"] if metric == "patched" else len(a["tokens"])
            vals.append(total)
        return np.array(vals)
    same = lambda q: q["arms"]["online"]["tokens"] == q["arms"]["full_rolling"]["tokens"]
    cost = dict(totals={m: {arm: float(per_article(m, arm).sum()) for arm in COST_ARMS + ("single_disagreement", "self_prompt", "self_rolling")}
                        for m in ("ms", "forwards", "patched", "tokens")})
    ratios = {}
    for m in ("ms", "forwards", "patched", "tokens"):
        ratios[m] = {}
        for a, b in (("online", "full_rolling"), ("scanned", "full_rolling"), ("online", "scanned")):
            den = per_article(m, b)
            ratios[m][f"{a}/{b}"] = bootstrap_ratio(per_article(m, a), den, rng) if den.sum() else None
    cost["ratios_all_queries"] = ratios
    n_same = sum(same(q) for r in rows for v in r["variants"].values() for q in v["queries"])
    cost["exact_match_queries"] = n_same
    cost["queries"] = 4 * len(rows)
    cost["ratios_exact_match"] = {m: bootstrap_ratio(per_article(m, "online", same), per_article(m, "full_rolling", same), rng)
                                  for m in ("ms", "forwards", "tokens")}
    # Per-query paired log ratio of elapsed time (robustness to a few slow queries).
    logr = np.array([np.log(q["arms"]["online"]["ms"] / q["arms"]["full_rolling"]["ms"])
                     for r in rows for v in r["variants"].values() for q in v["queries"]])
    cost["per_query_log_ratio"] = dict(median_ratio=float(np.exp(np.median(logr))),
                                       geometric_mean_ratio=float(np.exp(logr.mean())))
    stops = {arm: {} for arm in ARMS}
    for r in rows:
        for v in r["variants"].values():
            for q in v["queries"]:
                for arm in ARMS:
                    s = q["arms"][arm]["stop"]
                    stops[arm][s] = stops[arm].get(s, 0) + 1
    cost["stop_reasons"] = stops
    events = dict(trigger_histogram={}, modes={}, scanned_steps={}, scanned_disagreement=0)
    for r in rows:
        for v in r["variants"].values():
            for q in v["queries"]:
                e = q["arms"]["online"]["events"]
                key = "none" if e["trigger"] is None else str(e["trigger"])
                events["trigger_histogram"][key] = events["trigger_histogram"].get(key, 0) + 1
                events["modes"][e["mode"]] = events["modes"].get(e["mode"], 0) + 1
                s = q["arms"]["scanned"]["events"]
                events["scanned_steps"][str(s["step"])] = events["scanned_steps"].get(str(s["step"]), 0) + 1
                events["scanned_disagreement"] += bool(s["disagreement"])
    cost["policy_events"] = events
    out["cost"] = cost
    # Timing noise from repeated measurements.
    reps = [(q["arms"], q["timing_repeats"]) for r in rows for v in r["variants"].values() for q in v["queries"] if q["timing_repeats"]]
    if reps:
        cv = {}
        for arm in ("clean", "text", "full_prompt", "full_rolling", "scanned", "online"):
            m = np.array([[arms[arm]["ms"]] + [x[arm]["ms"] for x in rr] for arms, rr in reps])
            cv[arm] = float(np.median(m.std(1, ddof=1) / m.mean(1)))
        per_rep = []
        k = len(reps[0][1])
        for j in range(k + 1):
            on = sum((arms["online"]["ms"] if j == 0 else rr[j - 1]["online"]["ms"]) for arms, rr in reps)
            fr = sum((arms["full_rolling"]["ms"] if j == 0 else rr[j - 1]["full_rolling"]["ms"]) for arms, rr in reps)
            per_rep.append(on / fr)
        out["timing_noise"] = dict(queries=len(reps), median_cv=cv, online_over_every_step_by_repetition=per_rep,
                                   outputs_identical_across_repeats=all(x[a]["same"] for _, rr in reps for x in rr for a in x))
    out["self_donor_identity"] = dict(
        queries=4 * len(rows),
        self_prompt=sum(q["self_identity"]["self_prompt"] for r in rows for v in r["variants"].values() for q in v["queries"]),
        self_rolling=sum(q["self_identity"]["self_rolling"] for r in rows for v in r["variants"].values() for q in v["queries"]))
    # Decision gate.
    lat = ratios["ms"]["online/full_rolling"]
    lat_m = cost["ratios_exact_match"]["ms"]
    ni = comparisons.get("online-minus-full_rolling")
    tim = comparisons.get("single_disagreement-minus-full_prompt")
    out["gate"] = dict(
        latency_point_ok=lat["ratio"] <= 0.95,
        latency_upper_ok=lat["article_bootstrap95"][1] < 1.0,
        latency_matched_upper_ok=lat_m["article_bootstrap95"][1] < 1.0,
        noninferior=None if ni is None else ni["article_bootstrap95"][0] > -0.10,
        powered=len(eligible) >= 60,
        timing_replicates=None if tim is None else tim["article_bootstrap95"][0] > 0)
    out["gate"]["methods_ok"] = bool(out["gate"]["latency_point_ok"] and out["gate"]["latency_upper_ok"] and
                                     out["gate"]["latency_matched_upper_ok"] and out["gate"]["noninferior"])
    return out


def main():
    protocol = json.loads(Path(sys.argv[1]).read_text())
    seed = int(protocol["inference"]["bootstrap"].split("seed")[1])
    result = {}
    for spec in sys.argv[3:]:
        label, path = spec.split("=", 1)
        result[label] = score(json.loads(Path(path).read_text()), protocol, seed)
    if len(result) > 1:
        result["decision"] = dict(
            methods_claim=all(r["gate"]["methods_ok"] for r in result.values()),
            timing_claim_by_checkpoint={k: r["gate"]["timing_replicates"] for k, r in result.items()})
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1) + "\n")
    for label, r in result.items():
        if label == "decision":
            print("decision", r)
            continue
        print(label, json.dumps(r["denominators"]), json.dumps(r["gate"]))


if __name__ == "__main__":
    main()
