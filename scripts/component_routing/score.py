"""Reconstruct each method's available predictions before opening exhaustive labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def snapshot(run, cap):
    raw = np.array(run["raw"])
    if cap is None:
        return raw, np.zeros(len(raw), bool), run["raw_ms"], False
    verified = np.zeros(len(raw), bool)
    if run["raw_ms"] > cap:
        return np.zeros_like(raw), verified, 0.0, True
    values = raw.copy()
    last = run["raw_ms"]
    for step in run["history"]:
        if step["ms"] > cap:
            break
        if step["correction"] is not None:
            values[~verified] += step["correction"]
        values[step["indices"]] = step["values"]
        verified[step["indices"]] = True
        last = step["ms"]
    return values, verified, last, False


def gain_ratio(y, p):
    # Report tie-averaged recovery, including the all-zero timing fallback.
    best = y.max()
    if best <= 1e-7:
        return None
    return float(np.maximum(y[p == p.max()], 0).mean() / best)


def metrics(gold, pred, verified):
    # Convert donor-minus-clean margins to changes in the clean correct-answer margin.
    y, p = -gold, -pred
    pos = gain_ratio(y, p)
    neg = gain_ratio(-y, -p)
    available = [v for v in (pos, neg) if v is not None]
    err = np.abs(y - p)
    return dict(
        mae=float(err.mean()),
        directional=float(np.mean(available)) if available else None,
        increase=pos,
        decrease=neg,
        absolute=gain_ratio(np.abs(y), np.abs(p)),
        signed_bias=float((p - y).mean()),
        unverified_mae=float(err[~verified].mean()) if not verified.all() else None,
    )


def summarize(directory):
    read = lambda path: json.loads(path.read_text())
    results = {}
    cells = {}
    rng = np.random.default_rng(860135)
    for model in ("qwen", "large"):
        caps = read(directory / (model + "-budget.json"))
        for task in ("addition", "subtraction"):
            source = read(directory / (model + "-" + task + "-test.json"))
            rows = source["rows"]
            keys = source["keys"]
            mlp = np.array([v == keys[-1][1] for _, v in keys])
            methods = rows[0]["methods"].keys()
            cell = model + "-" + task
            report = dict(
                prompts=len(rows),
                components=len(keys),
                exact_measurements=len(rows) * len(keys),
                clean_correct=sum(r["clean_correct"] for r in rows),
                donor_correct=sum(r["donor_correct"] for r in rows),
                common_ms=float(np.median([r["shared_ms"] for r in rows])),
                exact_ms=float(np.median([r["oracle_ms"] for r in rows])),
                budgets={},
            )
            for level, cap in (("raw", None), ("tight", caps["tight"]), ("roomy", caps["roomy"])):
                stats = {}
                for method in methods:
                    sample = []
                    for row in rows:
                        run = row["methods"][method]
                        pred, known, elapsed, miss = snapshot(run, cap)
                        y = np.array(row["gold"])
                        vals = metrics(y, pred, known)
                        for kind, mask in (("heads", ~mlp), ("mlps", mlp)):
                            vals[kind + "_mae"] = float(np.abs(y[mask] - pred[mask]).mean())
                            vals[kind + "_directional"] = metrics(y[mask], pred[mask], known[mask])[
                                "directional"
                            ]
                        vals.update(measured=float(known.sum()), last_ms=elapsed, miss=float(miss))
                        sample.append(vals)
                    arr = {
                        k: np.array([np.nan if s[k] is None else s[k] for s in sample]) for k in sample[0]
                    }
                    cells[cell, level, method] = arr
                    stats[method] = {
                        k: float(np.nanmean(a)) if np.isfinite(a).any() else None for k, a in arr.items()
                    }
                report["budgets"][level] = stats
            results[cell] = report
    # Prespecified primary: unseen checkpoint AND task, tight allowance, both outcomes.
    # The same paired prompt draws are used for all competitors and both metrics.
    main = "large-subtraction"
    draws = rng.integers(0, results[main]["prompts"], (10000, results[main]["prompts"]))
    comparisons = {}
    bounds = []
    for method in results[main]["budgets"]["tight"]:
        if method == "ils":
            continue
        comparisons[method] = {}
        for metric, sign in (("mae", -1), ("directional", 1)):
            v = sign * (cells[main, "tight", "ils"][metric] - cells[main, "tight", method][metric])
            assert np.isfinite(v).all()
            mean = float(v.mean())
            boot = v[draws].mean(1)
            se = float(boot.std(ddof=1))
            entry = dict(advantage=mean)
            comparisons[method][metric] = entry
            bounds.append((mean, boot, se, entry))
    maxdev = np.max([abs(b - m) / s for m, b, s, _ in bounds if s > 1e-12], axis=0)
    crit = float(np.quantile(maxdev, 0.95))
    for m, b, s, e in bounds:
        e["simultaneous_ci95"] = [m - crit * s, m + crit * s]
    return dict(
        cells=results,
        joint_holdout_comparisons=comparisons,
        passes_both_against_all=all(e["simultaneous_ci95"][0] > 0 for _, _, _, e in bounds),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.results), indent=2, allow_nan=False) + "\n")
