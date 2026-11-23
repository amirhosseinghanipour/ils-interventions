"""Behavioral outcomes of single-site choices and errors across native head-group sizes."""

import argparse
import json
from pathlib import Path

import numpy as np

from score import snapshot


def read(path):
    return json.loads(path.read_text())


def behavioral(rows, keys, caps, training):
    gold = np.asarray([r["gold"] for r in rows])
    maximum = np.max(abs(gold), axis=1, keepdims=True)
    ties = abs(gold) == maximum
    votes = (ties / ties.sum(axis=1, keepdims=True)).sum(axis=0)
    flips = np.array([np.asarray(r["patch_top"]) == r["labels"][1] for r in rows])
    eligible = np.array([r["clean_correct"] for r in rows])
    feasible = eligible & flips.any(axis=1)
    winner = int(votes.argmax())
    mlp = np.array([j == keys[-1][1] for _, j in keys])
    out = dict(
        clean_correct=int(eligible.sum()),
        clean_and_donor_correct=sum(r["clean_correct"] and r["donor_correct"] for r in rows),
        clean_correct_can_switch_to_donor=int(feasible.sum()),
        clean_correct_can_switch_to_donor_via_head=int((eligible & flips[:, ~mlp].any(axis=1)).sum()),
        clean_correct_can_switch_to_donor_via_mlp=int((eligible & flips[:, mlp].any(axis=1)).sum()),
        any_top_token_change=sum(any(v != r["clean_top"] for v in r["patch_top"]) for r in rows),
        mlp_is_largest_absolute=float(votes[mlp].sum()),
        most_frequent_largest_absolute=dict(
            layer=keys[winner][0], component=keys[winner][1], prompts=float(votes[winner])
        ),
        policies={},
    )
    if training is not None:
        fixed = np.asarray([r["gold"] for r in training])
        relative = fixed / np.max(abs(fixed), axis=1, keepdims=True).clip(1e-12)
        site = int(relative.mean(axis=0).argmax())
        out["fixed_toward_donor_site"] = keys[site]
        out["fixed_toward_donor_switches"] = int(flips[eligible, site].sum())
    raw = {m: np.array([r["methods"][m]["raw"] for r in rows]) for m in ("ils", "midpoint")}
    out["raw_donor_correction"] = {}
    for name, mask in (("heads", ~mlp), ("mlps", mlp)):
        prev = abs(raw["midpoint"][:, mask] - gold[:, mask]).mean(1)
        cur = abs(raw["ils"][:, mask] - gold[:, mask]).mean(1)
        out["raw_donor_correction"][name] = dict(
            mean_absolute_change=float(abs(raw["ils"][:, mask] - raw["midpoint"][:, mask]).mean()),
            midpoint_mae=float(prev.mean()),
            ils_mae=float(cur.mean()),
            prompts_improved=int((cur < prev).sum()),
            prompts_worsened=int((cur > prev).sum()),
        )
    for level, cap in (("raw", None), ("tight", caps["tight"]), ("roomy", caps["roomy"])):
        by_method = {}
        for method in rows[0]["methods"]:
            hits = []
            for r, patch, use in zip(rows, flips, eligible):
                if use:
                    pred = snapshot(r["methods"][method], cap)[0]
                    hits.append(float(patch[pred == pred.max()].mean()))
            by_method[method] = sum(hits)
        out["policies"][level] = by_method
    return out


def grouped(source):
    methods = list(source[0]["predictions"])
    oracle = np.array([r["gold"][1:-1] for r in source])
    errors = {m: abs(np.array([r["predictions"][m][1:-1] for r in source]) - oracle) for m in methods}
    return dict(
        prompts=len(source),
        interior_masks_per_prompt=oracle.shape[1],
        sizes=source[0]["sizes"][1:-1],
        maximum_native_cut_disagreement=max(r["max_native_cut_error"] for r in source),
        native_effect_mean_absolute_by_size=abs(oracle).mean(axis=0).tolist(),
        methods={
            m: dict(mae=float(v.mean()), mae_by_size=v.mean(axis=0).tolist()) for m, v in errors.items()
        },
    ), {m: v.mean(axis=1) for m, v in errors.items()}


def summarize(directory):
    result = dict(behavior={}, groups={})
    rng = np.random.default_rng(652194)
    draws = {t: rng.integers(0, 24, (10000, 24)) for t in ("addition", "subtraction")}
    comparisons = []
    for model in ("qwen", "large"):
        caps = read(directory / (model + "-budget.json"))
        groups = read(directory / ("groups-" + model + ".json"))
        for task in ("addition", "subtraction"):
            cell = model + "-" + task
            records = read(directory / (cell + "-test.json"))
            if model == "qwen":
                train = read(directory / "qwen-addition-validation.json")["rows"]
                origin = "Qwen addition validation"
            elif task == "subtraction":
                train = read(directory / "large-addition-test.json")["rows"]
                origin = "SmolLM2 addition test, exploratory cross-task transfer"
            else:
                train = None
            # For large addition there is no independent fixed-site calibration.
            behavior = behavioral(records["rows"], records["keys"], caps, train)
            if train is not None:
                behavior["fixed_site_source"] = origin
            result["behavior"][cell] = behavior
            report, per_prompt = grouped(groups[task])
            result["groups"][cell] = report
            for method, values in per_prompt.items():
                if method == "smooth":
                    continue
                advantage = values - per_prompt["smooth"]
                b = advantage[draws[task]].mean(axis=1)
                entry = dict(smooth_mae_advantage=float(advantage.mean()))
                report["methods"][method].update(entry)
                comparisons.append(
                    (report["methods"][method], b, float(advantage.mean()), float(b.std(ddof=1)))
                )
    maxima = np.max([abs(b - avg) / se for _, b, avg, se in comparisons if se > 1e-12], axis=0)
    critical = float(np.quantile(maxima, 0.95))
    for entry, _, avg, se in comparisons:
        entry["simultaneous_ci95"] = [avg - critical * se, avg + critical * se]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.results), indent=2, allow_nan=False) + "\n")
