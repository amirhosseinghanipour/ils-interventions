"""Replay nested, complement-balanced calibration budgets for the fixed eight-head search."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policies import ALL, TRIPLES, calibration_masks, fit_observations, propose
from .score import ATOL, RTOL

BUDGETS = (0, 4, 8, 16, 24, 32)
DEPTH = 3
TARGET_GAIN = 0.90


def read(path):
    return json.loads(path.read_text())


def nested_masks(ordered, budget):
    """Choose complements by first appearance; retain collection order within each subset."""
    if budget not in BUDGETS or len(ordered) != 32 or len(set(ordered)) != 32:
        raise ValueError("expected a registered budget and 32 distinct training masks")
    pairs = list(dict.fromkeys(min(m, ALL ^ m) for m in ordered))
    if len(pairs) != 16 or any(m.bit_count() == 3 or (ALL ^ m) not in ordered for m in ordered):
        raise ValueError("training masks must be complete complement pairs disjoint from triples")
    included = set(pairs[: budget // 2])
    return tuple(m for m in ordered if min(m, ALL ^ m) in included)


def predictors(grad, ordered, budget, measure):
    gradients = {"midpoint": grad}
    masks = nested_masks(ordered, budget)
    if not masks:
        base = propose("midpoint", gradients, measure, seed=0)
        return {"linear": base, "ils": base}
    fit = fit_observations(gradients, measure, masks, (0, DEPTH))
    return {"linear": fit[0], "ils": fit[DEPTH]}


def choose_budget(loss):
    """Smallest observed cost retaining 90% of the development full-budget MAE reduction."""
    full_gain = loss[0] - loss[32]
    if full_gain <= 0:
        return 0
    threshold = loss[0] - TARGET_GAIN * full_gain
    return next(b for b in BUDGETS if loss[b] <= threshold)


def run_cell(rows, *, development):
    per_prompt = []
    audit = dict(
        full_score_max_abs_error=None if development else 0.0,
        full_mae_max_abs_error=0.0,
        full_regret_max_abs_error=0.0,
        full_choices_matching=0,
        full_choices_checked=0,
        zero_score_max_abs_error=0.0,
    )
    targets = list(TRIPLES)
    for n, row in enumerate(rows):
        y = np.asarray(row["gold"], dtype=float)
        ordered = calibration_masks((781023 if development else 225177) + n)
        if development:
            # Stored development records provide only exhaustive native effects, not the fit callback.
            paid = dict(zip(ordered, y[list(ordered)], strict=True))
        else:
            batches = [b for b in row["methods"]["ils"]["paid"] if b["role"] == "fit"]
            if len(batches) != 1 or tuple(batches[0]["masks"]) != ordered:
                raise ValueError("stored online calibration differs from the source design")
            paid = dict(zip(ordered, batches[0]["values"], strict=True))
        result = {}
        for b in BUDGETS:
            planned = nested_masks(ordered, b)

            def measure(ids):
                if tuple(ids) != planned:
                    raise ValueError("received exact query outside this budget's training set")
                return [paid[m] for m in ids]

            fits = predictors(row["midpoint"], ordered, b, measure)
            by_method = {}
            for method, p in fits.items():
                choice = p.ranking[0]
                vals = dict(
                    mae=float(abs(y[targets] - p.scores[targets]).mean()),
                    regret=float(y[targets].max() - y[choice]),
                    raw_mask=choice,
                )
                by_method[method] = vals
                if b == 0:
                    if development:
                        base = np.array(
                            [
                                sum(g for i, g in enumerate(row["midpoint"]) if mask & (1 << i))
                                for mask in range(256)
                            ]
                        )
                    else:
                        base = np.asarray(row["methods"]["midpoint"]["scores"])
                    err = float(abs(p.scores - base).max())
                    audit["zero_score_max_abs_error"] = max(audit["zero_score_max_abs_error"], err)
                    np.testing.assert_allclose(p.scores, base, rtol=RTOL, atol=ATOL)
                if b == 32:
                    if development:
                        old = row["grid"]["0" if method == "linear" else str(DEPTH)]
                    else:
                        baseline = row["methods"][method]
                        ref = np.asarray(baseline["scores"])
                        err = float(abs(p.scores - ref).max())
                        audit["full_score_max_abs_error"] = max(audit["full_score_max_abs_error"], err)
                        np.testing.assert_allclose(p.scores, ref, rtol=RTOL, atol=ATOL)
                        old_choice = baseline["raw_mask"]
                        old = dict(
                            mae=float(abs(y[targets] - ref[targets]).mean()),
                            raw_mask=old_choice,
                            regret=float(y[targets].max() - y[old_choice]),
                        )
                    for metric in ("mae", "regret"):
                        key = f"full_{metric}_max_abs_error"
                        audit[key] = max(audit[key], abs(vals[metric] - old[metric]))
                        np.testing.assert_allclose(vals[metric], old[metric], rtol=RTOL, atol=ATOL)
                    audit["full_choices_matching"] += int(choice == old["raw_mask"])
                    audit["full_choices_checked"] += 1
            result[b] = by_method
        per_prompt.append(dict(index=row["index"], family=row["family"], budgets=result))
    summary = {
        b: {
            m: {
                metric: float(np.mean([r["budgets"][b][m][metric] for r in per_prompt]))
                for metric in ("mae", "regret")
            }
            for m in ("linear", "ils")
        }
        for b in BUDGETS
    }
    return dict(prompts=len(rows), summary=summary, audit=audit, rows=per_prompt)


def summarize(directory):
    if read(directory / "frozen.json")["max_pairs"] != DEPTH:
        raise ValueError("expected frozen support size three")
    development = run_cell(read(directory / "development.json")["rows"], development=True)
    loss = {b: development["summary"][b]["ils"]["mae"] for b in BUDGETS}
    chosen = choose_budget(loss)
    cells = {}
    for model in ("qwen", "large"):
        for task in ("ioi", "colors"):
            key = f"{model}-{task}"
            cells[key] = run_cell(read(directory / f"primary-{key}.json")["rows"], development=False)
    return dict(
        fixed_max_pairs=DEPTH,
        budgets=list(BUDGETS),
        development_target_fraction=TARGET_GAIN,
        development_full_mae_gain=loss[0] - loss[32],
        chosen_budget=chosen,
        development=development,
        cells=cells,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.write_text(
        json.dumps(summarize(args.directory), separators=(",", ":"), allow_nan=False) + "\n"
    )
