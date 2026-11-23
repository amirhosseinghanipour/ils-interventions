"""Descriptive equal-calibration comparison and a development-selected tighter tolerance."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .score import BUDGETS, TASKS, average_grid, check_manifest, grid_replay, read, require, sha


MATCHED = (32, 128, 256)
FITTED = ("ils", "dense", "linear")


def paired_interval(contrasts, templates):
    """Pointwise bootstrap of a paired mean, enumerating all draws from six equal clusters."""
    names = sorted(set(templates))
    require(len(names) == 6 and set(Counter(templates).values()) == {4}, "need six four-prompt templates")
    delta = np.asarray(contrasts, dtype=float)
    require(delta.shape == (24,) and np.isfinite(delta).all(), "expected 24 finite paired differences")
    template = np.asarray(templates)
    means = np.array([delta[template == name].mean() for name in names])
    draws = np.indices((6,) * 6).reshape(6, -1).T
    changes = means[draws].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "pointwise_ci95": np.quantile(changes, [0.025, 0.975]).tolist(),
        "templates_with_lower_sparse_mae": int((means > 0).sum()),
    }


def compare(dev, held, templates):
    development = average_grid(dev)
    holdout = average_grid(held)
    target = development["midpoint"]["0"] / 2
    selected = {}
    for method in FITTED:
        count = next((n for n in BUDGETS if development[method][str(n)] <= target), None)
        selected[method] = {
            "count": count,
            "policy": "exact" if count is None else "midpoint" if count == 0 else method,
            "heldout_mae": 0.0 if count is None else holdout[method][str(count)],
        }
    equal = {}
    for count in MATCHED:
        key = str(count)
        per_prompt = [g["dense"][key] - g["ils"][key] for g in held]
        equal[key] = {
            "mae": {m: holdout[m][key] for m in FITTED},
            "dense_minus_sparse_mae": paired_interval(per_prompt, templates),
        }
    return {
        "development_prompts": len(dev),
        "heldout_prompts": len(held),
        "development_midpoint_mae": development["midpoint"]["0"],
        "heldout_midpoint_mae": holdout["midpoint"]["0"],
        "half_development_midpoint_target": target,
        "selected_for_half_target": selected,
        "equal_budget": equal,
    }


def summarize(directory, manifest_path):
    manifest = read(manifest_path)
    check_manifest(manifest)
    rows = manifest["rows"]
    models = {}
    result = {
        "scope": "Post-primary descriptive sensitivity introduced after observing the GPT-2 development choice at 0.10 logits. This artifact reports prediction error only.",
        "bootstrap": "Paired template percentile intervals from all 46,656 ordered resamples of six held-out templates per cell. Pointwise per fixed budget and cell; intervals condition on the development panel.",
        "reference": "Positive dense-minus-sparse MAE favors sparse ILS. A null calibration count marks exact fallback.",
        "manifest_sha256": sha(manifest_path),
        "models": models,
    }
    for model in ("gpt2", "qwen", "large"):
        dev_path = directory / f"development-{model}.json"
        dev, test = read(dev_path), read(directory / f"heldout-{model}.json")
        require(dev["model"] == model and test["freeze"] == sha(dev_path), "unexpected development freeze")
        require(dev["signature"]["manifest"] == result["manifest_sha256"], "development manifest changed")
        block = {}
        for task in TASKS:
            dev_rows = dev["tasks"][task]["rows"]
            test_rows = [r for r in test["rows"] if r["task"] == task]
            for split, measured in (("development", dev_rows), ("heldout", test_rows)):
                declared = [r for r in rows if r["task"] == task and r["split"] == split]
                require([r["id"] for r in measured] == [r["id"] for r in declared], "prompt mismatch")
                if split == "heldout":
                    require(
                        [r["template"] for r in measured] == [r["template"] for r in declared],
                        "template mismatch",
                    )
            block[task] = compare(
                [grid_replay(r) for r in dev_rows],
                [grid_replay(r) for r in test_rows],
                [r["template"] for r in test_rows],
            )
        models[model] = block
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("directory", "manifest", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(summarize(args.directory, args.manifest), indent=2, allow_nan=False) + "\n"
    )
