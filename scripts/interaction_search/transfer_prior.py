"""Retrospective zero-label transfer of an IOI-trained normalized correction to midpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .policies import BITS, PAIRS, RIDGE, TRIPLES, calibration_masks
from .score import _bootstrap

SCALE_FLOOR = 0.001
TARGETS = list(TRIPLES)
METHODS = ("midpoint", "transfer_additive", "transfer_pairs", "calibrated_ils")
CONTRASTS = (
    ("transfer_additive", "midpoint"),
    ("transfer_pairs", "midpoint"),
    ("transfer_pairs", "transfer_additive"),
    ("transfer_pairs", "calibrated_ils"),
)
LEFT, RIGHT = np.asarray(PAIRS).T


def design(gradient, masks):
    g = np.asarray(gradient, dtype=float)
    if g.shape != (8,) or not np.isfinite(g).all():
        raise ValueError("expected eight finite midpoint derivatives")
    # Layers keep their identities. Ties use the panel's original head order.
    order = np.concatenate(
        (np.argsort(-abs(g[:4]), kind="stable"), 4 + np.argsort(-abs(g[4:]), kind="stable"))
    )
    scale = max(float(np.linalg.norm(g)), SCALE_FLOOR)
    normalized = g[order] / scale
    x = BITS[list(masks)][:, order]
    pair = x[:, LEFT] * x[:, RIGHT] - 0.5 * (x[:, LEFT] + x[:, RIGHT])
    linear = x * normalized
    context = np.column_stack((linear, pair * normalized[LEFT], pair * normalized[RIGHT]))
    return BITS[list(masks)] @ g, scale, context


def collect_development(rows):
    """Only the pre-existing nontriple calibration masks can be indexed as training labels."""
    if len(rows) != 18 or len({(r["file"], r["index"]) for r in rows}) != 18:
        raise ValueError("expected 18 distinct development prompts")
    samples = []
    for n, r in enumerate(rows):
        if r["file"] != "ioi-test.json":
            raise ValueError("only IOI development rows may train the prior")
        masks = calibration_masks(781023 + n)
        if len(masks) != 32 or len(set(masks)) != 32 or any(m.bit_count() == 3 or m == 0 for m in masks):
            raise ValueError("invalid development calibration masks")
        samples.append((r["midpoint"], masks, [r["gold"][m] for m in masks]))
    return samples


def fit(samples):
    # Every prompt provides the same number of observations. Fit in normalized logit units.
    features, outcomes = [], []
    for gradient, masks, observed in samples:
        base, scale, x = design(gradient, masks)
        y = np.asarray(observed, dtype=float)
        if y.shape != base.shape or not np.isfinite(y).all():
            raise ValueError("invalid measured development values")
        features.append(x)
        outcomes.append((y - base) / scale)
    x, y = np.concatenate(features), np.concatenate(outcomes)
    models = {}
    for name, count in (("transfer_additive", 8), ("transfer_pairs", 64)):
        cols = x[:, :count]
        lengths = np.linalg.norm(cols, axis=0)
        lengths = np.where(lengths > 0, lengths, 1.0)
        z = cols / lengths
        w = np.linalg.solve(z.T @ z + RIDGE * np.eye(count), z.T @ y) / lengths
        models[name] = w.tolist()
    return models


def predict(models, gradient):
    """Generate all three zero-new-label predictions exclusively from the prompt gradient."""
    base, scale, features = design(gradient, range(256))
    out = {"midpoint": base}
    for name, coefs in models.items():
        out[name] = base + scale * (features[:, : len(coefs)] @ np.asarray(coefs))
    return out


def evaluate(models, rows):
    per_prompt = []
    for r in rows:
        predictions = predict(models, r["midpoint"])
        predictions["calibrated_ils"] = np.asarray(r["methods"]["ils"]["scores"])
        y = np.asarray(r["gold"])[TARGETS]
        metrics = {}
        for m, p in predictions.items():
            v = p[TARGETS]
            choice = int(np.argmax(v))
            metrics[m] = dict(
                raw_mask=TARGETS[choice], mae=float(np.mean(abs(v - y))), regret=float(y.max() - y[choice])
            )
        per_prompt.append(dict(index=r["index"], family=r["family"], metrics=metrics))
    losses = ("mae", "regret")
    summary = {
        m: {k: float(np.mean([r["metrics"][m][k] for r in per_prompt])) for k in losses} for m in METHODS
    }
    comparisons = [(a, b, k) for a, b in CONTRASTS for k in losses]
    diff = np.asarray(
        [[r["metrics"][b][k] - r["metrics"][a][k] for a, b, k in comparisons] for r in per_prompt]
    )
    means, ci = _bootstrap(diff, [r["family"] for r in rows], 605491, 10000)
    contrasts = {}
    for j, (a, b, k) in enumerate(comparisons):
        contrasts.setdefault(a + "_vs_" + b, {})[k] = dict(
            loss_reduction=float(means[j]), simultaneous_ci95=ci[j].tolist() if ci is not None else None
        )
    return dict(
        prompts=len(rows),
        families=len({r["family"] for r in rows}),
        summary=summary,
        contrasts=contrasts,
        rows=per_prompt,
    )


def read(path):
    return json.loads(path.read_text())


def summarize(directory):
    src = directory / "development.json"
    dev = read(src)["rows"]
    train_ids = {(r["file"], r["index"]) for r in dev}
    samples = collect_development(dev)
    models = fit(samples)
    result = dict(
        development_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        development_prompts=len(dev),
        development_labels=sum(len(y) for _, _, y in samples),
        heldout_labels_per_transferred_prediction=0,
        local_ils_comparator_labels=32,
        ridge=RIDGE,
        normalization_floor=SCALE_FLOOR,
        models=models,
        cells={},
    )
    for task in ("ioi", "colors"):
        keys = None
        for model in ("qwen", "large"):
            ref = f"{model}-{task}"
            rows = read(directory / f"primary-{ref}.json")["rows"]
            ids = [(r["file"], r["index"]) for r in rows]
            if (
                len(rows) != 48
                or len(set(ids)) != 48
                or train_ids.intersection(ids)
                or (keys is not None and keys != ids)
            ):
                raise ValueError("invalid or overlapping holdout sample")
            keys = ids
            result["cells"][ref] = evaluate(models, rows)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.write_text(
        json.dumps(summarize(args.directory), separators=(",", ":"), allow_nan=False) + "\n"
    )
