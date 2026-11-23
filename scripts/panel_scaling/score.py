"""Replay the 16-head accuracy and time comparison from released native measurements."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.component_routing.sampling import WEIGHTS
from .data import make_rows
from .fit import ALL, design, fit

CHECKPOINTS = {**WEIGHTS, "gpt2": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"}
BUDGETS = (0, 8, 16, 24, 32, 48, 64, 96, 128, 256)
METHODS = ("midpoint", "linear", "dense", "ils")
TASKS = ("gift", "route")
TOL = 0.10
CONTRASTS = ("accuracy_margin", "dense_time_advantage_ms", "exact_time_advantage_ms")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def budgets(method):
    return (0,) if method == "midpoint" else BUDGETS


def check_manifest(manifest):
    rows = manifest["rows"]
    require(rows == make_rows(), "manifest differs from seeded prompt pairs")
    require(set(manifest["tokenizers"]) == {"qwen", "smol", "gpt2"}, "incomplete tokenizer coverage")
    for tok in manifest["tokenizers"].values():
        require(tok["rows_validated"] == len(rows), "missing tokenizer validation")
    require(len({r["id"] for r in rows}) == len(rows), "duplicate prompt ID")
    texts = [r[k] for r in rows for k in ("clean", "donor")]
    require(len(set(texts)) == len(texts), "repeated recipient/donor text")
    for task in TASKS:
        subset = [r for r in rows if r["task"] == task]
        pair_keys = ["entity_pair"] + (["gift_pair"] if task == "gift" else [])
        for field in pair_keys:
            pairs = {tuple(sorted(r[field])) for r in subset}
            require(len(pairs) == len(subset), f"{task} reused {field}")
        groups = [{r["template"] for r in subset if r["split"] == s} for s in ("development", "heldout")]
        require([len(g) for g in groups] == [2, 6] and groups[0].isdisjoint(groups[1]), "template leakage")
        require(set(Counter(r["template"] for r in subset).values()) == {4}, "unbalanced templates")
    return rows


def grid_replay(row):
    g, y, gold = [np.asarray(row[k], dtype=float) for k in ("gradient", "calibration", "gold")]
    require(g.shape == (16,) and y.shape == (256,) and gold.shape == (512,), "incomplete native table")
    require(all(np.isfinite(v).all() for v in (g, y, gold)), "nonfinite native outcome")
    seed = int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4], "big")
    tr, tar = design(seed, pairs=128)
    require(len(set(tr + tar)) == 768 and all(0 < m < ALL for m in tr + tar), "mask overlap")
    require(all(a ^ b == ALL for a, b in zip(tr[::2], tr[1::2])), "unbalanced calibration design")
    require(set(row["errors"]) == set(METHODS), "method coverage changed")
    grid = {}
    for m in METHODS:
        require(set(row["errors"][m]) == {str(b) for b in budgets(m)}, "calibration grid changed")
        grid[m] = {}
        for b in budgets(m):
            forecast = fit(m if b else "midpoint", g, tr[:b], y[:b], tar)
            mae = float(np.mean(abs(forecast - gold)))
            np.testing.assert_allclose(row["errors"][m][str(b)], mae, rtol=1e-7, atol=2e-8)
            grid[m][str(b)] = mae
    return grid


def average_grid(grids):
    return {m: {str(b): float(np.mean([g[m][str(b)] for g in grids])) for b in budgets(m)} for m in METHODS}


def select_budgets(grid):
    return {m: next((b for b in budgets(m) if grid[m][str(b)] <= TOL), None) for m in METHODS}


def check_chunks(chunks):
    sizes = (*BUDGETS[1:], 512)
    require(set(chunks) == {str(n) for n in sizes}, "missing exact batch calibration")
    for n in sizes:
        record = chunks[str(n)]
        times = record["times_ms"]
        valid_chunks = {min(c, n) for c in (16, 32, 64, 128)}
        require(bool(times) and {int(c) for c in times} <= valid_chunks, "invalid exact chunk size")
        for reps in times.values():
            require(len(reps) == 2 and np.isfinite(reps).all() and min(reps) > 0, "invalid chunk timing")
        best = min(times, key=lambda c: np.median(times[c]))
        require(record["chunk"] == int(best), "chunk differs from development optimum")


def policy(budget, method):
    if budget is None:
        return "exact"
    return "midpoint" if budget == 0 else method


def point(row, declared, operating, grid):
    require(all(row[k] == declared[k] for k in ("id", "task", "template")), "held-out prompt alignment")
    keys = {m: policy(operating[m], m) for m in METHODS}
    require(row["methods"] == keys, "execution differs from frozen policy")
    timings = row["timings_ms"]
    require(set(timings) == set(keys.values()) | {"exact", "midpoint"}, "missing timed execution")
    common = row["common_ms"]
    require(np.isfinite(common) and common > 0, "invalid shared time")
    times = {}
    for name, reps in timings.items():
        require(len(reps) == 3 and np.isfinite(reps).all() and min(reps) > 0, "invalid repetitions")
        times[name] = float(common + np.median(reps))
    labels = row["labels"]
    require(len(labels) == 2 and len(set(labels)) == 2, "invalid answers")
    require(0 <= row["native_max_error"] <= 0.001, "native audit failed")
    times_by_method = {m: times[key] for m, key in keys.items()}
    errors = {m: 0.0 if key == "exact" else grid[m][str(operating[m])] for m, key in keys.items()}
    errors["exact"] = 0.0
    times_by_method["exact"] = times["exact"]
    return dict(
        template=row["template"],
        errors=errors,
        times=times_by_method,
        clean_correct=row["clean_top"] == labels[0],
        donor_correct=row["donor_top"] == labels[1],
        zero_mae=float(np.mean(np.abs(row["gold"]))),
        raw_midpoint_ms=times["midpoint"],
        common_ms=common,
        native_error=row["native_max_error"],
    )


def method_summary(points, operating, method):
    key = "exact" if method == "exact" else policy(operating[method], method)
    ms = np.mean([r["times"][method] for r in points])
    abs_errors = [r["errors"][method] for r in points]
    return dict(
        policy=key,
        budget=operating.get(method),
        mae=float(np.mean(abs_errors)),
        median_prompt_mae=float(np.median(abs_errors)),
        mean_ms=float(ms),
        speedup_vs_exact=float(np.mean([r["times"]["exact"] for r in points]) / ms),
    )


def cluster_inputs(points):
    groups = sorted({r["template"] for r in points})
    require(len(groups) == 6, "primary requires six template clusters")
    grouped = []
    for group in groups:
        rows = [r for r in points if r["template"] == group]
        require(len(rows) == 4, "expected four prompts per template")
        grouped.append(
            np.mean(
                [
                    [
                        TOL - r["errors"]["ils"],
                        r["times"]["dense"] - r["times"]["ils"],
                        r["times"]["exact"] - r["times"]["ils"],
                    ]
                    for r in rows
                ],
                axis=0,
            )
        )
    return np.array(grouped)


def simultaneous(cluster_means, seed=41683, draws=20000):
    """Joint one-sided max-standardized bootstrap using independent task resamples.

    Within a task all three contrasts share its template draw. Each template has
    four prompts, so sampling template means gives the original prompt weighting.
    """
    rng = np.random.default_rng(seed)
    means = []
    samples = []
    for task in TASKS:
        groups = np.asarray(cluster_means[task])
        require(groups.shape == (6, 3) and np.isfinite(groups).all(), "invalid primary clusters")
        means.append(groups.mean(axis=0))
        indices = rng.integers(0, len(groups), size=(draws, len(groups)))
        samples.append(groups[indices].mean(axis=1))
    center = np.concatenate(means)
    boot = np.column_stack(samples)
    se = boot.std(axis=0, ddof=1)
    varying = se > 1e-12 * np.maximum(1.0, abs(center))
    se[~varying] = 0.0
    crit = 0.0
    if varying.any():
        max_t = ((boot[:, varying] - center[varying]) / se[varying]).max(axis=1)
        crit = float(np.quantile(max_t, 0.95))
    lower = center - crit * se
    by_task = {}
    for offset, task in enumerate(TASKS):
        by_task[task] = {}
        for j, key in enumerate(CONTRASTS, start=3 * offset):
            by_task[task][key] = dict(estimate=float(center[j]), lower95=float(lower[j]), se=float(se[j]))
    return dict(
        tasks=by_task,
        positive_bounds=bool((lower > 0).all()),
        critical=crit,
        draws=draws,
        seed=seed,
        cluster="template",
    )


def audit_panel(development, model):
    attrs = np.asarray(development["mean_absolute_attribution"])
    require(attrs.ndim == 2 and attrs.shape[0] == 2 and attrs.shape[1] >= 8, "invalid head attributions")
    require(np.isfinite(attrs).all() and (attrs >= 0).all(), "invalid attribution magnitude")
    layers = (5, 8) if model == "gpt2" else (11, 17)
    sites = [
        [l, int(h)]
        for l, scores in zip(layers, attrs)
        for h in sorted(np.argsort(-scores, kind="stable")[:8])
    ]
    require(development["model"] == model and development["sites"] == sites, "development panel mismatch")


def report(directory):
    manifest_path = directory / "manifest.json"
    rows = check_manifest(read(manifest_path))
    dev_rows = [r for r in rows if r["split"] == "development"]
    holdout = [r for r in rows if r["split"] == "heldout"]
    signature = {f: sha(Path(__file__).with_name(f)) for f in ("run.py", "oracle.py", "data.py", "fit.py")}
    signature["manifest"] = sha(manifest_path)
    result = dict(
        tolerance=TOL,
        budgets=list(BUDGETS),
        unique_development=len(dev_rows),
        unique_holdout=len(holdout),
        queries_per_prompt=512,
        cells={},
    )
    primary_groups = {}
    for model, weights in CHECKPOINTS.items():
        devfile = directory / f"development-{model}.json"
        development = read(devfile)
        test = read(directory / f"heldout-{model}.json")
        require(test["freeze"] == sha(devfile), "development freeze digest mismatch")
        require(development["signature"] == dict(signature, weights=weights), "collection inputs changed")
        audit_panel(development, model)
        require([r["id"] for r in test["rows"]] == [r["id"] for r in holdout], "held-out sequence mismatch")
        require(set(development["tasks"]) == set(TASKS), "development task mismatch")
        for task in TASKS:
            dev = development["tasks"][task]
            ids = [r["id"] for r in dev_rows if r["task"] == task]
            require([r["id"] for r in dev["rows"]] == ids, "development sequence mismatch")
            dev_grid = average_grid([grid_replay(r) for r in dev["rows"]])
            operating = select_budgets(dev_grid)
            require(operating == dev["operating"], "policy differs from development-only budget rule")
            check_chunks(dev["chunks"])
            grids, points = [], []
            for r, original in zip(test["rows"], holdout, strict=True):
                if original["task"] != task:
                    continue
                grid = grid_replay(r)
                grids.append(grid)
                points.append(point(r, original, operating, grid))
            held_grid = average_grid(grids)
            methods = {m: method_summary(points, operating, m) for m in (*METHODS, "exact")}
            same_b = operating["ils"]
            matched_gain = (
                None if same_b is None else held_grid["linear"][str(same_b)] - held_grid["ils"][str(same_b)]
            )
            result["cells"][f"{model}-{task}"] = dict(
                development_prompts=len(ids),
                prompts=len(points),
                templates=len({r["template"] for r in points}),
                methods=methods,
                development_grid_mae=dev_grid,
                grid_mae=held_grid,
                zero_mae=float(np.mean([r["zero_mae"] for r in points])),
                raw_midpoint=dict(
                    mae=held_grid["midpoint"]["0"],
                    mean_ms=float(np.mean([r["raw_midpoint_ms"] for r in points])),
                ),
                equal_budget_linear_minus_ils_mae=matched_gain,
                clean_accuracy=float(np.mean([r["clean_correct"] for r in points])),
                donor_accuracy=float(np.mean([r["donor_correct"] for r in points])),
                mean_common_ms=float(np.mean([r["common_ms"] for r in points])),
                max_native_error=max(r["native_error"] for r in points),
            )
            if model == "gpt2":
                primary_groups[task] = cluster_inputs(points)
    primary = simultaneous(primary_groups)
    primary["fitted_on_both_tasks"] = all(
        result["cells"][f"gpt2-{t}"]["methods"]["ils"]["policy"] == "ils" for t in TASKS
    )
    primary["passes"] = primary["fitted_on_both_tasks"] and primary["positive_bounds"]
    result["primary"] = primary
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = report(args.directory)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out["primary"], indent=2))
