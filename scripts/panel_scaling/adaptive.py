"""Fresh timed adaptive calibration with disjoint paired validation masks."""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from .fit import ALL, fit
from .run import TASKS, TOL, WEIGHT_HASHES, clock, masks, read, save, sha

STAGES = (0, 16, 32, 64, 128, 224)
METHODS = ("ils", "dense", "exact")
DRAWS = 20000


def validate_masks(pool, targets):
    if (
        len(pool) != 256
        or len(targets) != 512
        or len(set(pool) | set(targets)) != 768
        or not all(a ^ b == ALL for a, b in zip(pool[::2], pool[1::2], strict=True))
    ):
        raise ValueError("expected 128 complementary calibration pairs and 512 disjoint queries")


def pair_score(predictions, values):
    diff = np.abs(np.asarray(predictions) - np.asarray(values))
    if diff.shape != (32,) or not np.all(np.isfinite(diff)):
        raise ValueError("need 32 finite validation residuals")
    means = diff.reshape(16, 2).mean(axis=1)
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(16))


def policy(method, gradient, measure, pool, targets):
    """Fit on paid nested prefixes, select on 16 independent validation pair units."""
    if method not in ("ils", "dense"):
        raise ValueError("unknown adaptive method")
    validate_masks(pool, targets)
    check = pool[-32:]
    g = gradient()
    val = measure(check)
    train_y, trace = [], []
    for b in STAGES:
        if b:
            train_y.extend(measure(pool[len(train_y) : b]))
        # Compute both sets in the same fit. The 512 query labels remain unseen.
        prediction = fit(method if b else "midpoint", g, pool[:b], train_y, check + targets)
        mae, se = pair_score(prediction[:32], val)
        trace.append(dict(count=b, mae=mae, se=se))
        if mae + se <= TOL:
            prediction = prediction[32:].tolist()
            break
    else:
        prediction = measure(targets)
        b = None
    return dict(
        accepted=b, gradient=g, training=train_y, validation=val, predictions=prediction, stages=trace
    )


def evaluate(model, tok, row, dev):
    from .data import encode
    from .oracle import Oracle

    train, targets = masks(row)
    ids, labels = encode(tok, row)
    start = clock()
    oracle = Oracle(model, ids, labels, dev["sites"])
    common = clock() - start
    choices = dev["tasks"][row["task"]]["chunks"]

    def measure(query):
        return oracle.exact(query, chunk=choices[str(len(query))]["chunk"])

    seed = int.from_bytes(hashlib.sha256(("adaptive-" + row["id"]).encode()).digest()[:4], "big")
    order = np.random.default_rng(seed).permutation(METHODS).tolist()
    runs = {}
    for method in order:
        start = clock()
        if method == "exact":
            result = dict(predictions=measure(targets))
        else:
            result = policy(method, oracle.gradient, measure, train, targets)
        result["ms"] = clock() - start
        runs[method] = result
    return dict(
        id=row["id"], task=row["task"], template=row["template"], common_ms=common, order=order, runs=runs
    )


def collect(inputs, manifest, directory, model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = inputs / model_name
    devfile = directory / f"development-{model_name}.json"
    primary = directory / f"heldout-{model_name}.json"
    protocol = read(devfile)
    assert sha(path / "model.safetensors") == WEIGHT_HASHES[model_name]
    assert read(primary)["freeze"] == sha(devfile)
    assert sha(manifest) == protocol["signature"]["manifest"]
    for file in ("run.py", "data.py", "fit.py", "oracle.py"):
        assert sha(Path(__file__).with_name(file)) == protocol["signature"][file]
    stamp = dict(primary=sha(primary), supplement=sha(Path(__file__)), manifest=sha(manifest))
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    out = []
    for row in (r for r in read(manifest)["rows"] if r["split"] == "heldout"):
        part = (
            directory
            / f"adaptive-parts-{model_name}"
            / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
        )
        if not part.exists():
            save(part, dict(stamp=stamp, row=evaluate(model, tok, row, protocol)))
        data = read(part)
        assert data["stamp"] == stamp and data["row"]["id"] == row["id"]
        out.append(data["row"])
        if len(out) % 8 == 0:
            print(model_name, len(out), flush=True)
    save(directory / f"adaptive-{model_name}.json", dict(stamp=stamp, rows=out))


def audited(held, record):
    """Replay the stop and verify every measured label against the earlier native pass."""
    assert all(held[k] == record[k] for k in ("id", "task", "template"))
    assert sorted(record["order"]) == sorted(METHODS)
    runs = record["runs"]
    pool, targets = masks(held)
    gold, train = np.asarray(held["gold"]), np.asarray(held["calibration"])
    result = {}
    for method, run in runs.items():
        pred = np.asarray(run["predictions"])
        if method == "exact":
            np.testing.assert_allclose(pred, gold, rtol=3e-4, atol=4e-4)
        else:
            np.testing.assert_allclose(run["gradient"], held["gradient"], rtol=3e-4, atol=4e-4)
            np.testing.assert_allclose(run["validation"], train[-32:], rtol=3e-4, atol=4e-4)
            np.testing.assert_allclose(run["training"], train[: len(run["training"])], rtol=3e-4, atol=4e-4)
            replay = dict(zip(pool[-32:], run["validation"], strict=True))
            replay.update(zip(pool[: len(run["training"])], run["training"], strict=True))

            # Only terminal exact fallback is allowed to consult target labels.
            def measure(query):
                if query == targets:
                    assert run["accepted"] is None
                    return runs["exact"]["predictions"]
                return [replay[m] for m in query]

            again = policy(method, lambda: run["gradient"], measure, pool, targets)
            assert again["accepted"] == run["accepted"] and len(again["training"]) == len(run["training"])
            np.testing.assert_allclose(again["predictions"], pred, rtol=3e-4, atol=4e-4)
            np.testing.assert_allclose(
                [[v["count"], v["mae"], v["se"]] for v in again["stages"]],
                [[v["count"], v["mae"], v["se"]] for v in run["stages"]],
                atol=1e-9,
            )
        assert len(pred) == 512 and np.isfinite(pred).all() and min(record["common_ms"], run["ms"]) > 0
        result[method] = dict(
            mae=float(abs(gold - pred).mean()),
            ms=float(record["common_ms"] + run["ms"]),
            accepted=run.get("accepted"),
            prediction_count=len(pred),
        )
    return result


def summarize_cell(records, seed):
    names = sorted({template for template, r in records})
    groups = [np.array([i for i, (name, _) in enumerate(records) if name == t]) for t in names]
    if not groups or min(map(len, groups)) != max(map(len, groups)):
        raise ValueError("expected balanced template clusters")
    s = [r for t, r in records]
    methods = {}
    for key in METHODS:
        exact = np.array([r["exact"]["ms"] for r in s])
        times = np.array([r[key]["ms"] for r in s])
        errors = np.array([r[key]["mae"] for r in s])
        methods[key] = dict(
            mean_mae=float(errors.mean()),
            mean_ms=float(times.mean()),
            median_paired_speedup_vs_exact=float(np.median(exact / times)),
            prompts_over_tolerance=int(sum(errors > TOL)),
        )
        if key != "exact":
            methods[key]["stage_counts"] = {str(b): sum(r[key]["accepted"] == b for r in s) for b in STAGES}
            methods[key]["fallback_count"] = sum(r[key]["accepted"] is None for r in s)
            accepted = np.array([r[key]["accepted"] is not None for r in s])
            methods[key]["accepted_only_mean_mae"] = (
                float(errors[accepted].mean()) if accepted.any() else None
            )
            methods[key]["accepted_only_mean_ms"] = (
                float(times[accepted].mean()) if accepted.any() else None
            )
    contrasts = [
        "tolerance_minus_ils_mae",
        "dense_minus_ils_ms",
        "exact_minus_ils_ms",
        "dense_minus_ils_mae",
    ]
    data = np.array(
        [
            [
                TOL - r["ils"]["mae"],
                r["dense"]["ms"] - r["ils"]["ms"],
                r["exact"]["ms"] - r["ils"]["ms"],
                r["dense"]["mae"] - r["ils"]["mae"],
            ]
            for r in s
        ]
    )
    means = np.stack([data[idx].mean(0) for idx in groups])
    draws = np.random.default_rng(seed).integers(len(groups), size=(DRAWS, len(groups)))
    dist = means[draws].mean(axis=1)
    paired = {
        name: dict(
            mean=float(data[:, i].mean()), pointwise_ci95=np.quantile(dist[:, i], [0.025, 0.975]).tolist()
        )
        for i, name in enumerate(contrasts)
    }
    for i in range(3):
        paired[contrasts[i]]["two_task_joint_lower95"] = float(np.quantile(dist[:, i], 0.05 / 6))
    positive_fits = sum(methods["ils"]["stage_counts"][str(b)] for b in STAGES[1:])
    passes = positive_fits > 0 and all(paired[c]["two_task_joint_lower95"] > 0 for c in contrasts[:3])
    return dict(prompts=len(records), templates=len(names), methods=methods, paired=paired, passes=passes)


def summarize(directory):
    models = {}
    for model_idx, model in enumerate(WEIGHT_HASHES):
        ref = directory / f"heldout-{model}.json"
        primary = read(ref)["rows"]
        record = read(directory / f"adaptive-{model}.json")
        assert sha(ref) == record["stamp"]["primary"]
        measurements = {t: [] for t in TASKS}
        for a, b in zip(primary, record["rows"], strict=True):
            measurements[a["task"]].append((a["template"], audited(a, b)))
        cells = {
            t: summarize_cell(measurements[t], 55213 + model_idx * 10 + i) for i, t in enumerate(TASKS)
        }
        models[model] = dict(cells=cells, passes=all(c["passes"] for c in cells.values()))
    return dict(
        primary_model="gpt2",
        passes=models["gpt2"]["passes"],
        bootstrap_draws=DRAWS,
        intervals="whole-template percentile; directional bounds Bonferroni at 0.05/6 within model",
        rule="Stop at the first stage with validation MAE plus one SE over 16 complement-pair means <=0.10. This stopping heuristic has no coverage guarantee. Both primary-task cells must have a nonzero accepted ILS fit and all six one-sided lower bounds above zero.",
        models=models,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("collect")
    for arg in ("inputs", "manifest", "directory"):
        g.add_argument(arg, type=Path)
    g.add_argument("model", choices=WEIGHT_HASHES)
    s = sub.add_parser("score")
    s.add_argument("directory", type=Path)
    s.add_argument("out", type=Path)
    a = p.parse_args()
    if a.stage == "collect":
        collect(a.inputs, a.manifest, a.directory, a.model)
    else:
        save(a.out, summarize(a.directory))
