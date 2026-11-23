"""Derivative-prior ablations and end-to-end timings for previously unseen eight-head masks."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policies import BITS, INTERACTIONS, RIDGE, TRIPLES, calibration_masks, propose
from .score import ATOL, RTOL, _bootstrap

ESTIMATORS = ("midpoint", "ig5", "linear", "ils", "sparse_zero", "ridge_zero", "ridge_midpoint")
ABLATIONS = ("sparse_zero", "ridge_zero", "ridge_midpoint")
CHUNKS = (8, 32, 64, 128)
EXACT = tuple(f"exact{k}" for k in CHUNKS)
METHODS = ESTIMATORS + EXACT
FULL_FEATURES = np.column_stack((BITS, INTERACTIONS))
SEED = 225177  # Same prompt-indexed calibration design as the primary comparison.


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(path)


def workload(seed):
    train = calibration_masks(seed)
    available = sorted(set(range(1, 256)) - set(train) - {7})
    order = np.random.default_rng(539221 + seed).permutation(available).tolist()
    return train, order


def predict(name, seed, max_pairs, gradient, measure):
    """Access only requested gradients and the 32 calibration effects through callbacks."""
    if name in ("midpoint", "ig5"):
        return BITS @ np.asarray(gradient(name))
    if name not in ESTIMATORS:
        raise ValueError(f"unknown predictor {name}")
    base = np.zeros(8) if name in ("sparse_zero", "ridge_zero") else np.asarray(gradient("midpoint"))
    if name in ("linear", "ils", "sparse_zero"):
        policy = "linear" if name == "linear" else "ils"
        return propose(policy, {"midpoint": base}, measure, seed=seed, max_pairs=max_pairs).scores

    # Same anchored design, per-column scaling and ridge penalty as the sparse fit,
    # here using every one of the 28 pair terms. The exact zero mask anchors the intercept.
    masks = list(calibration_masks(seed))
    y = np.asarray(measure(masks)) - BITS[masks] @ base
    scale = np.linalg.norm(FULL_FEATURES[masks], axis=0)
    x = FULL_FEATURES / np.where(scale > 0, scale, 1.0)
    x_fit = x[masks]
    coef = np.linalg.solve(x_fit.T @ x_fit + RIDGE * np.eye(x_fit.shape[1]), x_fit.T @ y)
    return BITS @ base + x @ coef


def timing_call(o, name, seed, max_pairs, masks):
    fit_values = []

    def measured(ids):
        vals = o.exact(ids)
        fit_values.extend(vals)
        return vals

    start = o._clock()
    if name in EXACT:
        values = o.exact(masks, batch_size=int(name[5:]))
    else:
        values = predict(name, seed, max_pairs, o.gradient, measured)[masks].tolist()
    return o._clock() - start, values, fit_values


def collect(inputs, protocol, directory, model_key):
    import torch
    from .data import data, encode, model
    from .oracle import Oracle

    checkpoint, tok = model(inputs, model_key)
    panel = read(protocol / f"panel-{model_key}.json")["sites"]
    refs = read(protocol / "manifest.json")
    support = read(directory / "frozen.json")["max_pairs"]
    if support != 3:
        raise ValueError("this follow-up requires the frozen three-pair development result")
    rows = []
    for task in ("ioi", "colors"):
        for n, ref in enumerate(refs[task][:2]):
            seed = SEED + n
            train, order = workload(seed)
            bank = data(inputs, ref["file"])
            ids, labels = encode(tok, bank[ref["index"]])
            o = Oracle(checkpoint, ids, labels, panel)
            rec = dict(
                **ref,
                task=task,
                seed=seed,
                common_ms=o.common_ms,
                train=list(train),
                order=order,
                batches={},
            )
            for k in (1, 32, 128, len(order)):
                masks = order[:k]
                stats = {name: {"times_ms": []} for name in METHODS}
                rng = np.random.default_rng(880313 + k + seed)
                for rep in range(4):
                    for name in rng.permutation(METHODS).tolist():
                        row = stats[name]
                        if row.get("unavailable"):
                            continue
                        try:
                            elapsed, values, paid = timing_call(o, name, seed, support, masks)
                        except torch.cuda.OutOfMemoryError:
                            if name != "exact128":
                                raise
                            row.clear()
                            row["unavailable"] = "cuda_out_of_memory"
                            torch.cuda.empty_cache()
                            continue
                        if rep:
                            row["times_ms"].append(elapsed)
                        if "values" in row:
                            np.testing.assert_allclose(values, row["values"], rtol=RTOL, atol=ATOL)
                            np.testing.assert_allclose(paid, row["train_values"], rtol=RTOL, atol=ATOL)
                        else:
                            row.update(values=values, train_values=paid)
                rec["batches"][str(k)] = stats
            print(model_key, task, n, flush=True)
            rows.append(rec)
    save(directory / f"amortized-{model_key}.json", dict(model=model_key, max_pairs=support, rows=rows))


def from_paid(row, seed, max_pairs):
    fit = [b for b in row["methods"]["ils"]["paid"] if b["role"] == "fit"]
    masks = list(calibration_masks(seed))
    if len(fit) != 1 or fit[0]["masks"] != masks:
        raise ValueError("expected the original ordered 32-label training batch")
    if any(m in TRIPLES or m == 0 for m in masks):
        raise ValueError("training includes a target or reference mask")

    def measure(ids):
        if list(ids) != masks:
            raise ValueError("estimator requested an unpaid label")
        return fit[0]["values"]

    base = row["midpoint"]
    result = {m: predict(m, seed, max_pairs, lambda key: base, measure) for m in (*ABLATIONS, "ils")}
    # Collection recomputes each gradient before the exhaustive pass. Check for
    # numerical equivalence without substituting stored evaluation labels into the fits.
    np.testing.assert_allclose(result["ils"], row["methods"]["ils"]["scores"], rtol=RTOL, atol=ATOL)
    return result


def accuracy(rows, max_pairs):
    per_prompt = []
    target = list(TRIPLES)
    for n, row in enumerate(rows):
        preds = from_paid(row, SEED + n, max_pairs)
        gold = np.asarray(row["gold"])[target]
        estimates = {}
        for m, full in preds.items():
            pred = full[target]
            chosen = int(np.argmax(pred))
            estimates[m] = dict(
                mae=float(abs(pred - gold).mean()),
                regret=float(gold.max() - gold[chosen]),
                raw_mask=target[chosen],
            )
        per_prompt.append(dict(index=row["index"], family=row["family"], estimates=estimates))
    methods = {}
    for m in (*ABLATIONS, "ils"):
        methods[m] = {
            k: float(np.mean([r["estimates"][m][k] for r in per_prompt])) for k in ("mae", "regret")
        }
    contrasts = [(m, k) for m in ABLATIONS for k in ("mae", "regret")]
    delta = np.asarray(
        [[r["estimates"][m][k] - r["estimates"]["ils"][k] for m, k in contrasts] for r in per_prompt]
    )
    avg, ci = _bootstrap(delta, [r["family"] for r in rows], 729552, 10000)
    for i, (m, k) in enumerate(contrasts):
        methods[m][k + "_ils_advantage"] = float(avg[i])
        methods[m][k + "_simultaneous_ci95"] = ci[i].tolist() if ci is not None else None
    return dict(methods=methods, rows=per_prompt)


def speed_report(rows, native):
    samples = {}
    for r in rows:
        task = r["task"]
        raw = native[task][r["index"]]
        gold = np.asarray(raw["gold"])
        train, order = workload(r["seed"])
        if list(train) != r["train"] or order != r["order"] or len(set(order)) != 222:
            raise ValueError("invalid or nonunique unseen evaluation mask set")
        if r["family"] != raw["family"] or r.get("file") != raw.get("file"):
            raise ValueError("misaligned primary and timing prompts")
        common = float(r["common_ms"])
        if common <= 0 or set(r["batches"]) != {"1", "32", "128", "222"}:
            raise ValueError("incomplete timing workload")
        for k, batch in r["batches"].items():
            y = gold[order[: int(k)]]
            if set(batch) != set(METHODS):
                raise ValueError("missing timing method")
            best_exact = min(
                np.median(batch[m]["times_ms"]) for m in EXACT if not batch[m].get("unavailable")
            )
            for m, v in batch.items():
                if v.get("unavailable"):
                    if m != "exact128":
                        raise ValueError("required exact control unavailable")
                    continue
                if (
                    len(v["times_ms"]) != 3
                    or not np.isfinite(v["times_ms"]).all()
                    or min(v["times_ms"]) <= 0
                ):
                    raise ValueError("need three valid warm-timed repetitions")
                prediction = np.asarray(v["values"])
                if prediction.shape != y.shape or not np.isfinite(prediction).all():
                    raise ValueError("invalid query predictions")
                trained = m not in (*EXACT, "midpoint", "ig5")
                expected_fit = gold[list(train)] if trained else np.empty(0)
                np.testing.assert_allclose(v["train_values"], expected_fit, rtol=RTOL, atol=ATOL)
                if m in EXACT:
                    np.testing.assert_allclose(prediction, y, rtol=RTOL, atol=ATOL)
                ms = float(np.median(v["times_ms"]))
                s = samples.setdefault((k, m), [])
                s.append(
                    dict(
                        marginal_ms=ms,
                        end_to_end_ms=ms + common,
                        paired_speedup=(best_exact + common) / (ms + common),
                        marginal_paired_speedup=best_exact / ms,
                        mae=float(abs(prediction - y).mean()),
                    )
                )
    report = {}
    for (k, m), values in samples.items():
        entry = {
            field: float(np.median([r[field] for r in values])) for field in values[0] if field != "mae"
        }
        entry.update(mae=float(np.mean([r["mae"] for r in values])), prompts=len(values))
        report.setdefault(k, {})[m] = entry
    return report


def score(directory):
    max_pairs = read(directory / "frozen.json")["max_pairs"]
    if max_pairs != 3:
        raise ValueError("unexpected development result")
    result = dict(cells={}, timings={})
    for model in ("qwen", "large"):
        lookup = {}
        refs = []
        for task in ("ioi", "colors"):
            rows = read(directory / f"primary-{model}-{task}.json")["rows"]
            result["cells"][f"{model}-{task}"] = accuracy(rows, max_pairs)
            lookup[task] = {r["index"]: r for r in rows}
            refs.extend((task, r["index"]) for r in rows[:2])
        sample = read(directory / f"amortized-{model}.json")
        if (
            sample["max_pairs"] != max_pairs
            or sample["model"] != model
            or [(r["task"], r["index"]) for r in sample["rows"]] != refs
        ):
            raise ValueError("timing run must use the first two held-out prompts of each task")
        result["timings"][model] = speed_report(sample["rows"], lookup)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    gpu = sub.add_parser("gpu")
    for k in ("inputs", "protocol", "directory"):
        gpu.add_argument(k, type=Path)
    gpu.add_argument("model", choices=("qwen", "large"))
    metrics = sub.add_parser("score")
    metrics.add_argument("directory", type=Path)
    metrics.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.stage == "gpu":
        collect(args.inputs, args.protocol, args.directory, args.model)
    else:
        save(args.out, score(args.directory))


if __name__ == "__main__":
    main()
