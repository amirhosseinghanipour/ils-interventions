"""Development-conditioned accuracy target, using the unchanged primary timing harness."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .score import (
    METHODS,
    TASKS,
    TOL,
    CHECKPOINTS,
    average_grid,
    budgets,
    check_manifest,
    cluster_inputs,
    grid_replay,
    method_summary,
    point,
    policy,
    read,
    sha,
    simultaneous,
)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")


def configure(dev):
    targets = {}
    tasks = {}
    for task, original in dev["tasks"].items():
        grid = average_grid([grid_replay(row) for row in original["rows"]])
        targets[task] = target = grid["midpoint"]["0"] / 2
        operating = {m: next((b for b in budgets(m) if grid[m][str(b)] <= target), None) for m in METHODS}
        tasks[task] = {**original, "operating": operating}
    return {**dev, "tasks": tasks, "half_midpoint_targets": targets}


def capture(oracle, row, train, target, config, common_ms, clock):
    """Retain each paid input and returned forecast outside the timed region."""
    from .fit import fit

    operating = config["operating"]
    chunks = {int(n): rec["chunk"] for n, rec in config["chunks"].items()}
    keys = {m: policy(b, m) for m, b in operating.items()}
    methods = sorted(set(keys.values()) | {"exact", "midpoint"})
    rng = np.random.default_rng(int(train[0]))
    timings, paid = {m: [] for m in methods}, {m: [] for m in methods}
    orders = []
    for _ in range(3):
        order = rng.permutation(methods).tolist()
        orders.append(order)
        for m in order:
            start = clock()
            if m == "exact":
                forecast = oracle.exact(target, chunk=chunks[512])
            else:
                b = 0 if m == "midpoint" else operating[m]
                gradient = oracle.gradient()
                labels = oracle.exact(train[:b], chunk=chunks[b]) if b else []
                forecast = fit(m, gradient, train[:b], labels, target)
            timings[m].append(clock() - start)
            rec = {"forecast": np.asarray(forecast, dtype=float).tolist()}
            if m != "exact":
                rec.update(gradient=list(gradient), calibration=list(labels))
            paid[m].append(rec)
    audit = oracle.audit([0, 65535, train[0], train[1], target[0], target[-1]])
    return dict(
        **{k: row[k] for k in ("id", "task", "template")},
        common_ms=common_ms,
        methods=keys,
        timings_ms=timings,
        orders=orders,
        paid=paid,
        native_max_error=float(audit),
    )


def replay_paid(raw, new, operating, grid):
    from .fit import design, fit

    if any(new[k] != raw[k] for k in ("id", "task", "template", "labels", "clean_top", "donor_top")):
        raise ValueError("prompt or answer mismatch")
    seed = int.from_bytes(hashlib.sha256(raw["id"].encode()).digest()[:4], "big")
    train, target = design(seed, pairs=128)
    out = point({**raw, **new}, raw, operating, grid)
    required = sorted(new["timings_ms"])
    if (
        sorted(new["paid"]) != required
        or len(new["orders"]) != 3
        or any(sorted(o) != required for o in new["orders"])
    ):
        raise ValueError("incomplete actual method trace")
    differences = {"calibration": 0.0, "gradient": 0.0, "exact": 0.0}
    grid_gap = {}

    def compare(key, actual, reference):
        a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all():
            raise ValueError("invalid paid " + key)
        diff = float(np.max(abs(a - b), initial=0.0))
        if diff > 0.001:
            raise ValueError(f"paid {key} differs from primary by {diff:g}")
        differences[key] = max(differences[key], diff)

    for m in required:
        reps = new["paid"][m]
        if len(reps) != 3:
            raise ValueError("expected three paid forecasts")
        for r in reps:
            pred = np.asarray(r["forecast"], dtype=float)
            if pred.shape != (512,) or not np.isfinite(pred).all():
                raise ValueError("incomplete paid forecast")
            np.testing.assert_allclose(pred, reps[0]["forecast"], rtol=2e-4, atol=3e-4)
            if m == "exact":
                compare("exact", pred, raw["gold"])
                continue
            b = 0 if m == "midpoint" else operating[m]
            expected = fit(m, r["gradient"], train[:b], r["calibration"], target)
            np.testing.assert_allclose(pred, expected, rtol=1e-8, atol=1e-9)
            compare("gradient", r["gradient"], raw["gradient"])
            compare("calibration", r["calibration"], raw["calibration"][:b])
    for m, key in new["methods"].items():
        err = float(
            np.mean(
                [
                    np.abs(np.asarray(x["forecast"]) - np.asarray(y["forecast"])).mean()
                    for x, y in zip(new["paid"][key], new["paid"]["exact"], strict=True)
                ]
            )
        )
        grid_gap[m] = abs(err - out["errors"][m])
        out["errors"][m] = err
    return out, differences, grid_gap


def collect(inputs, directory, name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import encode
    from .oracle import Oracle
    from .run import clock, masks, save as atomic_save

    devpath = directory / f"development-{name}.json"
    primary = directory / f"heldout-{name}.json"
    manifest = directory / "manifest.json"
    model_path = inputs / name
    dev = configure(read(devpath))
    orig = read(primary)
    if orig["freeze"] != sha(devpath) or sha(model_path / "model.safetensors") != CHECKPOINTS[name]:
        raise ValueError("unmatched primary policy or weights")
    stamp = dict(
        primary=sha(primary),
        development=sha(devpath),
        collector=sha(Path(__file__)),
        manifest=sha(manifest),
        scorer=sha(Path(__file__).with_name("score.py")),
    )
    if (
        any(
            dev["signature"][f] != sha(Path(__file__).with_name(f))
            for f in ("fit.py", "run.py", "oracle.py", "data.py")
        )
        or dev["signature"]["manifest"] != stamp["manifest"]
    ):
        raise ValueError("primary collection code or manifest changed")
    refs = [r for r in check_manifest(read(manifest)) if r["split"] == "heldout"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    records = []
    for row, raw in zip(refs, orig["rows"], strict=True):
        if row["id"] != raw["id"]:
            raise ValueError("manifest ordering changed")
        part = (
            directory
            / ("parts-half-timed-" + name)
            / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")
        )
        config = dev["tasks"][row["task"]]
        if not part.exists():
            ids, labels = encode(tok, row)
            start = clock()
            o = Oracle(model, ids, labels, dev["sites"])
            common = clock() - start
            rec = capture(o, row, *masks(row), config, common, clock)
            rec.update(labels=labels, clean_top=int(o.clean_top), donor_top=int(o.donor_top))
            atomic_save(part, dict(stamp=stamp, row=rec))
        stored = read(part)
        if stored["stamp"] != stamp:
            raise ValueError("stale half-target checkpoint")
        rec = stored["row"]
        replay_paid(raw, rec, config["operating"], raw["errors"])
        records.append(rec)
        if len(records) % 8 == 0:
            print(name, len(records), flush=True)
    atomic_save(
        directory / f"half-{name}.json", dict(**stamp, targets=dev["half_midpoint_targets"], rows=records)
    )


def report(directory):
    manifest = check_manifest(read(directory / "manifest.json"))
    refs = [r for r in manifest if r["split"] == "heldout"]
    cells = {}
    groups = {}
    for name in CHECKPOINTS:
        dev = configure(read(directory / f"development-{name}.json"))
        primary = directory / f"heldout-{name}.json"
        orig = read(primary)
        run = read(directory / f"half-{name}.json")
        if (
            sha(primary) != run["primary"]
            or sha(directory / f"development-{name}.json") != run["development"]
            or run["collector"] != sha(Path(__file__))
            or run["scorer"] != sha(Path(__file__).with_name("score.py"))
            or run["manifest"] != sha(directory / "manifest.json")
            or run["targets"] != dev["half_midpoint_targets"]
        ):
            raise ValueError("run mismatch")
        for task in TASKS:
            operating = dev["tasks"][task]["operating"]
            pts, diffs, gaps = [], [], []
            for ref, raw, new in zip(refs, orig["rows"], run["rows"], strict=True):
                if ref["id"] != raw["id"] or raw["id"] != new["id"]:
                    raise ValueError("record mismatch")
                if ref["task"] == task:
                    pt, diff, gap = replay_paid(raw, new, operating, grid_replay(raw))
                    pts.append(pt)
                    diffs.append(diff)
                    gaps.append(gap)
            target = run["targets"][task]
            cells[name + "-" + task] = dict(
                target=target,
                max_abs_paid_minus_primary={k: max(d[k] for d in diffs) for k in diffs[0]},
                max_abs_timed_mae_minus_grid={m: max(g[m] for g in gaps) for m in METHODS},
                methods={m: method_summary(pts, operating, m) for m in (*METHODS, "exact")},
            )
            if name == "gpt2":
                groups[task] = cluster_inputs(pts)
                groups[task][:, 0] += target - TOL
    gpt = simultaneous(groups)
    gpt["passes"] = gpt["positive_bounds"] and all(
        cells["gpt2-" + t]["methods"]["ils"]["policy"] == "ils" for t in TASKS
    )
    return dict(cells=cells, gpt2=gpt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    cmd = sub.add_parser("collect")
    cmd.add_argument("inputs", type=Path)
    cmd.add_argument("directory", type=Path)
    cmd.add_argument("model", choices=CHECKPOINTS)
    cmd = sub.add_parser("score")
    cmd.add_argument("directory", type=Path)
    cmd.add_argument("out", type=Path)
    a = p.parse_args()
    if a.stage == "collect":
        collect(a.inputs, a.directory, a.model)
    else:
        save(a.out, report(a.directory))
