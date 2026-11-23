"""Development-conditioned accuracy target, using the unchanged primary timing harness."""

import argparse
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


def pack(directory, alternate, name):
    dev = configure(read(directory / f"development-{name}.json"))
    devpath = alternate / f"development-{name}.json"
    if dev != read(devpath):
        raise ValueError("alternate policy differs from development rule")
    path = directory / f"heldout-{name}.json"
    reference = read(path)
    raw = read(alternate / f"heldout-{name}.json")
    if raw["freeze"] != sha(devpath) or reference["freeze"] != sha(directory / f"development-{name}.json"):
        raise ValueError("wrong policy checkpoint")
    records = []
    for a, b in zip(reference["rows"], raw["rows"], strict=True):
        for key in ("id", "task", "template", "labels", "clean_top", "donor_top"):
            if a[key] != b[key]:
                raise ValueError("unaligned prompt")
        discrepancies = {
            key: float(np.max(abs(np.array(a[key]) - np.array(b[key]))))
            for key in ("gradient", "calibration", "gold")
        }
        if max(discrepancies.values()) > 0.001 or b["native_max_error"] > 0.001:
            raise ValueError("changed native activations")
        records.append(
            {
                **{k: b[k] for k in ("id", "common_ms", "timings_ms", "methods", "native_max_error")},
                "max_abs": discrepancies,
            }
        )
    return dict(
        primary=sha(path),
        development=reference["freeze"],
        targets=dev["half_midpoint_targets"],
        rows=records,
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
            or run["targets"] != dev["half_midpoint_targets"]
        ):
            raise ValueError("run mismatch")
        for task in TASKS:
            operating = dev["tasks"][task]["operating"]
            pts = []
            for ref, raw, new in zip(refs, orig["rows"], run["rows"], strict=True):
                if ref["id"] != raw["id"] or raw["id"] != new["id"]:
                    raise ValueError("record mismatch")
                if ref["task"] == task:
                    pts.append(point({**raw, **new}, ref, operating, grid_replay(raw)))
            target = run["targets"][task]
            cells[name + "-" + task] = dict(
                target=target, methods={m: method_summary(pts, operating, m) for m in (*METHODS, "exact")}
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
    for action in ("configure", "pack"):
        cmd = sub.add_parser(action)
        for name in ("directory", "alternate"):
            cmd.add_argument(name, type=Path)
        cmd.add_argument("model", choices=CHECKPOINTS)
    cmd = sub.add_parser("score")
    cmd.add_argument("directory", type=Path)
    cmd.add_argument("out", type=Path)
    a = p.parse_args()
    if a.stage == "configure":
        save(
            a.alternate / f"development-{a.model}.json",
            configure(read(a.directory / f"development-{a.model}.json")),
        )
    elif a.stage == "pack":
        save(a.directory / f"half-{a.model}.json", pack(a.directory, a.alternate, a.model))
    else:
        save(a.out, report(a.directory))
