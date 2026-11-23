"""Verify public measurements, frozen sample identity and exact-query histories on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from sampling import FILES


def read(path):
    return json.loads(path.read_text())


def same_refs(rows, refs):
    assert len(rows) == len(refs)
    for r, ref in zip(rows, refs, strict=True):
        assert {k: r[k] for k in ref} == ref


def verify(directory, inputs=None):
    manifest = directory / "manifest.json"
    primary = read(manifest)
    group = read(directory / "group-manifest.json")
    frozen = read(directory / "frozen.json")
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == frozen["manifest_sha256"]
    dev = read(directory / "qwen-addition-validation.json")["rows"]
    same_refs(dev, primary["addition-validation"])
    assert len(dev) == 20
    for strength, loss in frozen["dev_mae"].items():
        errors = []
        for r in dev:
            mid, full, gold = (np.asarray(r[k]) for k in ("midpoint", "full", "gold"))
            assert mid.shape == full.shape == gold.shape == (45,)
            errors.append(abs(mid + float(strength) * (full - mid) - gold).mean())
        np.testing.assert_allclose(np.mean(errors), loss, rtol=1e-12)
    assert (
        float(min(frozen["dev_mae"], key=lambda k: (frozen["dev_mae"][k], float(k)))) == frozen["strength"]
    )
    if inputs is not None:
        seen = set()
        for split, digest in FILES.items():
            raw = (inputs / (split + ".jsonl")).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == digest
            records = [json.loads(line) for line in raw.splitlines()]
            task = split.split("-")[0]
            for ref in primary[split] + (group[task] if split.endswith("test") else []):
                row = records[ref["index"]]
                assert row["template"] == ref["template"]
                for src in (row, row[ref["counter"]]):
                    pair = (task, str(src["operand1"]), str(src["operand2"]))
                    assert pair not in seen
                    seen.add(pair)
    totals = dict(
        validation_patches=20 * 45,
        test_patches=0,
        timed_queries_checked=0,
        interior_group_patches=0,
        group_endpoint_controls=0,
    )
    for model, heads in (("qwen", 14), ("large", 32)):
        caps = read(directory / (model + "-budget.json"))
        a, b = (np.median(caps[k][1:]) for k in ("pilot_ils_ms", "pilot_batch_ms"))
        np.testing.assert_allclose([caps["tight"], caps["roomy"]], [1.2 * a + b, 1.2 * a + 4 * b])
        groups = read(directory / ("groups-" + model + ".json"))
        for task in ("addition", "subtraction"):
            split = task + "-test"
            raw = read(directory / (model + "-" + split + ".json"))
            assert raw["keys"] == [[i, j] for i in (5, 11, 17) for j in range(heads + 1)]
            same_refs(raw["rows"], primary[split])
            assert len(raw["rows"]) == 48
            for n, r in enumerate(raw["rows"]):
                y = np.array(r["gold"])
                assert len(y) == len(r["patch_top"]) == 3 * (heads + 1) and np.isfinite(y).all()
                assert r["clean_correct"] == (r["clean_top"] == r["labels"][0])
                assert r["donor_correct"] == (r["donor_top"] == r["labels"][1])
                assert len(set(r["labels"])) == 2
                totals["test_patches"] += len(y)
                methods = r["methods"]
                assert set(methods) == {"ils", "midpoint", "atp", "guided", "scan", "input_midpoint"}
                np.testing.assert_array_equal(methods["atp"]["raw"], methods["guided"]["raw"])
                for name, run in methods.items():
                    v = np.array(run["raw"])
                    assert v.shape == y.shape and np.isfinite(v).all()
                    rng = np.random.default_rng(94817 + n)
                    order = np.argsort(-abs(v), kind="stable").tolist()
                    if name == "guided":
                        pilot = [int(rng.choice(s)) for s in np.array_split(order, 4)]
                        order = pilot + [j for j in order if j not in pilot]
                    if name == "scan":
                        assert (v == 0).all()
                        order = rng.permutation(len(v)).tolist()
                    start = 0
                    previous = run["raw_ms"]
                    assert previous >= 0
                    for step in run["history"]:
                        k = min(8 if name == "scan" else 4, len(v) - start)
                        ix = step["indices"]
                        assert previous <= caps["roomy"] and step["ms"] >= previous
                        assert k > 0 and ix == order[start : start + k]
                        np.testing.assert_allclose(y[ix], step["values"], rtol=3e-4, atol=6e-4)
                        if name == "guided" and start == 0:
                            np.testing.assert_allclose(
                                step["correction"], np.median(step["values"] - v[ix])
                            )
                        else:
                            assert step["correction"] is None
                        start += k
                        previous = step["ms"]
                        totals["timed_queries_checked"] += k
            same_refs(groups[task], group[task])
            assert len(groups[task]) == 24
            for n, r in enumerate(groups[task]):
                np.testing.assert_array_equal(
                    r["order"], np.random.default_rng(34399 + n).permutation(heads)
                )
                assert r["layer"] == 11 and r["sizes"] == [
                    0,
                    1,
                    heads // 4,
                    heads // 2,
                    heads - heads // 4,
                    heads - 1,
                    heads,
                ]
                y = np.array(r["gold"])
                for key, v in r["predictions"].items():
                    assert len(v) == len(y) == 7 and np.isfinite(v).all()
                assert abs(y[0]) < 6e-4 and np.isclose(
                    y[-1], r["matched_endpoint_effect"], rtol=3e-4, atol=6e-4
                )
                for key in ("secant_midpoint", "secant_only", "nearest", "smooth"):
                    np.testing.assert_allclose(
                        np.array(r["predictions"][key])[[0, -1]], y[[0, -1]], rtol=3e-4, atol=6e-4
                    )
                totals["interior_group_patches"] += len(y) - 2
                totals["group_endpoint_controls"] += 2
    return totals


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--inputs", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.directory, a.inputs), indent=2))
