"""Summarize the prespecified larger-checkpoint replication by paired article."""

import argparse
import json
from pathlib import Path

import numpy as np


def scores(path):
    rows = json.loads(path.read_text())
    assert len(rows) == 48 and len(set(r["article"] for r in rows)) == 24
    assert all(
        a["article"] == b["article"] and (a["kind"], b["kind"]) == ("edited_answer", "other_answer")
        for a, b in zip(rows[::2], rows[1::2], strict=True)
    )
    draws = np.random.default_rng(710013).integers(24, size=(10000, 24))
    summary = {}
    for kind in ("all", "edited_answer", "other_answer"):
        part = [r for r in rows if kind == "all" or r["kind"] == kind]
        grouped = lambda v: np.array(v).reshape(24, -1).sum(1)
        energy = grouped([r["energy"] for r in part])
        table = {}
        for name in rows[0]["sse"]:
            err = grouped([r["sse"][name] for r in part])
            diff = err - grouped([r["sse"]["two_anchor"] for r in part])
            ci = np.quantile(diff[draws].sum(axis=1) / energy[draws].sum(axis=1), [0.025, 0.975])
            table[name] = dict(
                normalized_error=float(err.sum() / energy.sum()),
                gain_with_two_anchor=float(diff.sum() / energy.sum()),
                paired_ci95=ci.tolist(),
            )
        summary[kind] = dict(
            questions=len(part),
            clean_first_correct=sum(r["clean_top"] == r["reference_first"] for r in part),
            patch_changes_first=sum(r["exact_top"] != r["clean_top"] for r in part),
            methods=table,
        )
    checks = [r["relative_step_disagreement"] for r in rows if r["relative_step_disagreement"]]
    assert len(checks) == 8
    summary["relative_step_disagreement"] = {
        name: dict(median=float(np.median([d[name] for d in checks])), maximum=max(d[name] for d in checks))
        for name in checks[0]
    }
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    args = p.parse_args()
    read = lambda k: json.loads((args.dir / f"native-{k}.json").read_text())
    assert [r["qid"] for r in read("4b")] == [r["qid"] for r in read("9b")]
    out = {name: scores(args.dir / f"native-{name}.json") for name in ("4b", "9b")}
    (args.dir / "native-summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
