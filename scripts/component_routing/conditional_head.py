"""Measure the marginal effect of the last head in each saved random permutation."""

import argparse
import json
from pathlib import Path


def counts(rows, threshold):
    return {
        "effect_difference_at_least_threshold": sum(
            abs(r["with_other_heads"] - r["alone"]) >= threshold for r in rows
        ),
        "opposite_signs_both_at_least_threshold": sum(
            r["with_other_heads"] * r["alone"] < 0
            and min(abs(r["with_other_heads"]), abs(r["alone"])) >= threshold
            for r in rows
        ),
    }


def summarize(directory):
    records = []
    for model in ("qwen", "large"):
        groups = json.loads((directory / f"groups-{model}.json").read_text())
        controls = json.loads((directory / f"controls-{model}.json").read_text())
        for task in ("addition", "subtraction"):
            for group, control in zip(groups[task], controls[task], strict=True):
                assert (group["index"], group["counter"]) == (control["index"], control["counter"])
                heads = group["order"]
                assert sorted(heads) == list(range(len(heads)))
                assert group["sizes"][-2:] == [len(heads) - 1, len(heads)]
                observed = group["gold"]
                singles = control["predictions"]["exact_additive"]
                # Subtracting the two sums of exact singletons isolates the same last head.
                records.append(
                    dict(
                        model=model,
                        task=task,
                        index=group["index"],
                        counter=group["counter"],
                        layer=group["layer"],
                        head=heads[-1],
                        alone=singles[-1] - singles[-2],
                        with_other_heads=observed[-1] - observed[-2],
                    )
                )
    return dict(
        observations=len(records),
        threshold_counts={str(t): counts(records, t) for t in (0.001, 0.01, 0.1)},
        rows=records,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(summarize(args.directory), indent=2, allow_nan=False) + "\n")
