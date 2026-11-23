"""Audit the released prompt-level graph scores and compute paired template bootstrap intervals."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def summarize(data, draws=4000, seed=30177):
    full = np.array(data["test_full"])
    empty = np.array(data["test_empty"])
    labels = data["test_templates"]
    grid = data["grid"]
    assert len(full) == len(empty) == len(labels) == data["filtered_public_prompts"]
    assert len(set(labels).intersection(data["development_templates"])) == 0
    names, indices = np.unique(labels, return_inverse=True)
    counts = np.bincount(indices)
    rng = np.random.default_rng(seed)
    # Resample templates with replacement and retain every prompt within each sampled template.
    weights = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=draws)

    def sums(values):
        return np.stack([values[..., indices == j].sum(axis=-1) for j in range(len(names))], axis=-1)

    omit = np.ones((len(names), len(names))) - np.eye(len(names))
    omit_denom = omit @ sums(full - empty)
    denom = weights @ sums(full - empty)
    assert np.all(np.abs(denom) > 1e-6) and np.all(np.abs(omit_denom) > 1e-6)
    out = {
        "split": data["split"],
        "eligible_prompts": len(full),
        "development_calibration_prompts": len(data["development_indices"]),
        "eligible_development_prompts": data["eligible_development_prompts"],
        "public_test_templates_and_counts": dict(Counter(labels)),
        "clean_minus_empty": float(np.mean(full - empty)),
        "bootstrap": {
            "unit": "template",
            "draws": draws,
            "seed": seed,
            "clusters": len(counts),
            "interval": "paired percentile 95%; conditions on calibration prompts and policy",
        },
        "curves": {},
    }
    for mode, methods in data["curves"].items():

        def curve(ys, mode=mode):
            ordinate = ys if mode == "CPR" else np.abs(1 - ys)
            return np.trapezoid(ordinate, grid, axis=-1)

        stats, draws_by_method, leave_one_out = {}, {}, {}
        for method, result in methods.items():
            margins = np.array([pt["margins"] for pt in result["points"]])
            responses = margins - empty
            faith = responses.mean(axis=1) / np.mean(full - empty)
            np.testing.assert_allclose(faith, [pt["faithfulness"] for pt in result["points"]], atol=1e-10)
            score = curve(faith)
            np.testing.assert_allclose(score, result["score"], atol=1e-10)
            grouped = sums(responses).T
            samples = curve((weights @ grouped) / denom[:, None])
            leave_one_out[method] = curve((omit @ grouped) / omit_denom[:, None])
            stats[method] = {
                "score": float(score),
                "interval": np.quantile(samples, [0.025, 0.975]).tolist(),
                "faithfulness": faith.tolist(),
                "retained_edges": [pt["edges"] for pt in result["points"]],
            }
            draws_by_method[method] = samples
        comparisons = {}
        for comparator in methods:
            if comparator == "pair":
                continue
            deltas = draws_by_method["pair"] - draws_by_method[comparator]
            single_omissions = leave_one_out["pair"] - leave_one_out[comparator]
            comparisons[comparator] = {
                "pair_minus_comparator": stats["pair"]["score"] - stats[comparator]["score"],
                "interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
                "leave_one_template_out_delta_range": [
                    float(single_omissions.min()),
                    float(single_omissions.max()),
                ],
            }
        out["curves"][mode] = {"methods": stats, "paired_differences": comparisons}
    return out


def summarize_additional_example(primary, supplement):
    key = "organizer_hf_example"
    assert supplement["method"] == key and supplement["grid"] == primary["grid"]
    paired = dict(primary)
    paired["curves"] = {
        mode: {"pair": methods["pair"], key: supplement["curves"][mode][key]}
        for mode, methods in primary["curves"].items()
    }
    summary = summarize(paired)
    return {
        "scope": supplement["scope"],
        "source": supplement["source"],
        "revision": supplement["revision"],
        "primary_sha256": supplement["fingerprint"]["primary_sha256"],
        "eligible_prompts": summary["eligible_prompts"],
        "bootstrap": summary["bootstrap"],
        "graph_comparison": supplement["graph_comparison"],
        "runtime": supplement["runtime"],
        "curves": {
            mode: {
                key: curve["methods"][key],
                "primary_pair_minus_organizer_hf_example": curve["paired_differences"][key],
            }
            for mode, curve in summary["curves"].items()
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument(
        "--additional-example", type=Path, help="Summarize a separate post-primary organizer release"
    )
    a = p.parse_args()
    primary_bytes = a.artifact.read_bytes()
    data = json.loads(gzip.decompress(primary_bytes))
    if a.additional_example:
        supplement = json.loads(gzip.decompress(a.additional_example.read_bytes()))
        assert supplement["fingerprint"]["primary_sha256"] == hashlib.sha256(primary_bytes).hexdigest()
        result = summarize_additional_example(data, supplement)
    else:
        result = summarize(data)
    print(json.dumps(result, indent=2, allow_nan=False))
