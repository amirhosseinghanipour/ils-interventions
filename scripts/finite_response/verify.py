"""Recompute headline comparisons from public per-article errors and token IDs."""

import argparse
import json
from pathlib import Path

import numpy as np


def verify(root):
    read = lambda name: json.loads((root / name).read_text())
    manifest = read("articles.json")
    test = [r for r in manifest if r["split"] == "test"]
    assert len(test) == 80 and len(set(r["article"] for r in manifest)) == len(manifest) == 128
    rng = np.random.default_rng(531809)
    indices = rng.integers(0, len(test), size=(10000, len(test)))
    result = {}
    # Joint bands across both candidates, checkpoints, and comparator family.
    entries = []
    for model in ("qwen", "smol"):
        data = read(model + "-rows.json")
        decoded = read(model + "-decode.json")
        summary = read(model + ".json")
        assert [r["qid"] for r in test] == [r["qid"] for r in data] == [r["qid"] for r in decoded]
        energy = np.array([r["energy"] for r in data])
        assert (energy > 0).all()
        errors = {k: np.array([r["methods"][k]["sse"] for r in data]) for k in data[0]["methods"]}
        for k, v in errors.items():
            np.testing.assert_allclose(
                v.sum() / energy.sum(), summary["methods"][k]["normalized_squared_error"]["value"]
            )
        for row, gen in zip(data, decoded, strict=True):
            tokens = gen["tokens"]
            for k, seq in tokens.items():
                if k != "exact":
                    assert gen["token_agreement"][k] == (seq == tokens["exact"])
                    assert row["methods"][k]["changed"] == (seq != tokens["no_change"])
            assert row["actual_changed"] == (tokens["exact"] != tokens["no_change"])
            assert tokens["exact"][0] == row["first_token"]
        speed = [r["checks"]["median_ms"] for r in data if "median_ms" in r["checks"]]
        assert len(speed) == 8
        model_result = dict(
            articles=len(data),
            clean_answer_correct=sum(r["clean_correct"] for r in decoded),
            exact_ended=sum(r["exact_ended"] for r in decoded),
            exact_matches_replacement=sum(r["exact_matches_replacement"] for r in decoded),
            exact_text_changed=sum(r["actual_text_change"] for r in decoded),
            speed_ratio_median=float(np.median([d["exact"] / d["secant"] for d in speed])),
            full_text_matches={
                k: sum(r["text_agreement"][k] for r in decoded) for k in decoded[0]["text_agreement"]
            },
            changed_text_matches={
                k: sum(r["actual_text_change"] and r["text_agreement"][k] for r in decoded)
                for k in decoded[0]["text_agreement"]
            },
            max_article_energy_share=float(energy.max() / energy.sum()),
            comparisons={},
        )
        result[model] = model_result
        for candidate in ("secant_transport", "prefix_secants"):
            comp = {}
            model_result["comparisons"][candidate] = comp
            controls = [k for k in errors if k not in ("secant_transport", "prefix_secants")]
            for k in controls:
                diff = errors[k] - errors[candidate]
                value = diff.sum() / energy.sum()
                boot = diff[indices].sum(1) / energy[indices].sum(1)
                entry = dict(
                    gain=float(value),
                    articles_with_lower_error=int((diff > 0).sum()),
                    median_article_gain=float(np.median(diff / energy)),
                )
                comp[k] = entry
                entries.append((value, boot, entry))
    standard = np.array([v.std(ddof=1) for _, v, _ in entries])
    nonzero = standard > 1e-12
    standardized = (
        np.stack([samples - value for value, samples, _ in entries])[nonzero] / standard[nonzero, None]
    )
    critical = float(np.quantile(np.max(np.abs(standardized), axis=0), 0.95))
    for (value, _, entry), se in zip(entries, standard, strict=True):
        entry["simultaneous_ci95"] = [float(value - critical * se), float(value + critical * se)]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(verify(args.results), indent=2, allow_nan=False) + "\n")
