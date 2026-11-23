"""Audit the pairs and measure how much variation a source-only predictor must miss."""
import argparse
import json
from pathlib import Path

import numpy as np

from run import PROTOCOL, save_json, summarize


def analyze(root, split):
    rows = [json.loads(p.read_text()) for p in sorted(root.glob(split + "-[0-9]*.json"))]
    assert len(rows) == len({r["index"] for r in rows}) == PROTOCOL["counts"][split]
    saved = json.loads((root / (split + "-summary.json")).read_text())
    assert summarize(rows) == saved
    diagnostics = {}
    for label in ("donor1", "donor005", "random1"):
        irreducible, zero_error, logit_changes = [], [], []
        for r in rows:
            path = root / f'{split}-{r["index"]:03}.npz'
            with np.load(path, allow_pickle=False) as data:
                # Responses are one-sided (plus minus clean), in final residual coordinates.
                a = data[f"relevant_{label}_plus"].astype(np.float64)
                b = data[f"irrelevant_{label}_plus"].astype(np.float64)
                np.testing.assert_allclose(a[:r["cue"]], b[:r["cue"]], rtol=0, atol=2e-6)
                a, b = a[-1], b[-1]
                irreducible.append(np.dot(a-b, a-b)/4)
                zero_error.append((np.dot(a,a) + np.dot(b,b))/2)
            shifts = []
            for query in ("relevant", "irrelevant"):
                m = r["measurements"][query]
                shifts.append(m[label + "_plus"]["donor_minus_original"] - m["clean"]["donor_minus_original"])
            logit_changes.append(shifts)
        err, denom = map(np.asarray, (irreducible, zero_error))
        assert denom.sum() > 0
        draws = np.random.default_rng(112312).integers(len(rows), size=(10000, len(rows)))
        ratios = err[draws].sum(1) / denom[draws].sum(1)
        shifts = np.asarray(logit_changes)
        contrast = shifts[:,0] - shifts[:,1]
        diagnostics[label] = {
            "relevant_mean_logit_change": float(shifts[:,0].mean()),
            "irrelevant_mean_logit_change": float(shifts[:,1].mean()),
            "paired_mean_difference": float(contrast.mean()),
            "pairs_with_positive_difference": int(np.sum(contrast > 0)),
            "source_only_mse_lower_bound_over_zero_predictor_mse": float(err.sum()/denom.sum()),
            "lower_bound_fraction_ci95": np.quantile(ratios,[.025,.975]).tolist(),
        }
    return {"split":split, "behavior":saved, "destination_residual":diagnostics,
            "definition":"For equally weighted queries sharing an edit, any identical prediction has MSE at least ||a-b||²/4. Normalize the pooled bound by pooled zero-predictor MSE. Evaluation-only diagnostic conditional on sampled prefixes. One norm-matched random direction per prefix."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("split", choices=["dev","test"])
    args = p.parse_args()
    result = analyze(args.data, args.split)
    save_json(args.data / (args.split + "-audit.json"), result)
    print(json.dumps(result, indent=2))
