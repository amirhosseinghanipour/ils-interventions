"""Lower-bound variation across later positions from three saved horizon means."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pooled_energy_bound(means, kernels):
    # Rows of 'means' are K @ D. The minimum of sum_j p_j ||D_j||² over
    # feasible D is ||pinv(K diag(p^-1/2)) @ means||², with p from gamma=.9.
    design = kernels / np.sqrt(kernels[1])
    coefficients = np.linalg.pinv(design, rcond=1e-10)
    gram = means @ means.T
    return np.sum((coefficients.T @ coefficients) * gram)


def replay(repo, raw_data=None):
    root = raw_data or repo / "runs/internal-replication"
    raw_rows = json.loads((root / "rows.json").read_text())
    archive = json.loads((repo / "artifacts/qwen35-predictive-coverage.json").read_text())
    report = {}
    for model in archive["results"]:
        energy, mean_energy = [], []
        for i, row in enumerate(raw_rows):
            if row["split"] != "test":
                continue
            m = 63 - row["t"]
            if m <= 1:
                continue
            raw = (root / model["model"] / f"{i}.f32").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == model["samples_sha256"][i]
            data = np.frombuffer(raw, "<f4").reshape(5, 5, model["width"]).astype(float)
            w = np.array([g ** np.arange(m) for g in [.5, .9, 1.]])
            w /= w.sum(axis=1, keepdims=True)
            future_w = w[:, 1:] / (1. - w[:, :1])
            means = (data[:, 2:] - w[None, :, :1] * data[:, 1:2]) / (1. - w[None, :, :1])
            # The final block has analytically zero later response; omit roundoff there.
            bounds = np.array([pooled_energy_bound(y, future_w) for y in means[:-1]])
            e_mean = np.square(means[:-1, 1]).sum(axis=1)
            assert np.all(bounds >= e_mean * (1 - 1e-5))
            energy.append(bounds)
            mean_energy.append(e_mean)
        energy, mean_energy = np.array(energy), np.array(mean_energy)
        draw = np.random.default_rng(112313).integers(len(energy),size=(10000,len(energy)))
        lb = 1 - mean_energy.sum(0) / energy.sum(0)
        ci = np.quantile(1-mean_energy[draw].sum(1)/energy[draw].sum(1),[.025,.975],axis=0)
        report[model["model"]] = {str(l):dict(min_fraction=float(v),ci95=c.tolist())
                                  for l,v,c in zip([7,15,23,30],lb,ci.T)}
    return dict(gamma=.9, pairs=len(energy), results=report,
                metric="Lower bound, from three stored horizon projections, on the pooled fraction of strictly later gamma-.9 weighted squared response energy lost in the signed average. Variation in magnitude, direction, or both contributes. Percentile intervals resample held-out pairs.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--raw-data", type=Path, help="Path to the archived internal-replication directory")
    args = p.parse_args()
    args.output.write_text(json.dumps(replay(args.repo, args.raw_data), indent=2) + "\n")
