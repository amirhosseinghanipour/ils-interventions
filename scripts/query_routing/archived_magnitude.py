"""Measure the size of the observed later-only response using the existing signed runs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def measure(repo, raw_data=None):
    source = raw_data or repo / "runs/internal-replication"
    rows = json.loads((source / "rows.json").read_text())
    archive = json.loads((repo / "artifacts/qwen35-predictive-coverage.json").read_text())
    layers = archive["protocol"]["layers"]
    out = {}
    for model in archive["results"]:
        same, future = [], []
        for index, row in enumerate(rows):
            if row["split"] != "test":
                continue
            raw = (source / model["model"] / f"{index}.f32").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == model["samples_sha256"][index]
            count = 63 - row["t"]
            if count == 1:
                continue
            data = np.frombuffer(raw, "<f4").reshape(len(layers), 5, model["width"]).astype(np.float64)
            w0 = 0.1 / (1 - 0.9 ** count)
            y0 = data[:, 1]
            yl = (data[:, 3] - w0 * y0) / (1 - w0)
            same.append(np.einsum("ij,ij->i", y0, y0))
            future.append(np.einsum("ij,ij->i", yl, yl))
        same, future = np.asarray(same), np.asarray(future)
        assert len(same) == 62
        draws = np.random.default_rng(112311).integers(len(same), size=(10000, len(same)))
        pooled = np.sqrt(future.sum(0) / same.sum(0))
        interval = np.quantile(np.sqrt(future[draws].sum(1) / same[draws].sum(1)), [.025,.975], axis=0)
        out[model["model"]] = {str(layer): {"ratio_of_rms_norms": float(value), "ci95": bounds.tolist()}
                              for layer, value, bounds in zip(layers, pooled, interval.T)}
    return dict(gamma=0.9, epsilon=0.05, pairs=62,
                metric="RMS norm of normalized signed later-position mean / RMS norm of same-position signed change; bootstrap whole held-out pairs. Not per-token magnitudes.",
                results=out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--raw-data", type=Path, help="Path to the archived internal-replication directory")
    args = p.parse_args()
    args.output.write_text(json.dumps(measure(args.repo, args.raw_data), indent=2) + "\n")
