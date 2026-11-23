"""Dual-form ridge in fp64 for the low-sample high-dimensional replication."""

import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from ilslens.scalar_transport import scalar_transport
from ilslens.eval.faithfulness import paired_prompt_r2 as summarize
from ilslens.experiment import write_json

root = Path("runs/internal-replication")
config = json.loads((Path(__file__).parent / "protocol.json").read_text())
n = config["train_prompts"]
eps = config["penalties"]
layers = config["layers"]
gammas = config["gammas"]


def paths(x, y, xt, identity):
    # Positive penalties shrink toward the specified matrix. At zero use the
    # minimum-norm unregularized map, including off the span of training rows.
    vals, v = np.linalg.eigh(x @ x.T)
    keep = vals > vals.max() * max(x.shape) * np.finfo(float).eps
    cross = (xt @ x.T) @ v
    projected = v.T @ y
    adjustment = v.T @ (y - x) if identity else projected
    for lam in eps:
        if lam == 0:
            yield (cross[:, keep] / vals[keep]) @ projected[keep]
        else:
            pred = (cross / (np.maximum(vals, 0) + len(x) * lam)) @ adjustment
            yield pred + xt if identity else pred


def predict(x, y, xt, identity):
    costs = np.zeros(len(eps))
    folds = np.array_split(np.random.default_rng(config["seed"]).permutation(len(x)), 5)
    for held in folds:
        train = np.ones(len(x), dtype=bool)
        train[held] = False
        for j, pred in enumerate(paths(x[train], y[train], x[held], identity)):
            costs[j] += np.square(pred - y[held]).sum()
    winner = int(costs.argmin())
    return list(paths(x, y, xt, identity))[winner], eps[winner]


def score(model):
    rows = json.loads((root / "rows.json").read_text())
    paths_in = [root / model / f"{i}.f32" for i in range(len(rows))]
    arrays = np.stack(
        [
            np.fromfile(p, "<f4").reshape(len(layers), 1 + len(gammas), -1)
            for p in paths_in
        ]
    ).astype(float)
    assert np.isfinite(arrays).all()
    assert len(rows) == n + config["test_prompts"] and all(
        r["split"] == ("train" if i < n else "test") for i, r in enumerate(rows)
    )
    groups = torch.arange(len(rows) - n)
    pos = np.array([r["t"] for r in rows])
    results = []
    archive = {}
    for l, layer in enumerate(layers):
        x = arrays[:, l, 0]
        for g, gamma in enumerate(gammas):
            y = arrays[:, l, g + 1]
            count = config["tokens"] - 1 - pos
            w = 1 / count if gamma == 1 else (1 - gamma) / (1 - gamma**count)
            skip = x * w[:, None]
            if layer == 31:
                assert np.linalg.norm(skip - y) / np.linalg.norm(y) < 1e-4
            preds = dict(identity=x[n:], horizon_scaled_identity=skip[n:])
            penalties = {}
            for name, identity in [("ILS", True), ("zero_prior_ridge", False)]:
                preds[name], penalties[name] = predict(x[:n], y[:n], x[n:], identity)
            metrics = summarize(
                torch.from_numpy(y[n:]),
                {k: torch.from_numpy(v) for k, v in preds.items()},
                groups,
                archive,
                f"{layer}/{g}/total",
            )
            innovation = None
            if layer != 31:
                correction, lam = predict(x[:n], (y - skip)[:n], x[n:], False)
                scalar = scalar_transport(x[:n], y[:n], w[:n], x[n:], w[n:])
                total = summarize(
                    torch.from_numpy(y[n:]),
                    {
                        "ILS": torch.from_numpy(skip[n:] + correction),
                        "scalar": torch.from_numpy(scalar),
                        "direct": torch.from_numpy(skip[n:]),
                    },
                    groups,
                    archive,
                    f"{layer}/{g}/correction",
                )
                rem = summarize(
                    torch.from_numpy((y - skip)[n:]),
                    {"ILS": torch.from_numpy(correction)},
                    groups,
                    archive,
                    f"{layer}/{g}/remainder",
                )
                innovation = dict(
                    penalty=lam,
                    total=total["ILS"],
                    gain_over_direct=total["ILS minus direct"],
                    remainder=rem["ILS"],
                    scalar=total["scalar"],
                    gain_over_scalar=total["ILS minus scalar"],
                )
            results.append(
                dict(
                    layer=layer,
                    gamma=gamma,
                    penalties=penalties,
                    metrics=metrics,
                    innovation=innovation,
                )
            )
    np.savez_compressed(f"artifacts/qwen35-{model}-sufficient.npz", **archive)
    return dict(
        model=model,
        width=arrays.shape[-1],
        rows=results,
        samples_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths_in],
    )


if __name__ == "__main__":
    subprocess.run(
        ["sha256sum", "--quiet", "-c", str(root / "provenance.sha256")], check=True
    )
    write_json(
        "artifacts/qwen35-predictive-coverage.json",
        dict(
            protocol=config,
            provenance=(root / "provenance.sha256").read_text().splitlines(),
            results=[score(m) for m in ("4b", "9b")],
        ),
    )
