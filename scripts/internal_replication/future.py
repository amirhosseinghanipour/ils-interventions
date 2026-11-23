"""Strictly later-position control; no direct skip or source-position update in target."""

import json
from pathlib import Path
import numpy as np
import torch
from ilslens.experiment import write_json
from ilslens.eval.faithfulness import paired_prompt_r2
from ilslens.runtime import resolve_device
from scripts.predictive_coverage import samples, cv_ridge
from scripts.internal_replication.score import predict

gammas = [0.5, 0.9, 1.0]
out = []
archive = {}


def extract(data, length):
    x, y, t, g = data
    keep = t < length - 2
    x, y, t, g = x[keep], y[keep].double(), t[keep], g[keep]
    m = length - 1 - t
    targets = []
    for i, gamma in enumerate(gammas, 1):
        w = 1 / m.double() if gamma == 1 else (1 - gamma) / (1 - gamma ** m.double())
        targets.append((y[:, i] - w[:, None] * y[:, 0]) / (1 - w[:, None]))
    return x, torch.stack(targets, 1).float(), g


def evaluate(model, layer, train, test, predict_all):
    x, y, g = train
    xt, yt, gt = test
    preds, lams = predict_all(x, y, g, xt)
    for i, gamma in enumerate(gammas):
        pred = preds[i]
        scores = paired_prompt_r2(
            yt[:, i],
            {"ILS": pred, "training_mean": y[:, i].mean(0).expand_as(pred)},
            gt,
            archive,
            f"{model}/{layer}/{i}",
        )
        out.append(
            dict(
                model=model,
                layer=layer,
                gamma=gamma,
                penalty=lams[i],
                train=len(x),
                test=len(xt),
                test_prompts=len(gt.unique()),
                ridge=scores["ILS"],
                training_mean=scores["training_mean"],
                gain=scores["ILS minus training_mean"],
            )
        )


def primary_pred(x, y, g, xt):
    maps, lams = cv_ridge(x, y, g)
    return [xt @ a.T for a in maps], lams


def replica_pred(x, y, g, xt):
    pairs = [
        predict(
            x.double().numpy(), y[:, i].double().numpy(), xt.double().numpy(), False
        )
        for i in range(3)
    ]
    return [torch.from_numpy(v) for v, _ in pairs], [lam for _, lam in pairs]


if __name__ == "__main__":
    device = resolve_device("cuda")
    for layer in range(23):
        train = extract(samples("train", layer, device), 128)
        test = extract(samples("validation", layer, device), 128)
        evaluate("0.5b", layer, train, test, primary_pred)
    root = Path("runs/internal-replication")
    rows = json.loads((root / "rows.json").read_text())
    for model in ["4b", "9b"]:
        a = torch.from_numpy(
            np.stack(
                [
                    np.fromfile(root / model / f"{i}.f32", "<f4").reshape(5, 5, -1)
                    for i in range(192)
                ]
            )
        )
        t = torch.tensor([r["t"] for r in rows])
        groups = torch.arange(192)
        for i, layer in enumerate([7, 15, 23, 30]):
            data = (a[:, i, 0], a[:, i, 1:], t, groups)
            train = extract([v[:128] for v in data], 64)
            test = extract([v[128:] for v in data], 64)
            evaluate(model, layer, train, test, replica_pred)
    np.savez_compressed("artifacts/qwen-future-sufficient.npz", **archive)
    write_json(
        "artifacts/qwen-future-only.json",
        dict(
            rows=out,
            scope="Exploratory strictly future residual R2; zero-prior ridge, same fitting-fold procedure, paired prompt CIs.",
        ),
    )
