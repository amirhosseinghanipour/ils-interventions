"""Post hoc matched-target controls at the registered primary epsilon."""

import json
from pathlib import Path
import numpy as np
import torch
from ilslens.experiment import write_json
from ilslens.fit import ridge
from ilslens.eval.faithfulness import paired_prompt_r2 as summarize
from ilslens.runtime import resolve_device

root = Path("runs/qwen-proposal-v4")
manifest = json.loads((root / "manifest.json").read_text())
config = manifest["config"]
epsilon = 0.05

gammas = config["gammas"]


def samples(split, layer, device):
    values = []
    for p in sorted((root / "samples" / split).glob(f"prompt-*-layer-{layer:02}.pt")):
        c = torch.load(p, mmap=True, weights_only=True)
        meta = c["metadata"]
        assert (
            meta["fingerprint"] == manifest["fingerprint"]
            and meta["split"] == split
            and meta["layer"] == layer
        )
        assert meta["gamma"] == gammas and meta["epsilon"] == config["epsilons"]
        e = meta["epsilon"].index(epsilon)
        values.append(
            (
                c["delta"][:, e].clone(),
                c["response"][:, e].clone(),
                c["source_positions"],
                c["prompt_ids"],
            )
        )
    n = config["source_prompts" if split == "train" else "heldout_prompts"]
    assert len(values) == n
    return [torch.cat(col).to(device) for col in zip(*values)]


def cv_ridge(x, y, groups):
    # Share covariance and solves across horizon outputs, select penalties independently.
    lambdas = config["lambda_candidates"]
    target = y.flatten(1)
    zero = x.new_zeros(target.shape[1], x.shape[1])
    groups_cpu = groups.cpu().unique()
    order = torch.randperm(
        len(groups_cpu), generator=torch.Generator().manual_seed(config["seed"])
    )
    costs = x.new_zeros(len(lambdas), y.shape[1], dtype=torch.float64)
    for fold in torch.tensor_split(groups_cpu[order], 5):
        held = torch.isin(groups, fold.to(x.device))
        for i, lam in enumerate(lambdas):
            a = ridge(x[~held], target[~held], lam, zero)
            pred = (x[held] @ a.T).reshape_as(y[held])
            costs[i] += (pred - y[held]).double().square().sum((0, 2))
    penalties = [lambdas[i] for i in costs.argmin(0).tolist()]
    return [
        ridge(x, y[:, i], lam, zero[: x.shape[1]]) for i, lam in enumerate(penalties)
    ], penalties


def main():
    device = resolve_device("cuda")
    rows = []
    archive = {}
    for layer in range(24):
        x, y, _, groups = samples("train", layer, device)
        xt, yt, sites, test = samples("validation", layer, device)
        maps, penalties = cv_ridge(x, y, groups)
        for i, gamma in enumerate(gammas):
            fit = torch.load(
                root / f"fits/layer-{layer:02}-epsilon-{epsilon}-gamma-{gamma}.pt",
                weights_only=True,
            )
            assert fit["metadata"]["fingerprint"] == manifest["fingerprint"]
            length = config["sequence_length"] - 1 - sites
            weight = 1 / length if gamma == 1 else (1 - gamma) / (1 - gamma**length)
            preds = {
                "ILS": xt @ fit["matrix"].to(device).T,
                "identity": xt,
                "horizon_scaled_identity": xt * weight[:, None],
                "zero_prior_ridge": xt @ maps[i].T,
            }
            measures = summarize(yt[:, i], preds, test, archive, f"{layer}/{i}/total")
            for key, old in [("identity", "I"), ("ILS", "ILS")]:
                assert np.isclose(
                    measures[key]["estimate"],
                    fit["heldout"][old]["r2"],
                    rtol=1e-4,
                    atol=2e-5,
                )
            rows.append(
                dict(layer=layer, gamma=gamma, penalty=penalties[i], metrics=measures)
            )
        print("verified block", layer, flush=True)
    np.savez_compressed("artifacts/qwen-primary-sufficient.npz", **archive)
    write_json(
        "artifacts/qwen-predictive-coverage.json",
        dict(
            rows=rows,
            epsilon=epsilon,
            primary_fingerprint=manifest["fingerprint"],
            bootstrap="2000 paired held-out source-prompt resamples, seed 0, pointwise intervals",
            scope="Post hoc controls. Zero-prior ridge uses identical training rows and prompt folds. Scaled identity predicts only the immediate residual skip contribution.",
        ),
    )


if __name__ == "__main__":
    main()
