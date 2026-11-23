"""Estimate held-out predictive coverage beyond the known direct residual contribution."""

import numpy as np
from pathlib import Path
import torch
from ilslens.scalar_transport import scalar_transport
from scripts.predictive_coverage import samples, cv_ridge, summarize, config, gammas
from ilslens.experiment import write_json


def weights(sites):
    length = config["sequence_length"] - 1 - sites
    return torch.stack(
        [1 / length if g == 1 else (1 - g) / (1 - g**length) for g in gammas], dim=1
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    rows = []
    archive = {}
    for layer in range(23):
        x, y, sites, groups = samples("train", layer, "cpu")
        xt, yt, st, gt = samples("validation", layer, "cpu")
        residual = y - x[:, None] * weights(sites)[:, :, None]
        skip = xt[:, None] * weights(st)[:, :, None]
        maps, lams = cv_ridge(x, residual, groups)
        for i, g in enumerate(gammas):
            pred = xt @ maps[i].T
            # summarize() expects the distinguished model under its 'ILS' key.
            scalar = scalar_transport(
                x, y[:, i], weights(sites)[:, i], xt, weights(st)[:, i]
            )
            metrics = summarize(
                yt[:, i],
                {
                    "ILS": skip[:, i] + pred,
                    "direct": skip[:, i],
                    "scalar": torch.from_numpy(scalar),
                },
                gt,
                archive,
                f"{layer}/{i}/total",
            )
            remainder = summarize(
                yt[:, i] - skip[:, i],
                {"ILS": pred},
                gt,
                archive,
                f"{layer}/{i}/remainder",
            )["ILS"]
            rows.append(
                dict(
                    layer=layer,
                    gamma=g,
                    penalty=lams[i],
                    total=metrics["ILS"],
                    gain_over_direct=metrics["ILS minus direct"],
                    scalar=metrics["scalar"],
                    gain_over_scalar=metrics["ILS minus scalar"],
                    remainder=remainder,
                )
            )
        print("completed block", layer, flush=True)
    np.savez_compressed("artifacts/qwen-innovation-sufficient.npz", **archive)
    write_json(
        "artifacts/qwen-coverage-innovation.json",
        dict(
            rows=rows,
            final_block="Remainder identically zero in exact arithmetic; R2 undefined.",
        ),
    )
