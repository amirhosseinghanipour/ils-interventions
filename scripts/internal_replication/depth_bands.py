"""Simultaneous depth/horizon bands from the primary held-out prompt clusters."""

import argparse
import json
from pathlib import Path

import numpy as np


def depth_bands():
    # All 92 cells reuse the same validation prompts in ascending ID order.
    # One shared multinomial draw preserves their cross-cell dependence.
    gammas = (0, 0.5, 0.9, 1)
    coordinates = [(layer, i) for layer in range(23) for i in range(4)]
    trajectories = {"ILS": [], "gain_over_direct": []}
    with np.load("artifacts/qwen-primary-sufficient.npz") as archive:
        sizes = archive["0/0/total/n"]
        n = len(sizes)
        counts = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=10000)
        counts = np.vstack((np.ones(n), counts))
        for layer, horizon in coordinates:
            prefix = f"{layer}/{horizon}/total/"
            np.testing.assert_array_equal(archive[prefix + "n"], sizes)
            gram = archive[prefix + "sum_y_gram"]
            tss = counts @ archive[prefix + "sum_y2"]
            tss -= np.einsum("ij,ij->i", counts @ gram, counts) / (counts @ sizes)
            assert np.all(tss > 0)
            ils = counts @ archive[prefix + "SSE/ILS"]
            direct = counts @ archive[prefix + "SSE/horizon_scaled_identity"]
            trajectories["ILS"].append(1 - ils / tss)
            trajectories["gain_over_direct"].append((direct - ils) / tss)

    results = {}
    for name, trajectories_for_method in trajectories.items():
        samples = np.stack(trajectories_for_method, axis=1)
        estimates = samples[0]
        bootstrap = samples[1:]
        scales = bootstrap.std(axis=0, ddof=1)
        assert np.all(scales > 0)
        maximum = np.max(np.abs((bootstrap - estimates) / scales), axis=1)
        critical = float(np.quantile(maximum, 0.95))
        lower, upper = estimates - critical * scales, estimates + critical * scales
        rows = [
            dict(
                layer=layer,
                gamma=gammas[i],
                estimate=float(point),
                band=[float(lo), float(hi)],
            )
            for (layer, i), point, lo, hi in zip(coordinates, estimates, lower, upper)
        ]
        first = [
            next(
                (r["layer"] for r in rows if r["gamma"] == g and r["band"][0] > 0), None
            )
            for g in gammas
        ]
        results[name] = dict(critical=critical, first_positive=first, rows=rows)
    return dict(
        gammas=gammas,
        draws=10000,
        seed=0,
        method="Approximate two-sided 95% bootstrap maximum absolute standardized deviation over the 92 pre-final cells, separately for each reported quantity. Conditional on fitted maps.",
        results=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = Path("artifacts/qwen-depth-bands.json")
    summary = json.loads(json.dumps(depth_bands()))
    if args.verify:
        previous = json.loads(path.read_text())
        assert (
            previous["method"] == summary["method"]
            and previous["gammas"] == summary["gammas"]
        )
        assert (previous["draws"], previous["seed"]) == (
            summary["draws"],
            summary["seed"],
        )
        for name, computed in summary["results"].items():
            reported = previous["results"][name]
            assert reported["first_positive"] == computed["first_positive"]
            np.testing.assert_allclose(reported["critical"], computed["critical"])
            assert len(reported["rows"]) == len(computed["rows"])
            for old, new in zip(reported["rows"], computed["rows"]):
                assert (old["layer"], old["gamma"]) == (new["layer"], new["gamma"])
                np.testing.assert_allclose(
                    [old["estimate"], *old["band"]], [new["estimate"], *new["band"]]
                )
    else:
        path.write_text(json.dumps(summary, indent=2) + "\n")
    for name, result in summary["results"].items():
        print(name, result["first_positive"])
