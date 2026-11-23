"""Reconstruct every internal-effect table and interval from prompt-level sufficient statistics, without model weights."""

import json
from pathlib import Path
import numpy as np

artifacts = Path("artifacts")
gammas = [0, 0.5, 0.9, 1]


def read(name):
    return json.loads((artifacts / name).read_text())


def verify_archive(filename, expected):
    with np.load(artifacts / filename, allow_pickle=False) as data:
        cells = {k[:-2] for k in data if k.endswith("/n")}
        assert cells == set(expected), (filename, cells.symmetric_difference(expected))
        checked = 0
        for cell in sorted(cells):

            def a(field):
                return data[cell + "/" + field]

            n = a("n")
            gram = a("sum_y_gram")
            ss = a("sum_y2")
            counts = np.random.default_rng(0).multinomial(
                len(n), np.full(len(n), 1 / len(n)), 2000
            )
            counts = np.vstack([np.ones(len(n)), counts])
            den = counts @ ss - np.einsum("ij,ij->i", counts @ gram, counts) / (
                counts @ n
            )
            assert (den > 0).all()
            prefix = cell + "/SSE/"
            estimates = {
                k[len(prefix) :]: 1 - (counts @ data[k]) / den
                for k in data
                if k.startswith(prefix)
            }
            for k in list(estimates):
                if k != "ILS":
                    estimates["ILS minus " + k] = estimates["ILS"] - estimates[k]
            for k, reported in expected[cell].items():
                v = estimates[k]
                np.testing.assert_allclose(
                    [v[0], *np.quantile(v[1:], [0.025, 0.975])],
                    [reported["estimate"], *reported["ci"]],
                    atol=1e-8,
                    rtol=1e-8,
                )
                checked += 1
    print(
        f"{filename}: {len(cells)} cells, {checked} estimates and paired or marginal intervals verified"
    )


def by_layer(rows, layers):
    keys = [(r["layer"], r["gamma"]) for r in rows]
    assert len(keys) == len(set(keys)) and set(keys) == {
        (l, g) for l in layers for g in gammas
    }
    for r in rows:
        yield f"{r['layer']}/{gammas.index(r['gamma'])}", r


def correction(r):
    return dict(
        ILS=r["total"],
        scalar=r["scalar"],
        **{
            "ILS minus direct": r["gain_over_direct"],
            "ILS minus scalar": r["gain_over_scalar"],
        },
    )


if __name__ == "__main__":
    cells = {
        k + "/total": r["metrics"]
        for k, r in by_layer(read("qwen-predictive-coverage.json")["rows"], range(24))
    }
    verify_archive("qwen-primary-sufficient.npz", cells)
    cells = {}
    for k, r in by_layer(read("qwen-coverage-innovation.json")["rows"], range(23)):
        cells[k + "/total"] = correction(r)
        cells[k + "/remainder"] = {"ILS": r["remainder"]}
    verify_archive("qwen-innovation-sufficient.npz", cells)
    cells = {}
    for k, r in by_layer(
        read("qwen-fresh-predictive-coverage.json")["rows"], [0, 12, 19, 22, 23]
    ):
        cells[k + "/total"] = dict(
            ILS=r["corrected"],
            scalar=r["scalar"],
            direct=r["direct"],
            original_ILS=r["original_ILS"],
            **{
                "ILS minus scalar": r["gain_over_scalar"],
                "ILS minus original_ILS": r["gain_over_original"],
                "ILS minus direct": r["gain_over_direct"],
            },
        )
        if r["layer"] != 23:
            cells[k + "/remainder"] = {"ILS": r["remainder"]}
        else:
            assert r["remainder"] is None
    verify_archive("qwen-fresh-sufficient.npz", cells)
    rep = read("qwen35-predictive-coverage.json")["results"]
    assert [s["model"] for s in rep] == ["4b", "9b"]
    for model in rep:
        cells = {}
        for k, r in by_layer(model["rows"], [7, 15, 23, 30, 31]):
            cells[k + "/total"] = r["metrics"]
            if r["layer"] != 31:
                cells[k + "/correction"] = correction(r["innovation"])
                cells[k + "/remainder"] = {"ILS": r["innovation"]["remainder"]}
            else:
                assert r["innovation"] is None
        verify_archive("qwen35-" + model["model"] + "-sufficient.npz", cells)
    cells = {}
    for r in read("qwen-future-only.json")["rows"]:
        k = f"{r['model']}/{r['layer']}/{gammas[1:].index(r['gamma'])}"
        assert k not in cells
        cells[k] = dict(
            ILS=r["ridge"],
            training_mean=r["training_mean"],
            **{"ILS minus training_mean": r["gain"]},
        )
    grid = {
        f"{model}/{layer}/{i}"
        for model, layers in [
            ("0.5b", range(23)),
            ("4b", [7, 15, 23, 30]),
            ("9b", [7, 15, 23, 30]),
        ]
        for layer in layers
        for i in range(3)
    }
    assert set(cells) == grid
    verify_archive("qwen-future-sufficient.npz", cells)
    transfer = read("qwen-transfer.json")
    assert (
        transfer["plan"]["layer"],
        transfer["plan"]["gamma"],
        transfer["plan"]["pairs"],
    ) == (22, 0.9, 64)
    verify_archive(
        "qwen-transfer-sufficient.npz",
        dict(total=transfer["metrics"], remainder=transfer["remainder"]),
    )

    # Compare the typeset penultimate-block tables with the unrounded JSON values.
    # The 12 numeric table rows are public evidence. A local paper can also be checked.
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--manuscript", type=Path)
    manuscript_path = parser.parse_args().manuscript
    if manuscript_path:
        manuscript = manuscript_path.read_text()
    else:
        rows = (artifacts / "depth-reference-rows.tsv").read_text().splitlines()
        manuscript = "\n".join("| " + " | ".join(row.split("\t")) + " |" for row in rows)

    def number(value, signed=False):
        return format(value, "+.3f" if signed else ".3f").replace("-", "−")

    def difference(metric):
        lo, hi = metric["ci"]
        return f"{number(metric['estimate'], True)} [{number(lo)}, {number(hi)}]"

    def check_row(fields):
        line = "| " + " | ".join(fields) + " |"
        assert manuscript.splitlines().count(line) == 1, line

    for row in read("qwen-predictive-coverage.json")["rows"]:
        if row["layer"] == 22:
            metrics = row["metrics"]
            check_row(
                [
                    f"{row['gamma']:g}",
                    number(metrics["ILS"]["estimate"]),
                    number(metrics["horizon_scaled_identity"]["estimate"]),
                    number(metrics["zero_prior_ridge"]["estimate"]),
                    difference(metrics["ILS minus horizon_scaled_identity"]),
                ]
            )
    for model in rep:
        for row in model["rows"]:
            if row["layer"] == 30:
                metrics, innovation = row["metrics"], row["innovation"]
                check_row(
                    [
                        model["model"].upper(),
                        f"{row['gamma']:g}",
                        number(metrics["ILS"]["estimate"]),
                        number(metrics["horizon_scaled_identity"]["estimate"]),
                        number(innovation["total"]["estimate"]),
                        difference(innovation["gain_over_scalar"]),
                    ]
                )
    print("All 12 reference rows match the numeric sources after rounding.")
