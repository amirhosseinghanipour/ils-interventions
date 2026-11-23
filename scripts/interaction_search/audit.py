"""Check the sample freeze, calibration queries and raw pair-fit advantages."""

import argparse
import json
from pathlib import Path

import numpy as np

from .policies import PAIR_GRID, TRIPLES, calibration_masks, propose
from .score import _bootstrap, score_cell


def read(p):
    return json.loads(p.read_text())


def verify(directory):
    manifest = read(directory / "manifest.json")
    dev = read(directory / "development.json")["rows"]
    frozen = read(directory / "frozen.json")
    assert [r["index"] for r in dev] == [r["index"] for r in manifest["development"]]
    assert set(r["family"] for r in manifest["ioi"]).isdisjoint(r["family"] for r in dev)
    upstream = manifest["panel"] + manifest["development"] + manifest["ioi"]
    assert len({r["index"] for r in upstream}) == len(upstream)
    loss = {k: [np.mean([r["grid"][str(k)][m] for r in dev]) for m in ("regret", "mae")] for k in PAIR_GRID}
    assert min(PAIR_GRID, key=lambda k: (*loss[k], k)) == frozen["max_pairs"]
    for k, v in loss.items():
        np.testing.assert_allclose(frozen["development_regret_mae"][str(k)], v)
    result = dict(
        development_prompts=len(dev),
        heldout_prompt_checkpoint_pairs=0,
        exhaustive_heldout_masks=0,
        heldout_triples=0,
        paid_effects_audited=0,
        cells={},
    )
    summary = read(directory / "summary.json")
    for model in ("qwen", "large"):
        sites = read(directory / ("panel-" + model + ".json"))["sites"]
        assert len(set(map(tuple, sites))) == 8 and [s[0] for s in sites] == [11] * 4 + [17] * 4
        for task in ("ioi", "colors"):
            key = model + "-" + task
            source = read(directory / ("primary-" + key + ".json"))
            assert source["sites"] == sites
            rows = source["rows"]
            assert [{k: r[k] for k in ("file", "index", "family")} for r in rows] == manifest[task]
            raw_contrasts = []
            max_native = 0.0
            for n, r in enumerate(rows):
                ys = np.array(r["gold"])
                assert len(ys) == len(r["top"]) == 256 and abs(ys[0]) < 0.0008 and r["sites"] == sites
                max_native = max(max_native, r["native_audit_max_abs"])
                targets = list(TRIPLES)
                seed = 225177 + n
                raw = {}
                for method in ("ils", "linear"):
                    run = r["methods"][method]
                    train = calibration_masks(seed)
                    assert tuple(run["calibration_masks"]) == train and set(train).isdisjoint(targets)
                    fit = [b for b in run["paid"] if b["role"] == "fit"]
                    assert len(fit) == 1 and tuple(fit[0]["masks"]) == train

                    def measured(ids):
                        assert tuple(ids) == train
                        return fit[0]["values"]

                    expect = propose(
                        method,
                        {"midpoint": r["midpoint"]},
                        measured,
                        seed=seed,
                        max_pairs=frozen["max_pairs"],
                    )
                    np.testing.assert_allclose(run["scores"], expect.scores, rtol=1e-5, atol=1e-5)
                    raw[method] = np.array(run["scores"])
                # Raw predictions are fixed before receiving any three-head verification labels.
                linear, ils = (raw[m][targets] for m in ("linear", "ils"))
                raw_contrasts.append(
                    [
                        np.mean(abs(linear - ys[targets]) - abs(ils - ys[targets])),
                        ys[r["methods"]["ils"]["raw_mask"]] - ys[r["methods"]["linear"]["raw_mask"]],
                    ]
                )
            calc = score_cell(rows)
            assert calc == summary["cells"][key]
            center, bounds = _bootstrap(np.array(raw_contrasts), [r["family"] for r in rows], 840165, 10000)
            result["cells"][key] = dict(
                prompts=len(rows),
                native_audit_max_abs=max_native,
                raw_ils_mae_advantage_vs_linear=float(center[0]),
                raw_ils_top_choice_advantage_vs_linear=float(center[1]),
                simultaneous_ci95_within_cell=bounds.tolist(),
            )
            result["heldout_prompt_checkpoint_pairs"] += len(rows)
            result["exhaustive_heldout_masks"] += len(rows) * 256
            result["heldout_triples"] += len(rows) * len(targets)
            result["paid_effects_audited"] += sum(
                len(b["masks"]) for r in rows for m in r["methods"].values() for b in m["paid"]
            )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(verify(a.directory), indent=2, allow_nan=False) + "\n")
