"""Context-dependent pair interactions in complete eight-head intervention tables."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


PAIRS = tuple(combinations(range(8), 2))
MATERIAL = 0.01  # Logits; the threshold used in the preceding conditional-head diagnostic.
MASKS = np.arange(256)
EMPTY = np.stack([MASKS[(MASKS & ((1 << i) | (1 << j))) == 0] for i, j in PAIRS])
I = EMPTY | np.array([1 << i for i, _ in PAIRS])[:, None]
J = EMPTY | np.array([1 << j for _, j in PAIRS])[:, None]
IJ = I | J


def read(path):
    return json.loads(path.read_text())


def interactions(effects, tops, donor):
    """Dimensions are prompt, unordered pair, and the 64 settings of the other six heads."""
    if effects.ndim != 2 or effects.shape[-1] != 256 or tops.shape != effects.shape:
        raise ValueError("each prompt requires all 256 native effects and top tokens")
    if not np.isfinite(effects).all():
        raise ValueError("native effects must be finite")
    baseline, i, j, both = (effects[:, mask] for mask in (EMPTY, I, J, IJ))
    contrast = (both - j) - (i - baseline)

    def reversals(before, after):
        return ((before * after < 0) & (np.abs(before) >= MATERIAL) & (np.abs(after) >= MATERIAL)).mean(-1)

    donor_chosen = (tops == donor[:, None]).astype(np.int8)
    b0, bi, bj, bij = (donor_chosen[:, mask] for mask in (EMPTY, I, J, IJ))
    # A nonzero second difference in the donor-token indicator means that adding
    # one head has a different behavioral effect when its partner is present.
    return {
        "mean_abs_interaction": np.abs(contrast).mean(-1),
        "mean_signed_interaction": contrast.mean(-1),
        "positive_material_fraction": (contrast >= MATERIAL).mean(-1),
        "negative_material_fraction": (contrast <= -MATERIAL).mean(-1),
        "material_margin_reversal_fraction": (
            reversals(i - baseline, both - j) + reversals(j - baseline, both - i)
        )
        * 0.5,
        "donor_switch_depends_on_partner_fraction": ((bij - bj) != (bi - b0)).mean(-1),
    }


def additions(effects, tops, donor):
    """All eight possible added heads, uniformly over 128 backgrounds each."""
    before = np.stack([MASKS[(MASKS & (1 << h)) == 0] for h in range(8)])
    after = before | (1 << np.arange(8))[:, None]
    delta = effects[:, after] - effects[:, before]
    was_donor = tops[:, before] == donor[:, None, None]
    now_donor = tops[:, after] == donor[:, None, None]
    return dict(
        material_increase_fraction=float((delta >= MATERIAL).mean()),
        material_decrease_fraction=float((delta <= -MATERIAL).mean()),
        switches_to_donor_fraction=float((~was_donor & now_donor).mean()),
        switches_away_from_donor_fraction=float((was_donor & ~now_donor).mean()),
        any_top_token_change_fraction=float((tops[:, before] != tops[:, after]).mean()),
    )


def native_table(directory, model, task, manifest):
    content = read(directory / f"primary-{model}-{task}.json")
    sites = content["sites"]
    rows = content["rows"]
    if sites != read(directory / f"panel-{model}.json")["sites"]:
        raise ValueError("result panel differs from the declared model panel")
    records = [(r["file"], r["index"], r["family"]) for r in rows]
    if records != [(r["file"], r["index"], r["family"]) for r in manifest[task]]:
        raise ValueError("result rows differ from manifest")
    if len(set(records)) != len(records) or len(sites) != 8 or any(r["sites"] != sites for r in rows):
        raise ValueError("duplicate prompt or inconsistent sites")
    return (
        sites,
        np.array([r["gold"] for r in rows]),
        np.array([r["top"] for r in rows]),
        np.array([r["labels"][1] for r in rows]),
    )


def summarize_task(sites, effects, tops, donor):
    measurements = interactions(effects, tops, donor)
    magnitude = measurements["mean_abs_interaction"]
    # Each prompt awards one vote, divided uniformly among exact ties.
    strongest = magnitude == magnitude.max(axis=1, keepdims=True)
    votes = strongest / strongest.sum(axis=1, keepdims=True)
    same = np.array([sites[i][0] == sites[j][0] for i, j in PAIRS])
    ranking = sorted(range(len(PAIRS)), key=lambda t: (-votes[:, t].sum(), -magnitude[:, t].mean(), t))
    pair_rows = [
        dict(
            bits=PAIRS[t],
            sites=[sites[h] for h in PAIRS[t]],
            prompt_top_fraction=float(votes[:, t].mean()),
            **{k: float(v[:, t].mean()) for k, v in measurements.items()},
        )
        for t in range(len(PAIRS))
    ]
    by_layer = {
        label: dict(pairs=int(mask.sum()), **{k: float(v[:, mask].mean()) for k, v in measurements.items()})
        for label, mask in (("within_block", same), ("across_blocks", ~same))
    }
    return (
        dict(
            prompts=len(effects),
            pairs=pair_rows,
            by_layer=by_layer,
            marginal_head_additions=additions(effects, tops, donor),
            hindsight_most_recurrent_pair=pair_rows[ranking[0]],
        ),
        measurements,
        ranking,
    )


def select_on_ioi(report, measurements, ranking):
    idx = ranking[0]
    signed = float(measurements["mean_signed_interaction"][:, idx].mean())
    direction = int(np.sign(signed))
    return idx, dict(**report["pairs"][idx], selected_sign=direction)


def heldout(report, measurements, ranking, idx, sign):
    n = report["prompts"]
    rng = np.random.default_rng(202944)
    draws = rng.integers(0, n, size=(10000, n))
    magnitude = measurements["mean_abs_interaction"]
    metrics = dict(
        abs_interaction_logits=magnitude[:, idx],
        signed_interaction_logits=measurements["mean_signed_interaction"][:, idx],
        advantage_over_other_27_abs_logits=magnitude[:, idx] - np.delete(magnitude, idx, axis=1).mean(1),
    )
    if sign:
        metrics["interaction_aligned_to_ioi_sign_logits"] = sign * metrics["signed_interaction_logits"]
        key = "positive_material_fraction" if sign > 0 else "negative_material_fraction"
        metrics["material_context_fraction_aligned_to_ioi_sign"] = measurements[key][:, idx]
    uncertainty = {}
    for key, values in metrics.items():
        boot = values[draws].mean(1)
        uncertainty[key] = dict(
            mean=float(values.mean()), pointwise_ci95=np.quantile(boot, [0.025, 0.975]).tolist()
        )
    return dict(
        **report["pairs"][idx],
        selected_ioi_sign=sign,
        colors_recurrence_rank=ranking.index(idx) + 1,
        uncertainty=uncertainty,
    )


def summarize(directory):
    manifest = read(directory / "manifest.json")

    def keys(rows):
        return {(r["file"], r["index"]) for r in rows}

    if keys(manifest["ioi"]) & keys(manifest["panel"] + manifest["development"] + manifest["colors"]):
        raise ValueError("IOI selection overlaps a panel, development or transfer example")
    out = dict(material_threshold_logits=MATERIAL, models={})
    for model in ("qwen", "large"):
        ioi, estimates, ranking = summarize_task(*native_table(directory, model, "ioi", manifest))
        idx, chosen = select_on_ioi(ioi, estimates, ranking)
        # IOI selection closes before loading the CopyColors outcomes.
        colors, estimates, ranking = summarize_task(*native_table(directory, model, "colors", manifest))
        out["models"][model] = dict(
            ioi=ioi,
            colors=colors,
            selected_on_ioi=chosen,
            selected_pair_on_colors=heldout(colors, estimates, ranking, idx, chosen["selected_sign"]),
        )
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.write_text(json.dumps(summarize(a.directory), indent=2, allow_nan=False) + "\n")
