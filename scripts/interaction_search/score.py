"""Re-score fixed-size interventions from independent exhaustive labels and paid timelines."""

import argparse
import json
from pathlib import Path

import numpy as np


# Rebuild the candidate universe independently of the proposal and oracle implementations.
SINGLES = tuple(1 << i for i in range(8))
TRIPLES = tuple(m for m in range(256) if m.bit_count() == 3)
PAIRS = tuple(m for m in range(256) if m.bit_count() == 2)
SCORED_ESTIMATORS = ("ils", "linear", "midpoint", "atp", "ig5")
METHODS = (*SCORED_ESTIMATORS, "guided", "greedy", "scan8", "scan32", "exhaustive")
FALLBACK = 7
PRIMARY = "large-colors"
RTOL, ATOL = 3e-4, 8e-4


def read(path):
    return json.loads(path.read_text())


def _vector(values, name):
    a = np.asarray(values, dtype=float)
    if a.shape != (256,) or not np.isfinite(a).all():
        raise ValueError(f"expected 256 finite {name}")
    return a


def _contrast(scores, masks):
    """Joint change minus the sum of constituent singleton changes, relative to the empty mask."""
    return np.array(
        [
            scores[m] - sum(scores[s] for s in SINGLES if s & m) + (m.bit_count() - 1) * scores[0]
            for m in masks
        ]
    )


def _timeline(method, gold, fallback):
    winner = method["winner"]
    if winner not in TRIPLES:
        raise ValueError(f"selected mask {winner} has the wrong number of heads")
    cap = float(method["cap_ms"])
    last = 0.0
    seen = set()
    measured = {FALLBACK: fallback}
    total = timely = 0
    max_error = 0.0
    for batch in method["paid"]:
        masks = list(batch["masks"])
        values = np.asarray(batch["values"], dtype=float)
        if values.shape != (len(masks),) or any(m not in range(256) for m in masks):
            raise ValueError("invalid paid masks or values")
        timestamp = float(batch["ms"])
        if not np.isfinite(timestamp) or timestamp < last:
            raise ValueError("invalid paid timeline")
        decision = float(batch.get("decision_ms", timestamp))
        if not np.isfinite(decision) or decision < timestamp:
            raise ValueError("decision precedes its exact measurements")
        role = batch["role"]
        if role not in ("fit", "verify"):
            raise ValueError("unknown measurement role")
        if role == "verify" and (
            "decision_ms" not in batch or any(m not in TRIPLES or m == FALLBACK for m in masks)
        ):
            raise ValueError("verification must time the final choice and contain only unpaid triples")
        if role == "fit" and any(m in TRIPLES for m in masks):
            raise ValueError("training exposed a target triple label")
        if len(set(masks)) != len(masks) or seen.intersection(masks):
            raise ValueError("duplicate paid mask")
        seen.update(masks)
        last = decision
        np.testing.assert_allclose(values, gold[masks], rtol=RTOL, atol=ATOL)
        max_error = max(max_error, float(np.max(abs(values - gold[masks]), initial=0.0)))
        total += len(masks)
        if decision <= cap:
            timely += len(masks)
            measured.update((m, v) for m, v in zip(masks, values, strict=True) if m in TRIPLES)
    if winner not in measured:
        raise ValueError(f"mask {winner} was unavailable by the deadline {cap}")
    if measured[winner] < max(measured.values()) - 1e-12:
        raise ValueError(f"winner {winner} is below a higher timely observed effect")
    if set(method["calibration_masks"]) != {
        m for b in method["paid"] if b["role"] == "fit" for m in b["masks"]
    }:
        raise ValueError("declared calibration masks differ from paid training queries")
    return total, timely, len(measured), max_error


def _bootstrap(matrix, families, seed, samples):
    """Resample whole prompt families and keep all policy comparisons paired in each draw."""
    labels = list(dict.fromkeys(families))
    sums = np.zeros((len(labels), matrix.shape[1]))
    sizes = np.zeros(len(labels))
    lookup = {f: i for i, f in enumerate(labels)}
    for f, v in zip(families, matrix, strict=True):
        sums[lookup[f]] += v
        sizes[lookup[f]] += 1
    center = matrix.mean(axis=0)
    if len(labels) < 2:
        return center, None
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=samples)
    boot = (counts @ sums) / (counts @ sizes)[:, None]
    sd = boot.std(axis=0, ddof=1)
    varying = sd > 1e-12 * np.maximum(1.0, abs(center))
    width = np.zeros_like(center)
    if np.any(varying):
        z = abs(boot[:, varying] - center[varying]) / sd[varying]
        width[varying] = np.quantile(z.max(axis=1), 0.95) * sd[varying]
    return center, np.column_stack((center - width, center + width))


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def score_cell(rows, seed=613723, samples=10000):
    if not rows:
        raise ValueError("no intervention records")
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("duplicate prompt indices")
    stats = {m: [] for m in METHODS}
    signals = []
    caps = []
    max_error = 0.0
    all_paid = 0
    targets = list(TRIPLES)
    for row in rows:
        y = _vector(row["gold"], "exact effects")
        top = np.asarray(row["top"])
        if top.shape != (256,) or set(row["methods"]) != set(METHODS):
            raise ValueError("incomplete vocabulary labels or method set")
        fallback = row["fallback_value"]
        np.testing.assert_allclose(fallback, y[FALLBACK], rtol=RTOL, atol=ATOL)
        original, donor = row["labels"]
        if original == donor:
            raise ValueError("identical donor and original label")
        clean_correct = row["clean_top"] == original
        attainable = bool(np.any(top[targets] == donor))
        best = float(y[targets].max())
        calibration = set(row["methods"]["ils"]["calibration_masks"])
        if calibration != set(row["methods"]["linear"]["calibration_masks"]) or calibration & set(targets):
            raise ValueError("ILS and the additive fit must use equal, size-three-disjoint calibration")
        # Both constituents and their union are unseen by either fitted model for this contrast.
        held_pairs = [
            m for m in PAIRS if m not in calibration and all(s not in calibration for s in SINGLES if m & s)
        ]
        coalitions = [
            (a, b)
            for a in PAIRS
            for b in PAIRS
            if a < b and not (a & b) and all(mask not in calibration for mask in (a, b, a | b))
        ]
        if not held_pairs or not coalitions:
            raise ValueError("no fully held-out pair contrasts")
        pair_gold = _contrast(y, held_pairs)
        left, right = np.asarray(coalitions).T

        def coalition_contrast(v):
            return v[left | right] - v[left] - v[right] + v[0]

        coalition_gold = coalition_contrast(y)
        triple_gold = _contrast(y, TRIPLES)
        across = np.array([(m & 15) != 0 and (m & 240) != 0 for m in held_pairs])
        signals.append(
            dict(
                clean_correct=clean_correct,
                donor_correct=row["donor_top"] == donor,
                switchable=clean_correct and attainable,
                oracle_max=best,
                oracle_min=float(y[targets].min()),
                held_pairs=len(held_pairs),
                coalition_abs=float(abs(coalition_gold).mean()),
                held_coalitions=len(coalitions),
                pair_abs=float(abs(pair_gold).mean()),
                triple_abs=float(abs(triple_gold).mean()),
            )
        )
        row_caps = {float(p["cap_ms"]) for p in row["methods"].values()}
        if len(row_caps) != 1 or not np.isfinite(next(iter(row_caps))) or min(row_caps) <= 0:
            raise ValueError("methods must share one positive finite deadline")
        caps.append(next(iter(row_caps)))
        for name in METHODS:
            method = row["methods"][name]
            total, timely, verified, error = _timeline(method, y, fallback)
            max_error = max(max_error, error)
            all_paid += total
            w = method["winner"]
            s = dict(
                effect=float(y[w]),
                regret=best - float(y[w]),
                optimum=bool(y[w] == best),
                switches=int(clean_correct and top[w] == donor),
                verified=verified,
                paid=total,
                timely=timely,
                fallback=int(w == FALLBACK),
                initial_overrun=int(method["initial_ms"] > method["cap_ms"]),
            )
            if name in SCORED_ESTIMATORS:
                pred = _vector(method["scores"], f"{name} raw predictions")
                raw_mask = method["raw_mask"]
                if raw_mask not in TRIPLES or not np.isclose(
                    pred[raw_mask], pred[targets].max(), atol=1e-8, rtol=1e-7
                ):
                    raise ValueError("unverified recommendation must maximize raw triple scores")
                pair_err = abs(_contrast(pred, held_pairs) - pair_gold)
                s.update(
                    raw_mae=float(abs(pred[targets] - y[targets]).mean()),
                    unpaid_triple_mae=float(abs(pred[targets[1:]] - y[targets[1:]]).mean()),
                    coalition_mae=float(abs(coalition_contrast(pred) - coalition_gold).mean()),
                    raw_regret=best - float(y[raw_mask]),
                    pair_mae=float(pair_err.mean()),
                    cross_pair_mae=_mean(pair_err[across]),
                    within_pair_mae=_mean(pair_err[~across]),
                    triple_interaction_mae=float(abs(_contrast(pred, TRIPLES) - triple_gold).mean()),
                )
            stats[name].append(s)
    report = dict(
        prompts=len(rows),
        distinct_families=len({r["family"] for r in rows}),
        candidate_triples=len(TRIPLES),
        clean_correct=sum(s["clean_correct"] for s in signals),
        donor_correct=sum(s["donor_correct"] for s in signals),
        clean_correct_switchable=sum(s["switchable"] for s in signals),
        oracle_max_mean=_mean([s["oracle_max"] for s in signals]),
        oracle_min_mean=_mean([s["oracle_min"] for s in signals]),
        full_holdout_pair_contrasts=sum(s["held_pairs"] for s in signals),
        full_holdout_two_plus_two_contrasts=sum(s["held_coalitions"] for s in signals),
        zero_interaction_coalition_mae=_mean([s["coalition_abs"] for s in signals]),
        zero_interaction_pair_mae=_mean([s["pair_abs"] for s in signals]),
        zero_interaction_triple_mae=_mean([s["triple_abs"] for s in signals]),
        audited_paid_measurements=all_paid,
        maximum_paid_absolute_disagreement=max_error,
        deadline_ms_range=[min(caps), max(caps)],
        methods={},
    )
    for name, per_prompt in stats.items():
        d = {key: _mean([r[key] for r in per_prompt if r[key] is not None]) for key in per_prompt[0]}
        d["switches"] = sum(r["switches"] for r in per_prompt)
        report["methods"][name] = d
    others = [m for m in METHODS if m != "ils"]
    delta = np.asarray(
        [[a["effect"] - b["effect"] for a, b in zip(stats["ils"], stats[m], strict=True)] for m in others]
    ).T
    center, bounds = _bootstrap(delta, [r["family"] for r in rows], seed, samples)
    report["paired_comparisons"] = {
        m: dict(
            ils_effect_advantage=float(center[j]),
            simultaneous_ci95=bounds[j].tolist() if bounds is not None else None,
            prompts_better=int((delta[:, j] > 0).sum()),
            prompts_equal=int((delta[:, j] == 0).sum()),
            prompts_worse=int((delta[:, j] < 0).sum()),
        )
        for j, m in enumerate(others)
    }
    report["passes_all"] = bool(bounds is not None and np.all(bounds[:, 0] > 0))
    return report


def summarize(directory):
    cells = {}
    for task in ("ioi", "colors"):
        reference = None
        for model in ("qwen", "large"):
            rows = read(directory / f"primary-{model}-{task}.json")["rows"]
            refs = [(r["index"], r["family"]) for r in rows]
            if reference is not None and reference != refs:
                raise ValueError("checkpoints evaluated different task prompts or ordering")
            reference = refs
            cells[f"{model}-{task}"] = score_cell(rows)
    return dict(primary=PRIMARY, passes_primary=cells[PRIMARY]["passes_all"], cells=cells)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.write_text(json.dumps(summarize(args.directory), indent=2, allow_nan=False) + "\n")
