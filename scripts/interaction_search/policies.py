"""Eight-head search policies; only the measurement callback can reveal an exact patch effect."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np


HEADS = 8
ALL = 2**HEADS - 1
PAIR_GRID = (0, 1, 3, 6)
METHODS = ("midpoint", "atp", "ig5", "linear", "ils", "scan", "greedy")
PAIRS = tuple(combinations(range(HEADS), 2))
TRIPLES = tuple(m for m in range(ALL + 1) if m.bit_count() == 3)
BITS = ((np.arange(ALL + 1)[:, None] >> np.arange(HEADS)) & 1).astype(float)
# Each feature is zero at the unpatched state and has zero continuous gradient at 1/2.
# A unit coefficient contributes exactly one unit to the two-head interaction contrast.
INTERACTIONS = np.column_stack(
    [BITS[:, i] * BITS[:, j] - 0.5 * (BITS[:, i] + BITS[:, j]) for i, j in PAIRS]
)
RIDGE = 0.03  # Applied after scaling each measured feature column to unit L2 norm.


@dataclass
class Proposal:
    scores: np.ndarray
    ranking: tuple
    calibration_masks: tuple = ()
    selected_pairs: tuple = ()


def calibration_masks(seed, size=32):
    """Complement pairs keep every gate on for exactly half the fit labels."""
    if size not in (16, 32):
        raise ValueError("calibration size must be 16 or 32")
    rng = np.random.default_rng(seed)
    sets = [
        [m for m in range(1, ALL) if m.bit_count() == k and (k < 4 or m < (ALL ^ m))] for k in (1, 2, 4)
    ]
    for _ in range(100):
        half = []
        for pool, count in zip(sets, (size // 8, size // 8, size // 4), strict=True):
            half.extend(rng.choice(pool, count, replace=False).tolist())
        masks = tuple(int(v) for v in rng.permutation(half + [ALL ^ m for m in half]))
        if np.linalg.matrix_rank(BITS[list(masks)]) == HEADS:
            return masks
    raise ValueError("failed to draw a full-rank balanced calibration design")


def _rank(scores, masks=TRIPLES):
    return tuple(sorted(masks, key=lambda m: (-scores[m], m)))


def _measure(measure, masks):
    values = np.asarray(measure(list(masks)), dtype=float)
    if values.shape != (len(masks),) or not np.isfinite(values).all():
        raise ValueError("exact measurement must return one finite signed effect per mask, in order")
    return values


def _gradient_scores(gradients, key):
    grad = np.asarray(gradients[key], dtype=float)
    if grad.shape != (HEADS,) or not np.isfinite(grad).all():
        raise ValueError("expected eight finite gate derivatives")
    return BITS @ grad


def _fits(gradients, measure, seed, calibration_size, budgets):
    return fit_observations(gradients, measure, calibration_masks(seed, calibration_size), budgets)


def fit_observations(gradients, measure, masks, budgets):
    """Fit requested supports to the supplied calibration design via one paid callback."""
    indices = list(masks)
    base = _gradient_scores(gradients, "midpoint")
    residual = _measure(measure, masks) - base[indices]
    full = np.column_stack((BITS, INTERACTIONS))
    norm = np.linalg.norm(full[indices], axis=0)
    full = full / np.where(norm > 0, norm, 1.0)
    fit = full[indices]
    gram = fit.T @ fit + RIDGE * np.eye(fit.shape[1])
    target = fit.T @ residual
    selected = list(range(HEADS))
    available = list(range(HEADS, fit.shape[1]))
    proposals = {}
    for n in range(max(budgets) + 1):
        system = gram[np.ix_(selected, selected)]
        coef = np.linalg.solve(system, target[selected])
        if n in budgets:
            scores = base + full[:, selected] @ coef
            proposals[n] = Proposal(
                scores, _rank(scores), masks, tuple(PAIRS[k - HEADS] for k in selected[HEADS:])
            )
        if n < max(budgets):
            # Exact decrease in penalized least-squares loss for each one-feature extension.
            cross = gram[np.ix_(selected, available)]
            curvature = gram.diagonal()[available] - (cross * np.linalg.solve(system, cross)).sum(axis=0)
            linear = target[available] - coef @ cross
            gains = np.square(linear) / curvature
            selected.append(available.pop(int(np.argmax(gains))))
    return proposals


def propose_grid(gradients, measure, *, seed, calibration_size=32):
    """Predict with all support sizes using the same paid measurements. Intended for development."""
    return _fits(gradients, measure, seed, calibration_size, PAIR_GRID)


def propose(method, gradients, measure, *, seed, max_pairs=3, calibration_size=32):
    """Return scores indexed by a bit mask and a proposed verification order of distinct triples.

    Head zero is the least significant bit. All policies aim to maximize the donor-minus-original
    margin change. The callback takes a list of integer masks and returns aligned signed effects.
    No size-three masks are measured inside this function. The caller separately times and pays
    for verification. Scan and greedy need no gradients. Their scores are search heuristics.
    """
    if method in ("midpoint", "atp", "ig5"):
        scores = _gradient_scores(gradients, method)
        return Proposal(scores, _rank(scores))
    if method in ("linear", "ils"):
        budget = 0 if method == "linear" else max_pairs
        if budget not in PAIR_GRID:
            raise ValueError(f"pair budget must be in {PAIR_GRID}")
        return _fits(gradients, measure, seed, calibration_size, (budget,))[budget]
    if method == "scan":
        order = tuple(int(m) for m in np.random.default_rng(seed).permutation(TRIPLES))
        return Proposal(np.zeros(ALL + 1), order)
    if method != "greedy":
        raise ValueError(f"unknown policy {method!r}")

    singles = [1 << j for j in range(HEADS)]
    values = _measure(measure, singles)
    best_index = int(np.argmax(values))
    best = singles[best_index]
    others = [j for j in range(HEADS) if j != best_index]
    pairs = [best | singles[j] for j in others]
    pair_effects = _measure(measure, pairs)
    best_pair = pairs[int(np.argmax(pair_effects))]
    scores = BITS @ values
    # Estimate each partner's extra contribution conditional on the first selected head.
    for j, effect in zip(others, pair_effects, strict=True):
        scores += BITS[:, best_index] * BITS[:, j] * (effect - values[best_index] - values[j])
    ranked = _rank(scores)
    first = tuple(m for m in ranked if (m & best_pair) == best_pair)
    rest = tuple(m for m in ranked if (m & best_pair) != best_pair)
    return Proposal(scores, first + rest, tuple(singles + pairs))


def best_verified(ranking, verified):
    """Highest timely measured effect, resolving exact ties by proposal order."""
    choices = [m for m in ranking if m in verified]
    return max(choices, key=verified.__getitem__) if choices else ranking[0]
