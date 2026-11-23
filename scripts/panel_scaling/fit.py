"""Midpoint residual regression for a sixteen-head panel."""

from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HEADS = 16
ALL = (1 << HEADS) - 1
RIDGE = 0.03
MAX_PAIRS = 8
PAIRS = np.array(list(combinations(range(HEADS), 2)))
SHIFTS = np.arange(HEADS, dtype=np.int64)


def design(seed, pairs=64):
    """Draw complement pairs for calibration, then 512 distinct unseen queries.

    Each pair has a representative with the high bit clear. Random orientation makes
    every calibration position marginally uniform over the interior of the cube.
    """
    rng = np.random.default_rng(seed)
    representatives = 1 + rng.choice(ALL // 2, size=pairs, replace=False)
    first = representatives ^ (ALL * rng.integers(0, 2, size=pairs))
    train = np.column_stack((first, ALL ^ first)).ravel()
    available = np.ones(ALL + 1, dtype=bool)
    available[[0, ALL]] = False
    available[train] = False
    query = rng.choice(np.flatnonzero(available), size=512, replace=False)
    return train.tolist(), query.tolist()


def _bits(masks):
    ids = np.asarray(masks)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError("masks must be a vector of integers")
    if np.any(ids < 0) or np.any(ids > ALL):
        raise ValueError("mask outside the sixteen-head panel")
    return ((ids.astype(np.int64)[:, None] >> SHIFTS) & 1).astype(np.float64)


def _pairs(bits, pairs):
    # On Boolean gates this is m_i*m_j - (m_i + m_j)/2, the centered pair basis.
    return -0.5 * (bits[:, pairs[:, 0]] != bits[:, pairs[:, 1]])


def _normalize(x):
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = 1
    return x / norm, norm


def _solve(matrix, rhs):
    return cho_solve(cho_factor(matrix, check_finite=False), rhs, check_finite=False)


def _ridge(x, residual):
    if x.shape[0] < x.shape[1]:
        kernel = x @ x.T
        kernel.flat[:: kernel.shape[0] + 1] += RIDGE
        return x.T @ _solve(kernel, residual)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += RIDGE
    return _solve(gram, x.T @ residual)


def _select_pairs(x, residual):
    """Forward selection by exact reduction in the penalized objective.

    Eliminate the required main effects once. After each choice, a rank-one update
    gives the conditional curvature and linear term for every remaining pair.
    """
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += RIDGE
    target = x.T @ residual
    cross = gram[:HEADS, HEADS:]
    factor = cho_factor(gram[:HEADS, :HEADS], check_finite=False)
    curvature = gram[HEADS:, HEADS:] - cross.T @ cho_solve(factor, cross, check_finite=False)
    linear = target[HEADS:] - cross.T @ cho_solve(factor, target[:HEADS], check_finite=False)
    chosen = []
    for _ in range(MAX_PAIRS):
        gain = linear**2 / curvature.diagonal()
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        column = curvature[:, j].copy()
        diagonal = column[j]
        # Removing the pivot row/column is unnecessary as gains there are masked.
        linear -= column * (linear[j] / diagonal)
        curvature -= np.outer(column, column) / diagonal
        curvature[j, j] = 1.0
    return np.array(chosen)


def fit(method, gradient, masks, y, target_masks):
    """Predict signed margin changes for queries, fitting solely to (masks, y).

    All non-midpoint models use the same unit-norm-column ridge penalty. The
    sparse fit retains eight pair terms and the dense fit retains all 120.
    """
    if method not in ("midpoint", "linear", "ils", "dense"):
        raise ValueError(f"unknown method {method!r}")
    g = np.asarray(gradient, dtype=float)
    if g.shape != (HEADS,) or not np.isfinite(g).all():
        raise ValueError("expected sixteen finite midpoint derivatives")
    query = _bits(target_masks)
    base = query @ g
    if method == "midpoint":
        return base
    train = _bits(masks)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(train),) or not len(train) or not np.isfinite(y).all():
        raise ValueError("expected a finite calibration effect for each training mask")
    residual = y - train @ g
    if method == "linear":
        x, norm = _normalize(train)
        return base + query @ (_ridge(x, residual) / norm)

    x, norm = _normalize(np.column_stack((train, _pairs(train, PAIRS))))
    pairs = PAIRS
    if method == "ils":
        sel = _select_pairs(x, residual)
        columns = np.r_[np.arange(HEADS), HEADS + sel]
        x, norm, pairs = x[:, columns], norm[columns], PAIRS[sel]

    coef = _ridge(x, residual) / norm
    return base + query @ coef[:HEADS] + _pairs(query, pairs) @ coef[HEADS:]
