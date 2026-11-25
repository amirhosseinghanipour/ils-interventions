"""Prompt-local response models for real-valued donor-mask effects.

Masks are Boolean vectors, and gradient/Hessian entries are derivatives with
respect to continuous masks in [0,1]. Predictions use z = 2*m - 1.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def hvp_prior(directions, products):
    """Recover the symmetric sampled part of H while treating its diagonal as nuisance.

    For P projecting onto the probe span, S(H) = H P + P H - P H P
    contains exactly the identifiable part of a symmetric H. Diagonals have
    no varying Boolean effect. Find D diagonal that minimizes ||S(H-D)||_F
    before returning the remainder. A purely diagonal Hessian yields zero even
    when the probes span only a small fraction of the space.
    """
    v = np.asarray(directions, dtype=float).T
    w = np.asarray(products, dtype=float).T
    if v.ndim != 2 or w.shape != v.shape or not v.shape[1]:
        raise ValueError("expected matching, nonempty probe and product matrices")
    if not np.isfinite(v).all() or not np.isfinite(w).all():
        raise ValueError("nonfinite Hessian probe or product")
    inv = np.linalg.pinv(v)
    proj = v @ inv
    hp = w @ inv
    inside = proj @ hp
    sampled = hp + hp.T - (inside + inside.T) / 2
    diagonal_system = np.diag(2 * proj.diagonal()) - proj**2
    diag = np.linalg.lstsq(diagonal_system, sampled.diagonal(), rcond=None)[0]
    dp = diag[:, None] * proj
    off = sampled - dp - dp.T + proj @ dp
    np.fill_diagonal(off, 0)
    return (off + off.T) / 2


def _bits(masks, k):
    x = np.asarray(masks, dtype=float)
    if x.shape == (0,):
        x = x.reshape(0, k)
    if x.ndim != 2 or x.shape[1] != k or not ((x == 0) | (x == 1)).all():
        raise ValueError(f"expected Boolean masks with {k} columns")
    return x


@dataclass
class Polynomial:
    offset: float
    linear: np.ndarray
    quadratic: np.ndarray | None = None
    beta: float = 0.0

    def __call__(self, masks):
        z = 2 * _bits(masks, len(self.linear)) - 1
        out = self.offset + z @ self.linear
        if self.quadratic is not None:
            out += np.sum((z @ self.quadratic) * z, axis=1)
        return out


def _ridge(design, targets, penalty):
    if design.shape[1] <= design.shape[0]:
        gram = design.T @ design
        gram.flat[:: len(gram) + 1] += penalty
        return cho_solve(cho_factor(gram), design.T @ targets)
    gram = design @ design.T
    gram.flat[:: len(gram) + 1] += penalty
    return design.T @ cho_solve(cho_factor(gram), targets)


def _complement(x, y, midpoint, gradient, prior, ridge):
    if len(x) % 2 or not np.array_equal(x[::2], 1 - x[1::2]):
        raise ValueError("expected adjacent complement pairs")
    z = 2 * x[::2] - 1
    even = (y[::2] + y[1::2]) / 2
    odd = (y[::2] - y[1::2]) / 2
    linear = gradient / 2
    if not len(z):
        return Polynomial(midpoint, linear, None if prior is None else prior / 8,
                          float(prior is not None))
    linear = linear + _ridge(z, odd - z @ linear, ridge)

    # Gram of the unscaled upper-triangle Walsh pair features z_i*z_j.
    pair_gram = ((z @ z.T) ** 2 - z.shape[1]) / 2
    means = pair_gram.mean(axis=0)
    centered = pair_gram - means[:, None] - means[None, :] + means.mean()
    centered.flat[:: len(z) + 1] += ridge
    factor = cho_factor(centered)
    y_centered = even - even.mean()
    prior_vals = np.zeros(len(z)) if prior is None else np.sum((z @ prior) * z, axis=1) / 8
    h_centered = prior_vals - prior_vals.mean()
    solved_y = cho_solve(factor, y_centered)
    beta = 0.0
    if np.any(h_centered):
        solved_h = cho_solve(factor, h_centered)
        curvature = h_centered @ solved_h
        if curvature > 0:
            beta = float(np.clip((h_centered @ solved_y) / curvature, 0, 1))
            solved_y = solved_y - beta * solved_h
    # The dual weights of a centered ridge solve sum to zero.
    dual = solved_y - solved_y.mean()
    correction = (z.T * dual) @ z / 2
    np.fill_diagonal(correction, 0)
    offset = float(even.mean() - beta * prior_vals.mean() - means @ dual)
    if prior is not None:
        correction += beta * prior / 8
    return Polynomial(offset, linear, correction, beta)


def _dense(x, y, midpoint, gradient, ridge):
    """Use the original dense midpoint correction with unit-norm columns."""
    base = Polynomial(midpoint, gradient / 2)
    if not len(x):
        return base
    i, j = np.triu_indices(x.shape[1], k=1)
    pairs = -0.5 * (x[:, i] != x[:, j])
    design = np.column_stack((x, pairs))
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0] = 1
    coef = _ridge(design / scale, y - base(x), ridge) / scale
    main, pair = np.split(coef, [x.shape[1]])
    matrix = np.zeros((len(main), len(main)))
    matrix[i, j] = matrix[j, i] = pair / 8
    return Polynomial(midpoint + main.sum() / 2 - pair.sum() / 4,
                      (gradient + main) / 2, matrix)


def prepare_tree_backend():
    """Import the optional numerical backend once during process initialization."""
    global lgb
    import lightgbm as lgb
    return lgb.__version__


@dataclass
class TreePrediction:
    booster: object
    beta: float = 0.0

    def __call__(self, masks):
        return self.booster.predict(np.asarray(masks, dtype=float), num_threads=1)


@dataclass
class FourierPrediction:
    support: np.ndarray
    coefficients: np.ndarray
    beta: float = 0.0

    def __call__(self, masks):
        return _fourier_design(np.asarray(masks), self.support) @ self.coefficients


def _fourier_design(x, support):
    # (-1)^(sum m_i) on each subset; uint8 overflow preserves its parity.
    return 1.0 - 2.0 * ((x.astype(np.uint8) @ support.T) % 2)


def _fourier_terms(booster):
    """Expand LightGBM leaf indicators exactly on binary inputs.

    A path with fixed bits has an indicator equal to a product of factors
    (1 +/- (-1)^m_i)/2. Combine repeated paths before the Fourier expansion.
    """
    paths = {}

    def walk(node, path):
        if "split_index" not in node:
            key = tuple(sorted(path))
            paths[key] = paths.get(key, 0.0) + node["leaf_value"]
            return
        threshold = float(node["threshold"])
        if node["decision_type"] != "<=" or not 0 <= threshold < 1:
            raise ValueError("unexpected non-Boolean tree split")
        feature = node["split_feature"]
        walk(node["left_child"], (*path, (feature, 1)))
        walk(node["right_child"], (*path, (feature, -1)))

    for tree in booster.dump_model()["tree_info"]:
        walk(tree["tree_structure"], ())
    terms = {}
    for path, value in paths.items():
        coefficient = value / 2**len(path)
        for degree in range(len(path) + 1):
            for subset in combinations(path, degree):
                key = tuple(i for i, _ in subset)
                signed = coefficient
                for _, sign in subset:
                    signed *= sign
                terms[key] = terms.get(key, 0.0) + signed
    return terms


def _refine(booster, x, y):
    """ProxySPEX tree-support energy truncation followed by leave-one-out RidgeCV."""
    from sklearn.linear_model import RidgeCV

    terms = _fourier_terms(booster)
    nonconstant = sorted((key for key in terms if key), key=lambda key: -abs(terms[key]))
    refine = len(terms) > x.shape[1] and any(terms[key] for key in nonconstant)
    keys = list(terms)
    if refine:
        energy = np.cumsum([terms[key] ** 2 for key in nonconstant])
        count = np.searchsorted(energy, 0.95 * energy[-1]) + 1
        # Keep the intercept and resolve equal-energy ties consistently.
        threshold = abs(terms[nonconstant[count - 1]])
        keys = [key for key in keys if not key or abs(terms[key]) >= threshold]
    support = np.zeros((len(keys), x.shape[1]), dtype=np.uint8)
    for row, key in enumerate(keys):
        support[row, list(key)] = 1
    coefficients = np.asarray([terms[key] for key in keys])
    if refine:
        design = _fourier_design(x, support)
        model = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False)
        coefficients = model.fit(design, y).coef_
    return FourierPrediction(support, coefficients)


def _tree(x, y, seed, fourier):
    """Fixed shallow LightGBM from the paper's grid; no per-prompt HPO."""
    if not len(x):
        raise ValueError("tree proxy needs at least one exact observation")
    if "lgb" not in globals():
        prepare_tree_backend()
    config = dict(objective="regression", learning_rate=0.1, max_depth=3,
                  num_leaves=8, min_data_in_leaf=2, min_data_in_bin=1,
                  num_threads=1, deterministic=True, force_col_wise=True,
                  seed=int(seed) % (2**31 - 1), verbosity=-1)
    booster = lgb.train(config, lgb.Dataset(x, label=y), num_boost_round=500)
    return _refine(booster, x, y) if fourier else TreePrediction(booster)


def fit(method, masks, values, midpoint, gradient, *, prior=None, ridge=1.0, seed=0):
    """Return a callable mask predictor; data, derivatives, and predictions are on CPU.

    The complement residual ridge uses unscaled Walsh features. Dense ridge
    follows the earlier experiment's unit-norm-column ridge with penalty 0.03.
    'proxyspex' uses the raw tree predictor, 'proxyspex_fourier' adds the
    authors' Fourier extraction and energy-truncated CV ridge refinement.
    """
    g = np.asarray(gradient, dtype=float)
    if g.ndim != 1 or not len(g) or not np.isfinite(g).all() or not np.isfinite(midpoint):
        raise ValueError("invalid midpoint response or gradient")
    x = _bits(masks, len(g))
    y = np.asarray(values, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("expected a finite response for each mask")
    if ridge <= 0 or not np.isfinite(ridge):
        raise ValueError("ridge penalty must be positive and finite")
    if prior is not None:
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (len(g), len(g)) or not np.isfinite(prior).all():
            raise ValueError("invalid Hessian prior")
        if not np.allclose(prior, prior.T) or not np.allclose(prior.diagonal(), 0):
            raise ValueError("expected a symmetric, off-diagonal Hessian prior")
    if method == "midpoint":
        return Polynomial(float(midpoint), g / 2)
    if method == "complement":
        return _complement(x, y, midpoint, g, prior, ridge)
    if method == "dense":
        return _dense(x, y, midpoint, g, 0.03)
    if method in ("proxyspex", "proxyspex_fourier"):
        return _tree(x, y, seed, method == "proxyspex_fourier")
    raise ValueError(f"unknown estimator {method!r}")
