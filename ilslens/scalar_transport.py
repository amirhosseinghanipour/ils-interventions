"""Fit a two-coefficient transport using only training interventions."""

import numpy as np


def scalar_transport(x, y, w, xt, wt):
    # Solve the two-column least-squares problem without materializing a
    # design matrix of shape (number of samples times residual width, 2).
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w, wt = np.asarray(w, dtype=float), np.asarray(wt, dtype=float)
    energy = np.einsum("ij,ij->i", x, x)
    dot = np.einsum("ij,ij->i", x, y)
    basis = np.column_stack([np.ones(len(w)), w])
    coef = np.linalg.pinv(basis.T @ (energy[:, None] * basis), hermitian=True) @ (
        basis.T @ dot
    )
    return np.asarray(xt) * (coef[0] + wt[:, None] * coef[1])
