"""Checks on the sixteen-head calibration design and the sparse ridge objective."""

import unittest
from itertools import combinations

import numpy as np

from scripts.panel_scaling.fit import ALL, HEADS, RIDGE, design, fit


def features(masks):
    gates = np.array([[(m >> i) & 1 for i in range(HEADS)] for m in masks], dtype=float)
    pair_cols = [
        gates[:, i] * gates[:, j] - 0.5 * (gates[:, i] + gates[:, j])
        for i, j in combinations(range(HEADS), 2)
    ]
    return np.column_stack([gates, *pair_cols])


class PanelFitTest(unittest.TestCase):
    def test_nested_complement_design_and_separate_uniform_query_pool(self):
        for seed in (3, 17, 401):
            train, query = design(seed)
            self.assertEqual((len(set(train)), len(set(query))), (128, 512))
            self.assertEqual(design(seed), (train, query))
            self.assertTrue(set(train).isdisjoint(query))
            self.assertTrue(all(0 < m < ALL for m in train + query))
            self.assertTrue(all(a ^ b == ALL for a, b in zip(train[::2], train[1::2])))
            for n in (8, 16, 32, 64, 128):
                np.testing.assert_array_equal(
                    features(train[:n])[:, :HEADS].sum(axis=0), np.full(HEADS, n // 2)
                )
        self.assertNotEqual(design(3)[1], design(17)[1])

    def test_sparse_greedy_matches_reference_penalized_loss(self):
        train, q = design(51)
        g = np.random.default_rng(17).normal(size=HEADS)
        a = features(train)
        target = features(q[:16])
        residual = np.random.default_rng(12).normal(size=len(train))
        y = a[:, :HEADS] @ g + residual
        norm = np.linalg.norm(a, axis=0)
        a /= norm
        active = list(range(HEADS))

        def solve(cols):
            x = a[:, cols]
            c = np.linalg.solve(x.T @ x + RIDGE * np.eye(len(cols)), x.T @ residual)
            err = residual - x @ c
            return err @ err + RIDGE * (c @ c), c

        for _ in range(8):
            candidates = [k for k in range(HEADS, a.shape[1]) if k not in active]
            active.append(min(candidates, key=lambda k: solve(active + [k])[0]))
        coef = solve(active)[1]
        expected = target[:, :HEADS] @ g + (target[:, active] / norm[active]) @ coef
        np.testing.assert_allclose(fit("ils", g, train, y, q[:16]), expected, atol=1e-11)

    def test_dense_matches_primal_ridge_and_query_batch_does_not_change_predictions(self):
        train, q = design(181)
        train = train[:32]
        g = np.random.default_rng(21).normal(size=HEADS)
        a = features(train)
        b = features(q[:20])
        y = np.random.default_rng(32).normal(size=len(train))
        norm = np.linalg.norm(a, axis=0)
        norm[norm == 0] = 1
        z = a / norm
        beta = np.linalg.solve(z.T @ z + RIDGE * np.eye(z.shape[1]), z.T @ (y - a[:, :HEADS] @ g))
        np.testing.assert_allclose(fit("dense", g, train, y, q[:20]), b[:, :HEADS] @ g + (b / norm) @ beta)
        for m in ("midpoint", "linear", "dense", "ils"):
            pred = fit(m, g, train, y, q)
            ix = [2, 103, 29, 19]
            isolated = fit(m, g, train, y, [q[i] for i in ix])
            np.testing.assert_allclose(isolated, pred[ix], atol=1e-11)
        with self.assertRaises(ValueError):
            fit("ils", g, train, np.r_[y, np.zeros(len(q))], q)

    def test_recovers_shared_pair_signal_on_unseen_masks(self):
        train, q = design(187)
        g = np.random.default_rng(31).normal(size=HEADS)

        def truth(masks):
            b = features(masks)[:, :HEADS]
            return b @ g - 1.5 * (b[:, 1] != b[:, 13]) + (b[:, 5] != b[:, 7])

        y = truth(train)
        loss = {
            m: np.abs(truth(q) - fit(m, g, train, y, q)).mean()
            for m in ("midpoint", "linear", "dense", "ils")
        }
        self.assertLess(loss["ils"], 0.12)
        self.assertLess(loss["ils"] * 3, min(loss[m] for m in ("midpoint", "linear", "dense")))
        np.testing.assert_allclose(fit("midpoint", g, [], [], [0, ALL]), [0, g.sum()], atol=1e-12)
