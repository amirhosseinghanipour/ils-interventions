import itertools
import unittest
from importlib.util import find_spec

import numpy as np

from scripts.anytime.estimators import (
    FourierPrediction, _fourier_terms, fit, hvp_prior,
)


class AnytimeEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(734)
        self.k = 8
        x = self.rng.integers(0, 2, size=(24, self.k))
        self.train = np.stack([x, 1 - x], axis=1).reshape(-1, self.k)
        self.query = np.asarray(list(itertools.product([0, 1], repeat=self.k)))
        self.g = self.rng.normal(size=self.k)

    def test_sketch_removes_a_pure_diagonal_even_with_only_two_probes(self):
        v = self.rng.choice([-1, 1], size=(2, self.k))
        diag = self.rng.normal(size=self.k)
        np.testing.assert_allclose(hvp_prior(v, v * diag), np.zeros((self.k, self.k)),
                                   atol=1e-12)

    def test_full_rank_sketch_recovers_planted_pair_and_ignores_diagonal(self):
        h = np.diag(self.rng.normal(size=self.k))
        h[1, 4] = h[4, 1] = 2.0
        v = self.rng.normal(size=(self.k, self.k))
        est = hvp_prior(v, v @ h)
        np.testing.assert_allclose(est, h - np.diag(h.diagonal()), atol=1e-10)

    def test_even_residual_recovers_known_prior_scale_independent_of_odd_channel(self):
        h = np.zeros((self.k, self.k))
        h[2, 7] = h[7, 2] = 1.2

        def oracle(x, beta):
            z = 2*x - 1
            return 1.7 + z @ self.g / 2 + beta * z[:, 2] * z[:, 7] * 1.2 / 4

        for beta in (0.0, 0.3, 1.0):
            model = fit("complement", self.train, oracle(self.train, beta), 1.7, self.g, prior=h)
            self.assertAlmostEqual(model.beta, beta, places=10)
            np.testing.assert_allclose(model(self.query), oracle(self.query, beta), atol=1e-10)
        zero = fit("complement", self.train, oracle(self.train, 0.0), 1.7, self.g)
        np.testing.assert_allclose(zero(self.query), oracle(self.query, 0.0), atol=1e-10)
        wrong = fit("complement", self.train, oracle(self.train, -1.0), 1.7, self.g, prior=h)
        self.assertEqual(wrong.beta, 0.0)

    def test_four_way_parity_hidden_at_midpoint_is_always_seen_by_corner_check(self):
        g = np.zeros(self.k)
        zero_h = np.zeros((self.k, self.k))
        model = fit("complement", [], [], 0.0, g, prior=zero_h)
        corners = self.train[:8]
        true = np.prod(2*corners[:, :4] - 1, axis=1)
        # Four complement pairs, dependent within each pair, still all have |error| 1.
        self.assertEqual(float(np.abs(true - model(corners)).mean()), 1.0)

    def test_dense_matches_original_normalized_column_design_on_all_masks(self):
        midpoint = 0.23
        y = self.rng.normal(size=len(self.train))
        i, j = np.triu_indices(self.k, 1)

        def columns(x):
            return np.column_stack([x, -0.5 * (x[:, i] != x[:, j])])

        def base(x):
            return midpoint + (x - 0.5) @ self.g

        x = columns(self.train)
        norms = np.linalg.norm(x, axis=0)
        norms[norms == 0] = 1
        scaled = x / norms
        coefs = np.linalg.solve(scaled.T @ scaled + 0.03*np.eye(len(norms)),
                                scaled.T @ (y - base(self.train)))
        model = fit("dense", self.train, y, midpoint, self.g)
        np.testing.assert_allclose(model(self.query), base(self.query) + columns(self.query) @
                                   (coefs / norms), atol=1e-11)

    @unittest.skipUnless(find_spec("lightgbm"), "optional tree baseline")
    def test_exact_tree_to_fourier_and_complete_fit(self):
        z = 2*self.train - 1
        y = 0.3 + z[:, 0] - 2*z[:, 2] + 1.4*z[:, 1]*z[:, 3]
        raw = fit("proxyspex", self.train, y, 0.0, self.g)
        terms = _fourier_terms(raw.booster)
        support = np.array([[i in term for i in range(self.k)] for term in terms], dtype=np.uint8)
        direct = FourierPrediction(support, np.array(list(terms.values())))
        np.testing.assert_allclose(direct(self.query), raw(self.query), atol=1e-10)
        refined = fit("proxyspex_fourier", self.train, y, 0.0, self.g)
        self.assertTrue(np.isfinite(refined(self.query)).all())


if __name__ == "__main__":
    unittest.main()
