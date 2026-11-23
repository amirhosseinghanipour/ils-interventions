import unittest
import numpy as np
import torch
from ilslens.scalar_transport import scalar_transport
from ilslens.eval.faithfulness import paired_prompt_r2
from scripts.internal_replication.score import paths, eps


class InternalCoverageTests(unittest.TestCase):
    def test_scalar_constant_and_variable_horizon(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(17, 11))
        for w in [np.ones(17), rng.random(17)]:
            y = (0.32 + 0.21 * w[:, None]) * x
            pred = scalar_transport(x[:13], y[:13], w[:13], x[13:], w[13:])
            np.testing.assert_allclose(pred, y[13:], atol=1e-12)

    def test_dual_matches_primal_for_both_priors(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(13, 19))
        y = rng.normal(size=x.shape)
        xt = rng.normal(size=(7, 19))
        for use_identity in [False, True]:
            for lam, pred in zip(eps, paths(x, y, xt, use_identity)):
                if not lam:
                    reference = xt @ np.linalg.pinv(x) @ y
                else:
                    gram = x.T @ x + len(x) * lam * np.eye(x.shape[1])
                    rhs = x.T @ y + len(x) * lam * float(use_identity) * np.eye(
                        x.shape[1]
                    )
                    reference = xt @ np.linalg.solve(gram, rhs)
                np.testing.assert_allclose(pred, reference, atol=1e-8)

    def test_paired_difference_matches_brute_prompt_resampling(self):
        y = torch.arange(28, dtype=torch.double).reshape(7, 4) / 13
        groups = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        pred = y * 0.61
        control = y * 0.35
        ans = paired_prompt_r2(y, {"ILS": pred, "control": control}, groups)
        counts = np.random.default_rng(0).multinomial(3, [1 / 3] * 3, size=2000)
        draws = []
        for c in counts:
            idx = np.repeat(np.arange(len(y)), c[groups.numpy()])
            yy = y[idx]
            ss = (yy - yy.mean(0)).square().sum()
            draws.append(
                float(
                    (
                        (yy - control[idx]).square().sum()
                        - (yy - pred[idx]).square().sum()
                    )
                    / ss
                )
            )
        np.testing.assert_allclose(
            ans["ILS minus control"]["ci"], np.quantile(draws, [0.025, 0.975])
        )
