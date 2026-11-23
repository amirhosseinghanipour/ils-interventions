import unittest

import numpy as np

from scripts.panel_scaling.equal_budget import compare, paired_interval
from scripts.panel_scaling.score import BUDGETS


def example(mid, sparse, dense, linear):
    result = {"midpoint": {"0": mid}}
    for name, loss in (("ils", sparse), ("dense", dense), ("linear", linear)):
        result[name] = {str(n): (loss if n >= 128 else mid) for n in BUDGETS}
    return result


class EqualBudgetTests(unittest.TestCase):
    def test_paired_template_resampling_and_dev_selection(self):
        templates = np.repeat(["a", "b", "c", "d", "e", "f"], 4)
        diff = np.tile([-10.0, -9.0, 10.0, 11.0], 6)
        paired = paired_interval(diff, templates)
        self.assertEqual(paired["pointwise_ci95"], [0.5, 0.5])
        self.assertEqual(paired["templates_with_lower_sparse_mae"], 6)
        dev = [example(0.2, 0.08, 0.12, 0.15)] * 8
        holdout = [example(0.1, 0.13, 0.01, 0.2)] * 24
        report = compare(dev, holdout, templates)
        selected = report["selected_for_half_target"]
        self.assertAlmostEqual(report["half_development_midpoint_target"], 0.1)
        self.assertEqual(selected["ils"]["count"], 128)
        self.assertAlmostEqual(selected["ils"]["heldout_mae"], 0.13)
        self.assertEqual(selected["dense"], {"count": None, "policy": "exact", "heldout_mae": 0.0})
        np.testing.assert_allclose(
            report["equal_budget"]["256"]["dense_minus_sparse_mae"]["pointwise_ci95"], [-0.12, -0.12]
        )
