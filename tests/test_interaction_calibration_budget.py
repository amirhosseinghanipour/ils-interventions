import unittest

import numpy as np

from scripts.interaction_search import calibration_budget as cb
from scripts.interaction_search.policies import BITS, calibration_masks, propose_grid


class CalibrationBudgetTests(unittest.TestCase):
    def test_complement_prefixes_use_only_mask_order(self):
        full = calibration_masks(517)
        last = ()
        for b in cb.BUDGETS:
            cur = cb.nested_masks(full, b)
            self.assertEqual(len(cur), b)
            self.assertTrue(set(last).issubset(cur))
            self.assertEqual(tuple(m for m in full if m in cur), cur)
            self.assertTrue(all((m ^ 255) in cur for m in cur))
            self.assertTrue(all(m.bit_count() != 3 for m in cur))
            last = cur
        self.assertEqual(last, full)
        with self.assertRaises(ValueError):
            cb.nested_masks(full[:-1] + full[:1], 16)

    def test_no_extra_labels_at_reduced_budgets_and_full_reproduces_original(self):
        full = calibration_masks(517)
        grad = np.linspace(-1, 1, 8)
        measured = BITS @ grad + 0.3 * BITS[:, 2] * BITS[:, 7]
        original = propose_grid({"midpoint": grad}, lambda ids: measured[ids], seed=517)
        for b in cb.BUDGETS:
            queries = []

            def observe(ids):
                queries.append(tuple(ids))
                return measured[ids]

            p = cb.predictors(grad, full, b, observe)
            if b:
                self.assertEqual(queries, [cb.nested_masks(full, b)])
            else:
                self.assertEqual(queries, [])
                np.testing.assert_array_equal(p["ils"].scores, BITS @ grad)
            if b == 32:
                for m, k in [("linear", 0), ("ils", 3)]:
                    np.testing.assert_array_equal(p[m].scores, original[k].scores)
                    self.assertEqual(p[m].ranking, original[k].ranking)

    def test_budget_choice_uses_mae_threshold_and_handles_no_gain(self):
        self.assertEqual(cb.choose_budget(dict(zip(cb.BUDGETS, [10, 9, 8, 3, 2, 1]))), 32)
        self.assertEqual(cb.choose_budget(dict(zip(cb.BUDGETS, [10, 8, 1.5, 3, 1.2, 1]))), 8)
        self.assertEqual(cb.choose_budget(dict(zip(cb.BUDGETS, [1, 2, 0.8, 0.7, 0.8, 1]))), 0)


if __name__ == "__main__":
    unittest.main()
