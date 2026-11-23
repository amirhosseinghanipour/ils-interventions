import copy
import unittest

import numpy as np

from scripts.interaction_search import score


def row(index=0, family="x"):
    # One interaction between a gate from each of the two blocks, worth two logits.
    masks = np.arange(256)
    y = 2.0 * ((masks & 1 != 0) & (masks & 16 != 0))
    best = next(m for m in score.TRIPLES if y[m] == 2)
    top = np.where(y > 0, 12, 9)
    methods = {}
    for name in score.METHODS:
        fit = [3, 252] if name in ("ils", "linear") else []
        paid = [dict(role="fit", masks=fit, values=y[fit].tolist(), ms=1.0)] if fit else []
        scores = y.copy() if name == "ils" else np.zeros(256)
        raw = best if name == "ils" else score.FALLBACK
        winner = score.FALLBACK
        if name == "ils":
            paid.append(dict(role="verify", masks=[best], values=[2.0], ms=2.0, decision_ms=2.5))
            winner = best
        methods[name] = dict(
            cap_ms=3.0,
            initial_ms=1.0,
            winner=winner,
            raw_mask=raw,
            paid=paid,
            scores=scores.tolist() if name in score.SCORED_ESTIMATORS else None,
            calibration_masks=fit,
        )
    return dict(
        family=family,
        index=index,
        gold=y.tolist(),
        top=top.tolist(),
        clean_top=9,
        donor_top=12,
        labels=[9, 12],
        fallback_value=0.0,
        methods=methods,
    )


class SearchScoreTests(unittest.TestCase):
    def test_exact_labels_switches_regret_and_interaction_residuals(self):
        r = score.score_cell([row(0, "a"), row(1, "b")], samples=100)
        self.assertTrue(r["passes_all"])
        self.assertEqual(r["clean_correct_switchable"], 2)
        self.assertEqual(r["methods"]["ils"]["switches"], 2)
        self.assertEqual(r["methods"]["ils"]["regret"], 0)
        self.assertEqual(r["methods"]["linear"]["regret"], 2)
        self.assertEqual(r["methods"]["ils"]["pair_mae"], 0)
        self.assertEqual(r["methods"]["ils"]["coalition_mae"], 0)
        self.assertEqual(r["methods"]["ils"]["triple_interaction_mae"], 0)
        self.assertGreater(r["methods"]["linear"]["coalition_mae"], 0)
        self.assertGreater(r["methods"]["linear"]["cross_pair_mae"], 0)
        self.assertEqual(r["methods"]["linear"]["within_pair_mae"], 0)
        self.assertEqual(r["paired_comparisons"]["linear"]["simultaneous_ci95"], [2.0, 2.0])

    def test_no_superiority_for_tied_policy_or_one_independent_cluster(self):
        rows = [row(0, "a"), row(1, "b")]
        for r in rows:
            r["methods"]["linear"] = copy.deepcopy(r["methods"]["ils"])
        summary = score.score_cell(rows, samples=100)
        self.assertFalse(summary["passes_all"])
        self.assertEqual(summary["paired_comparisons"]["linear"]["simultaneous_ci95"], [0, 0])
        self.assertIsNone(
            score.score_cell(rows[:1], samples=100)["paired_comparisons"]["linear"]["simultaneous_ci95"]
        )

    def test_reject_late_decisions_unavailable_winners_and_falsified_measurements(self):
        r = row()
        r["methods"]["ils"]["paid"][-1]["decision_ms"] = 3.01
        with self.assertRaisesRegex(ValueError, "unavailable"):
            score.score_cell([r], samples=100)
        r = row()
        r["methods"]["ils"]["winner"] = 224
        with self.assertRaisesRegex(ValueError, "unavailable"):
            score.score_cell([r], samples=100)
        r = row()
        r["methods"]["linear"]["paid"][0]["values"][0] = 0.2
        with self.assertRaises(AssertionError):
            score.score_cell([r], samples=100)

    def test_uncertainty_resamples_families(self):
        m = np.array([[1.0], [1.0], [1.0], [-1.0]])
        center, ci = score._bootstrap(m, ["a", "a", "a", "b"], 291, 1000)
        np.testing.assert_array_equal(center, [0.5])
        _, indep = score._bootstrap(m, list("abcd"), 291, 1000)
        self.assertGreater(ci[0, 1] - ci[0, 0], indep[0, 1] - indep[0, 0])


if __name__ == "__main__":
    unittest.main()
