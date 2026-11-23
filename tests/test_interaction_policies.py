import unittest

import numpy as np

from scripts.interaction_search import policies


class InteractionPolicyTests(unittest.TestCase):
    def test_balanced_calibration_excludes_search_targets(self):
        for n in (16, 32):
            for seed in (0, 17, 412):
                masks = policies.calibration_masks(seed, n)
                self.assertEqual(len(set(masks)), n)
                self.assertTrue(all((255 ^ m) in masks for m in masks))
                self.assertTrue(set(masks).isdisjoint(policies.TRIPLES))
                x = policies.BITS[list(masks)]
                np.testing.assert_array_equal(x.sum(axis=0), np.full(8, n // 2))
                self.assertEqual(np.linalg.matrix_rank(x), 8)
                self.assertEqual(masks, policies.calibration_masks(seed, n))

    def test_shared_labels_and_zero_pair_nesting(self):
        base = np.linspace(-0.8, 0.8, 8)
        truth = policies.BITS @ base + 3.0 * policies.INTERACTIONS[:, policies.PAIRS.index((0, 1))]
        truth += 0.75 * policies.INTERACTIONS[:, policies.PAIRS.index((3, 7))]
        seen = []

        def measure(ids):
            self.assertTrue(all(i.bit_count() != 3 for i in ids))
            seen.append(ids)
            return truth[ids]

        grid = policies.propose_grid({"midpoint": base}, measure, seed=31)
        self.assertEqual(len(seen), 1)
        linear = policies.propose("linear", {"midpoint": base}, measure, seed=31)
        zero = policies.propose("ils", {"midpoint": base}, measure, seed=31, max_pairs=0)
        for other in (zero, grid[0]):
            np.testing.assert_array_equal(other.scores, linear.scores)
            self.assertEqual(other.calibration_masks, linear.calibration_masks)
        sparse = policies.propose("ils", {"midpoint": base}, measure, seed=31, max_pairs=3)
        np.testing.assert_array_equal(grid[3].scores, sparse.scores)
        self.assertEqual(sparse.selected_pairs[:2], ((0, 1), (3, 7)))
        masks = list(policies.TRIPLES)
        err = np.abs(sparse.scores[masks] - truth[masks]).mean()
        self.assertLess(err, 0.1 * np.abs(linear.scores[masks] - truth[masks]).mean())
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(ids == seen[0] for ids in seen))

    def test_static_methods_skip_measurements_and_greedy_completes_best_pair(self):
        def forbidden(ids):
            self.fail(f"unexpected measurements {ids}")

        slopes = np.array([-1, -2, -4, -8, -16, -32, -64, -128], dtype=float)
        gradients = {method: slopes for method in ("midpoint", "atp", "ig5")}
        for method in gradients:
            result = policies.propose(method, gradients, forbidden, seed=0)
            self.assertEqual(result.ranking[0], 7)
        random = policies.propose("scan", {}, forbidden, seed=31)
        self.assertEqual(set(random.ranking), set(policies.TRIPLES))
        proposed = []
        truth = policies.BITS @ slopes

        def observed(ids):
            proposed.append(ids)
            return truth[ids]

        greedy = policies.propose("greedy", {}, observed, seed=0)
        self.assertEqual(proposed[0], [1 << j for j in range(8)])
        self.assertEqual(len(proposed[1]), 7)
        self.assertTrue(all((m & 1) and m.bit_count() == 2 for m in proposed[1]))
        self.assertEqual(set(greedy.ranking[:6]), {m for m in policies.TRIPLES if m & 3 == 3})
        self.assertEqual(len(set(greedy.ranking)), 56)

    def test_target_labels_are_irrelevant_until_explicit_verification(self):
        gradients = {"midpoint": np.arange(8, dtype=float)}
        y = policies.BITS @ gradients["midpoint"]
        alt = y.copy()
        alt[list(policies.TRIPLES)] *= -101
        a, b = [policies.propose("ils", gradients, lambda ids: values[ids], seed=87) for values in (y, alt)]
        np.testing.assert_array_equal(a.scores, b.scores)
        self.assertEqual(a.ranking, b.ranking)
        first, second, third = a.ranking[:3]
        self.assertEqual(policies.best_verified(a.ranking, {}), first)
        self.assertEqual(policies.best_verified(a.ranking, {third: 5, second: 5, first: -1}), second)
        self.assertEqual(policies.best_verified(a.ranking, {255: 1e10, second: 2, first: -1}), second)


if __name__ == "__main__":
    unittest.main()
