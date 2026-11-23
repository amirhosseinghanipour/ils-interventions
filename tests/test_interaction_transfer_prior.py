import unittest

import numpy as np

from scripts.interaction_search import transfer_prior as tp


class RestrictedGold:
    def __init__(self, allowed, data):
        self.allowed = set(allowed)
        self.data = data
        self.observed = []

    def __getitem__(self, mask):
        if mask not in self.allowed:
            raise AssertionError("training read an unpermitted measurement")
        self.observed.append(mask)
        return self.data[mask]


class TransferredPriorTests(unittest.TestCase):
    def test_collect_development_reads_only_registered_nontarget_masks(self):
        rows = []
        for n in range(18):
            masks = tp.calibration_masks(781023 + n)
            g = np.linspace(-0.2, 0.7, 8)
            labels = RestrictedGold(masks, tp.BITS @ g)
            rows.append(dict(file="ioi-test.json", index=n, midpoint=g, gold=labels))
        samples = tp.collect_development(rows)
        self.assertEqual(sum(len(s[2]) for s in samples), 576)
        for n, r in enumerate(rows):
            self.assertEqual(tuple(r["gold"].observed), tp.calibration_masks(781023 + n))
            self.assertTrue(set(r["gold"].observed).isdisjoint(tp.TRIPLES))
        rows[2]["file"] = "colors-validation.jsonl"
        with self.assertRaises(ValueError):
            tp.collect_development(rows)

    def test_rank_alignment_and_margin_sign_equivariance(self):
        # Permuting physical heads within either block preserves the canonical fit coordinates.
        g = np.array([-0.5, 0.1, 0.2, 0.8, 0.7, -0.15, 0.3, 0.45])
        swap = np.array([2, 3, 0, 1, 7, 4, 5, 6])
        rng = np.random.default_rng(730)
        models = {"transfer_additive": rng.normal(size=8), "transfer_pairs": rng.normal(size=64)}
        original = tp.predict(models, g)
        permuted = tp.predict(models, g[swap])
        negated = tp.predict(models, -g)
        power = 1 << np.arange(8)
        new_masks = (tp.BITS[:, swap] @ power).astype(int)
        for method, p in original.items():
            np.testing.assert_allclose(permuted[method][new_masks], p, atol=1e-12)
            np.testing.assert_allclose(negated[method], -p, atol=1e-12)
            self.assertEqual(p[0], 0)
        zero = tp.predict(models, np.zeros(8))
        for v in zero.values():
            np.testing.assert_array_equal(v, np.zeros(256))

    def test_transferred_prediction_is_independent_of_holdout_effects(self):
        # Build the prior on independent 32-mask batches, then invert the holdout objective.
        rows = []
        for n in range(18):
            rng = np.random.default_rng(42 + n)
            g = rng.normal(size=8)
            y = tp.BITS @ (1.2 * g)
            rows.append(dict(file="ioi-test.json", index=n, midpoint=g, gold=y))
        models = tp.fit(tp.collect_development(rows))
        g = np.arange(8, dtype=float) * 0.1
        y = tp.BITS @ g
        hold = dict(index=0, family="a", midpoint=g, gold=y, methods={"ils": {"scores": y.tolist()}})
        first = tp.evaluate(models, [hold])
        hold["gold"] = -y
        second = tp.evaluate(models, [hold])
        for name in tp.METHODS:
            self.assertEqual(
                first["rows"][0]["metrics"][name]["raw_mask"],
                second["rows"][0]["metrics"][name]["raw_mask"],
            )
        self.assertNotEqual(first["summary"]["midpoint"]["regret"], second["summary"]["midpoint"]["regret"])


if __name__ == "__main__":
    unittest.main()
