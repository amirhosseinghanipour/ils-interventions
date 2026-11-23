import unittest

import numpy as np

from scripts.interaction_search import amortized as a


class AmortizedSearchTests(unittest.TestCase):
    def test_unseen_masks_and_calibration_do_not_overlap(self):
        train, query = a.workload(1985)
        self.assertEqual(len(set(train)), 32)
        self.assertEqual(len(set(query)), 222)
        self.assertEqual(set(query) | set(train) | {0, 7}, set(range(256)))
        self.assertTrue(set(query).isdisjoint((*train, 0, 7)))
        self.assertEqual(query, a.workload(1985)[1])
        self.assertNotEqual(query, a.workload(1986)[1])

    def test_each_model_pays_for_its_own_labels_and_derivatives(self):
        seed = 125
        base = np.linspace(-0.4, 1.2, 8)
        truth = a.BITS @ base + 2.3 * a.INTERACTIONS[:, 5]
        fit, unseen = a.workload(seed)
        outputs = {}
        for method in a.ESTIMATORS:
            calls = []

            def measure(masks):
                self.assertEqual(tuple(masks), fit)
                self.assertTrue(set(masks).isdisjoint(unseen))
                calls.append("patch")
                return truth[masks].tolist()

            def gradient(name):
                calls.append(name)
                return base

            outputs[method] = a.predict(method, seed, 3, gradient, measure)
            self.assertEqual(calls.count("patch"), int(method not in ("midpoint", "ig5")))
            self.assertEqual(
                calls.count("midpoint"), int(method not in ("ig5", "sparse_zero", "ridge_zero"))
            )
            self.assertEqual(calls.count("ig5"), int(method == "ig5"))
        self.assertTrue(np.isfinite(outputs["ils"]).all())
        self.assertLess(
            np.mean(abs(outputs["ils"][unseen] - truth[unseen])),
            np.mean(abs(outputs["midpoint"][unseen] - truth[unseen])),
        )
        self.assertGreater(np.max(abs(outputs["sparse_zero"] - outputs["ils"])), 1e-6)
        self.assertGreater(np.max(abs(outputs["ridge_zero"] - outputs["ridge_midpoint"])), 1e-6)

    def test_offline_control_rejects_unpaid_or_reordered_training_labels(self):
        seed = 511
        train, _ = a.workload(seed)
        base = np.linspace(-1, 1, 8)
        f = a.BITS @ base
        row = dict(
            midpoint=base.tolist(),
            methods={
                "ils": {
                    "paid": [dict(role="fit", masks=list(train), values=f[list(train)].tolist())],
                    "scores": f.tolist(),
                }
            },
        )
        pred = a.from_paid(row, seed, 3)
        np.testing.assert_allclose(pred["ils"], f)
        row["methods"]["ils"]["paid"][0]["masks"][-1] = 7
        with self.assertRaisesRegex(ValueError, "training batch"):
            a.from_paid(row, seed, 3)

    def test_scorer_checks_exact_predictions_against_independent_labels(self):
        seed = a.SEED
        train, order = a.workload(seed)
        gold = a.BITS @ np.arange(8, dtype=float)
        rec = dict(
            task="ioi",
            index=12,
            family="A",
            common_ms=3.0,
            seed=seed,
            train=list(train),
            order=order,
            batches={},
        )
        for k in (1, 32, 128, 222):
            batch = {}
            for m in a.METHODS:
                values = gold[order[:k]].tolist()
                trained = m not in (*a.EXACT, "midpoint", "ig5")
                ms = 2.0 if m in a.EXACT else 1.0
                batch[m] = dict(
                    times_ms=[ms] * 3,
                    train_values=gold[list(train)].tolist() if trained else [],
                    values=values,
                )
            rec["batches"][str(k)] = batch
        native = {"ioi": {12: dict(gold=gold.tolist(), family="A")}}
        summary = a.speed_report([rec], native)
        self.assertEqual(summary["128"]["ils"]["paired_speedup"], 1.25)
        self.assertEqual(summary["128"]["ils"]["marginal_paired_speedup"], 2.0)
        rec["batches"]["222"]["exact32"]["values"][0] += 1
        with self.assertRaises(AssertionError):
            a.speed_report([rec], native)


if __name__ == "__main__":
    unittest.main()
