"""The adaptive rule must pay for failed calibration and keep validation independent."""

import unittest
from unittest.mock import patch

import numpy as np

from scripts.panel_scaling import adaptive
from scripts.panel_scaling.fit import design


class AdaptivePolicyTests(unittest.TestCase):
    def test_stopping_reads_only_authorized_masks(self):
        pool, targets = design(5432, pairs=128)
        recorded = []
        source = {m: float(m & 1) for m in pool}

        def measure(ids):
            recorded.append(ids)
            return [source[i] for i in ids]

        for method in ("ils", "dense"):
            with self.subTest(method=method):
                recorded.clear()
                seen = []
                original = adaptive.fit

                def checked(mode, g, ids, y, query):
                    seen.append(list(ids))
                    self.assertEqual(len(y), len(ids))
                    return original(mode, g, ids, y, query)

                with patch.object(adaptive, "fit", side_effect=checked):
                    out = adaptive.policy(method, lambda: [0.0] * 16, measure, pool, targets)
                self.assertGreater(out["accepted"], 0)
                self.assertEqual(recorded[0], pool[-32:])
                self.assertEqual(sum(recorded[1:], []), pool[: out["accepted"]])
                self.assertEqual(len(out["predictions"]), 512)
                for ids in seen:
                    self.assertTrue(set(ids).isdisjoint(pool[-32:]))
                    self.assertTrue(set(ids).isdisjoint(targets))

    def test_midpoint_accept_and_paid_fallback(self):
        pool, targets = design(7654, pairs=128)
        for fail in (False, True):
            with self.subTest(fallback=fail):
                paid = []

                def measure(query):
                    paid.append(query)
                    return [float(fail and m in pool[-32:]) for m in query]

                out = adaptive.policy("ils", lambda: [0.0] * 16, measure, pool, targets)
                self.assertEqual(out["accepted"], None if fail else 0)
                sizes = [32, 16, 16, 32, 64, 96, 512] if fail else [32]
                self.assertEqual([len(q) for q in paid], sizes)
                if fail:
                    self.assertEqual(paid[-1], targets)
                self.assertEqual(out["predictions"], [0.0] * 512)

    def test_one_se_uses_complement_pairs(self):
        y = np.repeat([0.0] * 8 + [0.15] * 8, 2)
        mae, se = adaptive.pair_score(np.zeros(32), y)
        self.assertAlmostEqual(mae, 0.075)
        self.assertAlmostEqual(se, 0.075 / np.sqrt(15))
