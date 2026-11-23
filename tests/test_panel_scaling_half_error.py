"""Check that half-target errors use the actual paid fitting batch and exact outputs."""

import copy
import hashlib
import unittest

import numpy as np

from scripts.panel_scaling.fit import design
from scripts.panel_scaling.half_error import capture, replay_paid


class HalfTargetTests(unittest.TestCase):
    def test_replay_uses_paid_forecasts_and_rejects_corrupt_evidence(self):
        row = dict(id="route-02-2", task="route", template="route-02")
        seed = int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:4], "big")
        train, query = design(seed, pairs=128)
        weights = np.linspace(-0.04, 0.025, 16)

        def native(masks):
            bits = (np.array(masks)[:, None] >> np.arange(16)) & 1
            return bits @ weights - 0.16 * (bits[:, 0] != bits[:, 4])

        class Oracle:
            def exact(self, masks, chunk):
                # Batch-dependent numerical perturbations are inside the native audit tolerance.
                x = native(masks)
                return (x + (0.0002 if len(masks) < 512 else 0.00003)).tolist()

            def gradient(self):
                return (weights + 0.00015).tolist()

            def audit(self, masks):
                return 0.00001

        op = dict(ils=64, dense=32, linear=None, midpoint=None)
        cfg = dict(operating=op, chunks={str(k): {"chunk": min(k, 64)} for k in [32, 64, 512]})
        ticks = iter(range(1000))
        rec = capture(Oracle(), row, train, query, cfg, 6.0, lambda: next(ticks))
        labels = dict(labels=[22, 13], clean_top=22, donor_top=13)
        rec.update(labels)
        raw = dict(
            **row,
            **labels,
            gradient=weights,
            calibration=native(train),
            gold=native(query),
            native_max_error=0.0,
        )
        # These retrospective grid entries deliberately disagree with the actual paid fits.
        grid = dict(ils={"64": 0.004}, dense={"32": 0.34})
        result, diff, gaps = replay_paid(raw, rec, op, grid)
        self.assertEqual(result["errors"]["linear"], 0.0)
        self.assertEqual(result["times"]["linear"], result["times"]["exact"])
        for method in ("dense", "ils"):
            actual = np.mean(
                [
                    np.abs(np.asarray(p["forecast"]) - np.asarray(g["forecast"])).mean()
                    for p, g in zip(rec["paid"][method], rec["paid"]["exact"], strict=True)
                ]
            )
            self.assertAlmostEqual(result["errors"][method], actual)
            self.assertAlmostEqual(gaps[method], abs(actual - grid[method][str(op[method])]))
        self.assertAlmostEqual(diff["calibration"], 0.0002)
        self.assertAlmostEqual(diff["gradient"], 0.00015)
        self.assertAlmostEqual(diff["exact"], 0.00003)
        broken = copy.deepcopy(rec)
        for rep in broken["paid"]["dense"]:
            rep["forecast"][1] += 0.005
        with self.assertRaises(AssertionError):
            replay_paid(raw, broken, op, grid)
        altered = dict(raw, calibration=raw["calibration"] + 0.01)
        with self.assertRaisesRegex(ValueError, "paid calibration"):
            replay_paid(altered, rec, op, grid)
