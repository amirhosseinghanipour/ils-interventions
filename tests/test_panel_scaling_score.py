"""Regression tests for the frozen many-query operating-point analysis."""

import copy
import hashlib
import unittest

import numpy as np

from scripts.panel_scaling import score
from scripts.panel_scaling.fit import design, fit


class PanelScalingScoreTests(unittest.TestCase):
    def test_all_budget_replay_detects_corruption_and_development_only_rule(self):
        prompt_id = "route-02-0"
        seed = int.from_bytes(hashlib.sha256(prompt_id.encode()).digest()[:4], "big")
        tr, tar = design(seed, pairs=128)
        g = np.linspace(0.01, 0.04, 16)
        y = fit("midpoint", g, [], [], tr)
        gold = fit("midpoint", g, [], [], tar)
        grid = {m: {str(b): 0.0 for b in score.budgets(m)} for m in score.METHODS}
        row = dict(id=prompt_id, gradient=g, calibration=y, gold=gold, errors=grid)
        report = score.grid_replay(row)
        self.assertEqual(set(score.select_budgets(report).values()), {0})
        grid["ils"]["32"] = 0.005
        with self.assertRaises(AssertionError):
            score.grid_replay(row)
        # The first passing budget wins even for non-monotone development MAE.
        dev = {m: {str(b): 0.2 for b in score.budgets(m)} for m in score.METHODS}
        dev["ils"].update({"24": 0.1, "64": 0.099})
        self.assertEqual(
            score.select_budgets(dev), {"ils": 24, "linear": None, "dense": None, "midpoint": None}
        )

    def test_fallback_times_use_the_same_paid_exact_call_and_charge_shared_capture_once(self):
        meta = dict(id="gift-02-0", task="gift", template="gift-02")
        operating = dict(midpoint=None, linear=None, dense=0, ils=32)
        row = dict(
            **meta,
            methods=dict(midpoint="exact", linear="exact", dense="midpoint", ils="ils"),
            timings_ms=dict(exact=[100, 160, 110], midpoint=[2, 20, 3], ils=[53, 55, 95]),
            common_ms=11,
            labels=[2, 5],
            clean_top=2,
            donor_top=7,
            native_max_error=1e-5,
            gold=[1, -1],
        )
        grid = dict(midpoint={"0": 0.3}, linear={"0": 0.3}, dense={"0": 0.3}, ils={"32": 0.09})
        r = score.point(row, meta, operating, grid)
        self.assertEqual(r["times"], dict(midpoint=121, linear=121, dense=14, ils=66, exact=121))
        self.assertEqual(r["errors"], dict(midpoint=0, linear=0, dense=0.3, ils=0.09, exact=0))
        self.assertTrue(r["clean_correct"])
        self.assertFalse(r["donor_correct"])
        self.assertEqual(r["raw_midpoint_ms"], 14)
        bad = copy.deepcopy(row)
        bad["timings_ms"]["exact"].pop()
        with self.assertRaisesRegex(ValueError, "repetitions"):
            score.point(bad, meta, operating, grid)

    def test_simultaneous_lower_bounds_resample_templates_independently_between_tasks(self):
        gift = np.stack([np.arange(6) + 2, np.ones(6) * 10, np.arange(6) * 7 + 1], axis=1)
        route = gift[::-1] * np.array([1, 2, 4])
        seed, draws = 239931, 300
        s = score.simultaneous(dict(gift=gift, route=route), seed=seed, draws=draws)
        rng = np.random.default_rng(seed)
        matrices = [gift, route]
        boots = []
        centers = []
        for a in matrices:
            centers.extend(a.mean(axis=0))
            weights = np.array([np.bincount(rng.integers(0, 6, 6), minlength=6) for _ in range(draws)])
            boots.append(weights @ a / 6)
        centers = np.array(centers)
        boot = np.column_stack(boots)
        se = boot.std(axis=0, ddof=1)
        v = se > 1.0e-10
        crit = np.quantile(((boot[:, v] - centers[v]) / se[v]).max(axis=1), 0.95)
        want = centers - crit * se
        actual = [s["tasks"][t][c]["lower95"] for t in score.TASKS for c in score.CONTRASTS]
        np.testing.assert_allclose(actual, want)
        self.assertEqual(s["tasks"]["gift"]["dense_time_advantage_ms"]["lower95"], 10)
        self.assertTrue(s["positive_bounds"])
        gift[:, 1] = 0
        self.assertFalse(score.simultaneous(dict(gift=gift, route=route), draws=100)["positive_bounds"])
