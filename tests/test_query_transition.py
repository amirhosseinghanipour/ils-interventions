import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/query_transition"))
from affine import analyze
from real_control import crossed
from centering import evaluate
from score import count_flips
from transport import paths


class QueryTransitionTests(unittest.TestCase):
    def test_training_scalar_recovers_unseen_coordinates(self):
        # Exact antisymmetric source pairs create the rank deficiency of the real fit.
        x=np.array([[[1.,0,0],[-1.,0,0]],[[0,1.,0],[0,-1.,0]]])
        xt=np.array([[[0.,0,3.],[0.,0,-3.]]])
        for prediction in paths(x,2*x,xt,True):
            np.testing.assert_allclose(prediction,2*xt)
        for prediction in paths(x,2*x,xt,False):
            np.testing.assert_allclose(prediction,0.)

    def test_wrong_first_word_cannot_count_as_a_two_word_flip(self):
        rows=[dict(second=[10,20],first=1,first_top=[9,1],
                   clean=[dict(top_token=10),dict(top_token=20)])]
        d=count_flips(rows,np.array([[20,10]]))
        self.assertEqual(d["alternate_second_after_forced_first"],2)
        self.assertEqual(d["alternate_two_word_prefix"],1)
        self.assertEqual(d["flips_from_clean_correct"],1)

    def test_shared_intercept_and_sign_symmetry_can_be_distinguished(self):
        rng = np.random.default_rng(121)
        x = rng.normal(size=(16,1,6))
        x = np.concatenate([x,-x],axis=1)
        xt = 2*x[:5]
        raw = {"x":x,"y":3*x+1,"xt":xt,"yt":3*xt+1}
        out = analyze(raw)
        self.assertAlmostEqual(out["affine_scalar_r2"],1.)
        self.assertAlmostEqual(out["oracle_odd_with_optimal_constant_r2_max"],1.)
        self.assertLess(out["oracle_odd_r2_max"],1.)

    def test_crossed_donor_uses_next_direction_and_target_norm(self):
        edits = np.array([[0.,3.],[2.,0.]])
        np.testing.assert_array_equal(crossed(edits,0),[3.,0.])
        np.testing.assert_array_equal(crossed(edits,1),[0.,2.])

    def test_crossfit_removes_a_common_task_offset(self):
        bank=np.eye(4)
        target=np.array([[0,1],[2,3],[0,1],[2,3]])
        y=bank[target]+np.array([100.,0,0,0])
        out=evaluate(bank,y,target)
        self.assertEqual(out["crossfit"]["correct"],8)
        self.assertLess(out["carrier"]["correct"],8)


if __name__=="__main__":unittest.main()
