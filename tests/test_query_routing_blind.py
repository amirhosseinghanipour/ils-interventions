import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/query_routing"))
from blind_score import infer


class BlindRetrievalTests(unittest.TestCase):
    def test_permutation_and_sign_with_hidden_query_order(self):
        rng=np.random.default_rng(79)
        dictionary=rng.normal(size=(8,20))
        query=rng.normal(size=(8,4,20))
        old=np.array([0,5,3,1])
        new=np.array([1,7,4,6])
        asked=np.array([5,5,3,0])
        delta=dictionary[new]-dictionary[old]
        for order in (np.arange(4),np.array([3,1,0,2])):
            a,b,q=infer(dictionary,query,delta[order],query[asked[order]])
            np.testing.assert_array_equal(a,old[order])
            np.testing.assert_array_equal(b,new[order])
            np.testing.assert_array_equal(q,np.repeat(asked[order,None],4,axis=1))
        back,forward,_=infer(dictionary,query,-delta,query[asked])
        np.testing.assert_array_equal(back,new)
        np.testing.assert_array_equal(forward,old)


if __name__=="__main__":unittest.main()
