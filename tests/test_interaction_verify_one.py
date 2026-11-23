import unittest

import numpy as np

from scripts.interaction_search.policies import BITS, TRIPLES
from scripts.interaction_search.verify_one import FALLBACK, METHODS, run_method, score_cell, subset


class SingleVerificationTests(unittest.TestCase):
    def test_deadlines_apply_to_the_same_paid_query_and_include_decision(self):
        weights = np.arange(8, dtype=float)
        effects = BITS @ weights

        class Oracle:
            def exact(self, masks):
                return effects[masks].tolist()

            def gradient(self, name):
                return weights

        o = Oracle()
        timings = []
        primary = []
        for n in range(12):
            ref = dict(file="fixed", index=n, family=n // 2, labels=[1, 2])
            methods = {
                m: run_method(o, m, 91 + n, 3, effects[FALLBACK], iter([3.0, 18.0]).__next__)
                for m in METHODS
            }
            original = {
                m: dict(
                    calibration_masks=[i for b in trace["paid"] if b["role"] == "fit" for i in b["masks"]]
                )
                for m, trace in methods.items()
            }
            timings.append(
                dict(
                    **ref,
                    original_row=n,
                    fallback_value=effects[FALLBACK],
                    methods=methods,
                    common_ms=3.0,
                    fallback_ms=2.0,
                )
            )
            primary.append(dict(**ref, clean_top=1, top=[1] * 256, gold=effects, methods=original))
            self.assertEqual(methods["ils"]["paid"][-1]["masks"], [224])
            self.assertEqual(methods["exhaustive"]["paid"][-1]["masks"], list(TRIPLES))
        t = dict(sites=list(range(8)), cap_ms=10.0, max_pairs=3, rows=timings)
        p = dict(sites=list(range(8)), rows=primary)
        result = score_cell(t, p, "ioi", 10.0, 3)
        d = result["methods"]["ils"]
        self.assertEqual(d["deadlines"]["1"]["completed"], 0.0)
        self.assertEqual(d["deadlines"]["1"]["regret"], effects[224] - effects[FALLBACK])
        self.assertEqual(d["deadlines"]["2"]["regret"], 0.0)
        self.assertEqual(d["mean_end_to_end_ms"], 20.0)
        self.assertEqual(result["methods"]["fallback"]["deadlines"]["1"]["completed"], 1.0)

    def test_ioi_balanced_subset_retains_original_row_numbers(self):
        records = [dict(family=i // 8) for i in range(48)]
        self.assertEqual(
            [n for n, _ in subset(records, "ioi")], [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41]
        )
        self.assertEqual([n for n, _ in subset(records, "colors")], list(range(12)))
