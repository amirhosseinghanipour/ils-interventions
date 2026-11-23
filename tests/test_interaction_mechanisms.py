"""Exact contrasts, conditional behavior and selection on an independent task."""

import unittest

import numpy as np

from scripts.interaction_search.mechanisms import MASKS, PAIRS, heldout, select_on_ioi, summarize_task


class InteractionMechanismTests(unittest.TestCase):
    def test_planted_context_interaction_and_frozen_selection(self):
        sites = [[layer, head] for layer in (11, 17) for head in range(4)]
        bits = ((MASKS[:, None] >> np.arange(8)) & 1).T
        a, b, c = bits[[0, 4, 1]]

        def table(values, donor_selected):
            return summarize_task(sites, values[None, :], donor_selected[None, :], np.array([1]))

        io, measurements, rank = table(-2 * a + b + 5 * a * b + 8 * a * b * c, a * b)
        idx, selection = select_on_ioi(io, measurements, rank)
        self.assertEqual(PAIRS[idx], (0, 4))
        self.assertEqual(selection["mean_signed_interaction"], 9.0)
        self.assertEqual(selection["mean_abs_interaction"], 9.0)
        self.assertEqual(selection["material_margin_reversal_fraction"], 0.5)
        self.assertEqual(selection["donor_switch_depends_on_partner_fraction"], 1.0)
        self.assertEqual(io["by_layer"]["within_block"]["pairs"], 12)
        self.assertEqual(io["by_layer"]["across_blocks"]["pairs"], 16)
        # A different pair dominates on transfer; the chosen pair reverses its sign.
        co, measurements, rank = table(-3 * a * b + 8 * c * bits[2], np.zeros_like(a))
        check = heldout(co, measurements, rank, idx, selection["selected_sign"])
        self.assertEqual(check["bits"], (0, 4))
        self.assertEqual(check["selected_ioi_sign"], 1)
        self.assertEqual(check["colors_recurrence_rank"], 2)
        self.assertEqual(check["uncertainty"]["interaction_aligned_to_ioi_sign_logits"]["mean"], -3.0)
        self.assertEqual(check["uncertainty"]["material_context_fraction_aligned_to_ioi_sign"]["mean"], 0.0)

    def test_additive_margin_can_have_partner_dependent_argmax(self):
        sites = [[layer, head] for layer in (11, 17) for head in range(4)]
        a, b = ((MASKS >> b) & 1 for b in (2, 6))
        y = a + b
        r, _, _ = summarize_task(sites, y[None, :], (y == 2)[None, :], np.array([1]))
        pair = r["pairs"][PAIRS.index((2, 6))]
        self.assertEqual(pair["mean_abs_interaction"], 0.0)
        self.assertEqual(pair["donor_switch_depends_on_partner_fraction"], 1.0)
        self.assertAlmostEqual(pair["prompt_top_fraction"], 1 / 28)
