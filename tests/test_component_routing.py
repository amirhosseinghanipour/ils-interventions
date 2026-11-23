"""Budget causality and native substitution invariants of the component benchmark."""

import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from scripts.component_routing.amortize import masks_for, predict
from scripts.component_routing.engine import Circuit
from scripts.component_routing.groups import experiment
from scripts.component_routing.score import metrics, snapshot


class ComponentRoutingTests(unittest.TestCase):
    def test_cutoffs_do_not_leak_later_exact_labels(self):
        run = dict(
            raw=[2.0, 1.0, 0.0],
            raw_ms=4.0,
            history=[
                dict(indices=[1], values=[3.0], correction=2.0, ms=8.0),
                dict(indices=[0], values=[-1.0], correction=None, ms=11.0),
            ],
        )
        for cap, expected, known, missed in [
            (3.0, [0, 0, 0], [], True),
            (6.0, [2, 1, 0], [], False),
            (8.0, [4, 3, 2], [1], False),
            (12.0, [-1, 3, 2], [0, 1], False),
        ]:
            actual, mask, _, fallback = snapshot(run, cap)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.flatnonzero(mask), known)
            self.assertEqual(fallback, missed)

    def test_direction_and_uniform_tie_credit(self):
        # Raw margins favor the donor; the named positive direction favors the original.
        y = np.array([-4.0, 1.0, 3.0])
        r = metrics(y, np.array([-2.0, 2.0, 2.0]), np.zeros(3, bool))
        self.assertEqual(r["increase"], 1.0)
        self.assertAlmostEqual(r["decrease"], 2 / 3)
        self.assertAlmostEqual(r["directional"], 5 / 6)

    def test_native_patches_and_gradients_on_both_decoder_types(self):
        for cfg, cls in ((Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM)):
            torch.manual_seed(731)
            config = cfg(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=64,
            )
            config._attn_implementation = "eager"
            model = cls(config).eval().requires_grad_(False)
            circuit = Circuit(model, [[1, 7, 9, 10, 11], [1, 21, 9, 10, 11]], [3, 4])
            circuit.audit()
            np.testing.assert_allclose(
                circuit.estimates("ils", 0.0), circuit.estimates("midpoint", 0.0), atol=1e-7
            )
            grads = circuit.global_grad(0.0)
            for n, (layer, kind, delta) in enumerate(circuit.groups):
                h = circuit.ends[0][0][layer, kind]
                move = delta[:1]
                fd = (
                    circuit.tail(layer, kind, h + 0.02 * move) - circuit.tail(layer, kind, h - 0.02 * move)
                ) / 0.04
                tangent = (grads[n] * move).sum().item()
                self.assertAlmostEqual(float(fd), tangent, delta=1e-5 + abs(tangent) * 0.015)
            same = Circuit(model, [[1, 7, 9], [1, 7, 9]], [3, 4])
            np.testing.assert_allclose(same.estimates("ils", 1.0), 0.0, atol=1e-6)
            np.testing.assert_allclose(same.exact([0, 4, 5]), 0.0, atol=1e-6)

    def test_gram_predictions_match_explicit_head_substitutions(self):
        masks = masks_for(14, 34399)
        self.assertEqual(len(np.unique(masks, axis=0)), 128)
        torch.manual_seed(10)
        config = LlamaConfig(
            vocab_size=40,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
        )
        config._attn_implementation = "eager"
        c = Circuit(
            LlamaForCausalLM(config).eval().requires_grad_(False), [[1, 7, 2, 9], [1, 8, 2, 9]], [3, 4]
        )
        measured = experiment(c, 34399)
        m = torch.tensor(np.argsort(measured["order"])[None, :] < np.array(measured["sizes"])[:, None])
        for method in ("recipient", "midpoint", "nearest", "smooth", "secant_midpoint"):
            expected = measured["predictions"]["matched_midpoint" if method == "midpoint" else method]
            np.testing.assert_allclose(predict(c, m, method).numpy(), expected, atol=5e-7, rtol=2e-5)
