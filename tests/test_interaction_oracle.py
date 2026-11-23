"""Cross-check two-block suffix reuse and gating derivatives against full native forwards."""

import unittest
from contextlib import ExitStack

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from scripts.interaction_search.oracle import Oracle


def direct(model, ids, sites, gates):
    """Patch at each o_proj from the untouched donor in a complete model forward."""
    heads = model.config.num_attention_heads
    layers = sorted({layer for layer, head in sites})
    donor = {}
    with torch.no_grad(), ExitStack() as hooks:
        for layer in layers:

            def capture(module, args, layer=layer):
                donor[layer] = args[0][0].reshape(len(ids[1]), heads, -1)

            hooks.enter_context(
                model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(capture)
            )
        model(torch.tensor([ids[1]]), use_cache=False)
    with torch.no_grad(), ExitStack() as hooks:
        for layer in layers:

            def substitute(module, args, layer=layer):
                native = args[0].unflatten(-1, (heads, -1))
                parts = []
                for h in range(heads):
                    value = native[:, :, h]
                    if (layer, h) in sites:
                        j = sites.index((layer, h))
                        value = torch.lerp(value, donor[layer][None, :, h], gates[:, j, None, None])
                    parts.append(value)
                return (torch.stack(parts, dim=2).flatten(2),)

            hooks.enter_context(
                model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(substitute)
            )
        tokens = torch.tensor([ids[0]]).expand(len(gates), -1)
        return model(tokens, use_cache=False).logits[:, -1]


class InteractionOracleTests(unittest.TestCase):
    def test_cached_replay_and_gradients_match_native_multi_block_patches(self):
        # Bits are deliberately interleaved by layer, and heads within layers are unsorted.
        sites = [(2, 6), (0, 3), (2, 2), (0, 0), (0, 6), (2, 3), (2, 7), (0, 5)]
        ids = [[1, 7, 5, 4, 9, 3, 2], [1, 9, 5, 2, 16, 3, 9]]
        labels = [8, 4]
        for config_cls, model_cls in ((Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM)):
            with self.subTest(architecture=model_cls.__name__):
                torch.manual_seed(8301)
                cfg = config_cls(
                    vocab_size=64,
                    hidden_size=32,
                    num_attention_heads=8,
                    num_key_value_heads=2,
                    intermediate_size=48,
                    num_hidden_layers=4,
                    initializer_range=0.085,
                )
                cfg._attn_implementation = "eager"
                model = model_cls(cfg).eval().double().requires_grad_(False)
                oracle = Oracle(model, ids, labels, sites)
                bits = torch.arange(8)
                masks = ((torch.arange(256)[:, None] >> bits) & 1).double()
                logits = torch.cat([direct(model, ids, sites, gate) for gate in masks.split(37)])
                margin = logits[:, labels[1]] - logits[:, labels[0]]
                measured = oracle.measure()
                self.assertEqual(measured["top"], logits.argmax(-1).tolist())
                self.assertEqual(measured["clean_top"], int(logits[0].argmax()))
                self.assertEqual(measured["sites"], sites)
                # Some HF versions normalize the residual internally in FP32.
                torch.testing.assert_close(
                    torch.tensor(measured["gold"], dtype=torch.double),
                    margin - margin[0],
                    atol=2e-6,
                    rtol=2e-5,
                )
                ids_to_patch = [253, 0, 1, 253, 117, 19, 255, 5]
                actual = torch.tensor(oracle.exact(ids_to_patch, batch_size=3), dtype=torch.double)
                torch.testing.assert_close(actual, margin[ids_to_patch] - margin[0], atol=2e-6, rtol=2e-5)
                self.assertEqual(oracle.exact([]), [])

                # Perturb each continuous gate at each quadrature node in independent native forwards.
                step = 0.0008
                eye = torch.eye(8, dtype=torch.double) * step
                derivatives = {}
                for point in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9):
                    values = direct(model, ids, sites, torch.cat((point + eye, point - eye)))
                    v = values[:, labels[1]] - values[:, labels[0]]
                    derivatives[point] = (v[:8] - v[8:]) / (2 * step)
                for name, nodes in (
                    ("atp", [0.0]),
                    ("midpoint", [0.5]),
                    ("ig5", [0.1, 0.3, 0.5, 0.7, 0.9]),
                ):
                    grad = torch.tensor(measured[name], dtype=torch.double)
                    expected = torch.stack([derivatives[a] for a in nodes]).mean(0)
                    torch.testing.assert_close(grad, expected, atol=3e-4, rtol=7e-3)
