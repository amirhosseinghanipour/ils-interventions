"""Compare the cached sixteen-head oracle to full native decoder interventions."""

import unittest
from contextlib import ExitStack

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from scripts.panel_scaling.oracle import Oracle


def native_margins(model, ids, labels, sites, gates):
    layers = sorted({site[0] for site in sites})
    if model.config.model_type == "gpt2":
        projections = {layer: model.transformer.h[layer].attn.c_proj for layer in layers}
    else:
        projections = {layer: model.model.layers[layer].self_attn.o_proj for layer in layers}
    donors = {}
    with torch.no_grad(), ExitStack() as stack:
        for layer, projection in projections.items():

            def capture(module, args, layer=layer):
                donors[layer] = args[0]

            stack.enter_context(projection.register_forward_pre_hook(capture))
        model(torch.tensor([ids[1]]), use_cache=False)
    with ExitStack() as stack:
        for layer, projection in projections.items():

            def mix(module, args, layer=layer):
                count = model.config.num_attention_heads
                native = args[0].unflatten(-1, (count, -1))
                donor = donors[layer].unflatten(-1, (count, -1))
                parts = []
                for h in range(count):
                    if (layer, h) in sites:
                        g = gates[:, sites.index((layer, h)), None, None]
                        parts.append(torch.lerp(native[:, :, h], donor[:, :, h], g))
                    else:
                        parts.append(native[:, :, h])
                return (torch.stack(parts, dim=2).flatten(2),)

            stack.enter_context(projection.register_forward_pre_hook(mix))
        logits = model(torch.tensor([ids[0]]).expand(len(gates), -1), use_cache=False).logits[:, -1]
    return logits[:, labels[1]] - logits[:, labels[0]]


class PanelOracleTests(unittest.TestCase):
    def test_exact_and_midpoint_match_native_all_architectures(self):
        ids, labels = [[1, 3, 12, 4, 18, 2], [1, 5, 17, 4, 15, 7]], [8, 24]
        sites = [(layer, h) for h in (6, 7, 2, 4, 5, 0, 9, 3) for layer in (2, 0)]
        kwargs = dict(
            vocab_size=37,
            hidden_size=40,
            intermediate_size=48,
            num_attention_heads=10,
            num_key_value_heads=2,
            num_hidden_layers=4,
            initializer_range=0.08,
        )
        architectures = [
            (
                GPT2LMHeadModel,
                GPT2Config(
                    n_embd=40,
                    n_head=10,
                    n_layer=4,
                    vocab_size=37,
                    n_positions=24,
                    bos_token_id=1,
                    eos_token_id=2,
                ),
            ),
            (LlamaForCausalLM, LlamaConfig(**kwargs)),
            (Qwen2ForCausalLM, Qwen2Config(**kwargs)),
        ]
        for cls, config in architectures:
            with self.subTest(model=cls.__name__):
                torch.manual_seed(78)
                config._attn_implementation = "eager"
                m = cls(config).eval().requires_grad_(False)
                o = Oracle(m, ids, labels, sites)
                masks = [62301, 0, 17, 8192, 65535, 62301, 497, 1, 173]
                gates = ((torch.tensor(masks)[:, None] >> torch.arange(16)) & 1).float()
                expected = native_margins(m, ids, labels, sites, gates) - o.clean_margin
                torch.testing.assert_close(torch.tensor(o.exact(masks, 3)), expected, atol=4e-6, rtol=1e-4)
                self.assertEqual(o.exact([]), [])
                self.assertLess(o.audit(masks), 4e-6)
                midpoint = torch.full((1, 16), 0.5, requires_grad=True)
                effect = native_margins(m, ids, labels, sites, midpoint)
                grad = torch.autograd.grad(effect.sum(), midpoint)[0][0]
                torch.testing.assert_close(torch.tensor(o.gradient()), grad, atol=4e-6, rtol=1e-4)
                raw = m(torch.tensor(ids), use_cache=False).logits[:, -1]
                self.assertEqual([o.clean_top, o.donor_top], raw.argmax(dim=-1).tolist())
                torch.testing.assert_close(
                    torch.tensor([o.clean_margin, o.donor_margin]),
                    raw[:, labels[1]] - raw[:, labels[0]],
                    atol=4e-6,
                    rtol=1e-4,
                )
                with self.assertRaises(ValueError):
                    o.exact([65536])
