"""Exact two-block donor substitutions and derivatives of the same gated intervention."""

from collections import defaultdict
from contextlib import ExitStack
from operator import index
from time import perf_counter

import torch
from torch.nn import functional as F


def _state(output):
    return output[0] if isinstance(output, tuple) else output


class Oracle:
    """Bit i patches sites[i] over every prompt position using the fixed native donor."""

    @torch.no_grad()
    def __init__(self, model, ids, labels, sites):
        self.model = model
        self.decoder = model.model
        self.layers = self.decoder.layers
        self.device = model.device
        start = self._clock()
        self.head_count = model.config.num_attention_heads
        self.sites = [tuple(map(index, site)) for site in sites]
        if len(self.sites) != 8 or len(set(self.sites)) != 8:
            raise ValueError("provide eight distinct (layer, head) sites")
        positions = defaultdict(list)
        for bit, (layer, head) in enumerate(self.sites):
            if not (0 <= layer < len(self.layers) and 0 <= head < self.head_count):
                raise ValueError("panel contains an out-of-range layer or head")
            positions[layer].append(bit)
        if len(positions) != 2 or any(len(bits) != 4 for bits in positions.values()):
            raise ValueError("provide four head sites from each of two blocks")
        if model.training:
            raise ValueError("interventions require a model in evaluation mode")
        self.first, self.second = sorted(positions)
        self.bits = dict(positions)
        self.heads = {layer: [self.sites[b][1] for b in bits] for layer, bits in self.bits.items()}
        if len(ids) != 2 or not len(ids[0]) or len(ids[0]) != len(ids[1]):
            raise ValueError("recipient and donor require the same nonzero input length")
        self.ids = [torch.as_tensor(x, device=self.device, dtype=torch.long).reshape(1, -1) for x in ids]
        self.labels = list(map(index, labels))
        if len(self.labels) != 2 or self.labels[0] == self.labels[1]:
            raise ValueError("provide distinct recipient and donor answer token ids")
        self.adaptor = self.model.lm_head.weight[self.labels]
        self.kwargs = {}
        self.donor = {}
        ends = []
        for donor, tokens in enumerate(self.ids):
            raw = {}
            with ExitStack() as hooks:
                for layer in self.bits:

                    def record_heads(module, args, layer=layer):
                        raw[layer] = self._selected(layer, args[0])

                    hooks.enter_context(
                        self.layers[layer].self_attn.o_proj.register_forward_pre_hook(record_heads)
                    )
                if not donor:

                    def first_cut(module, args):
                        self.clean_cut = args[0]

                    hooks.enter_context(
                        self.layers[self.first].post_attention_layernorm.register_forward_pre_hook(
                            first_cut
                        )
                    )
                    for layer in range(self.first + 1, len(self.layers)):

                        def record_kwargs(module, args, kwargs, layer=layer):
                            if kwargs.get("past_key_values") is not None:
                                raise ValueError("suffix evaluation requires use_cache=False")
                            self.kwargs[layer] = kwargs.copy()

                        hooks.enter_context(
                            self.layers[layer].register_forward_pre_hook(record_kwargs, with_kwargs=True)
                        )
                out = self.decoder(tokens, use_cache=False).last_hidden_state[:, -1]
                logits = self.model.lm_head(out)[0]
                ends.append((float(logits[self.labels[1]] - logits[self.labels[0]]), int(logits.argmax())))
            if donor:
                self.donor = raw
            else:
                clean_heads = raw[self.first]
        self.first_delta = self._project(self.first, self.donor[self.first] - clean_heads)[0]
        (self.clean_margin, self.clean_top), (self.donor_margin, self.donor_top) = ends
        self.common_ms = self._clock() - start

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter() * 1000.0

    def _selected(self, layer, state):
        return state.unflatten(-1, (self.head_count, -1))[:, :, self.heads[layer], :]

    def _project(self, layer, head_diff):
        w = self.layers[layer].self_attn.o_proj.weight.unflatten(-1, (self.head_count, -1))
        return torch.einsum("btnd,ond->bnto", head_diff, w[:, self.heads[layer]])

    def _after_attention(self, layer, cut):
        block = self.layers[layer]
        return cut + block.mlp(block.post_attention_layernorm(cut))

    def _condition(self, early):
        # Only the first block admits precomputed edit vectors for all masks.
        cut = self.clean_cut + (early @ self.first_delta.flatten(1)).reshape(
            len(early), *self.clean_cut.shape[1:]
        )
        state = self._after_attention(self.first, cut)
        for layer in range(self.first + 1, self.second):
            state = _state(self.layers[layer](state, **self.kwargs[layer]))
        current = {}
        block = self.layers[self.second]

        def heads(module, args):
            current["heads"] = self._selected(self.second, args[0])

        def residual(module, args):
            current["cut"] = args[0]

        with ExitStack() as hooks:
            hooks.enter_context(block.self_attn.o_proj.register_forward_pre_hook(heads))
            hooks.enter_context(block.post_attention_layernorm.register_forward_pre_hook(residual))
            block(state, **self.kwargs[self.second])
        # Later substitutions replace the *current* recipient head, conditional on early gates.
        delta = self._project(self.second, self.donor[self.second] - current["heads"])
        return current["cut"], delta

    def _finish(self, cut, delta, late):
        mixed = cut + torch.einsum("bn,bntd->btd", late, delta)
        state = self._after_attention(self.second, mixed)
        for layer in range(self.second + 1, len(self.layers)):
            state = _state(self.layers[layer](state, **self.kwargs[layer]))
        return self.decoder.norm(state[:, -1])

    def _mask_gates(self, ids):
        masks = torch.tensor(ids, dtype=torch.long, device=self.device)
        bits = torch.arange(len(self.sites), device=self.device)
        return ((masks[:, None] >> bits) & 1).to(self.clean_cut.dtype)

    def _margin(self, hidden):
        logits = F.linear(hidden, self.adaptor)
        return logits[:, 1] - logits[:, 0]

    @torch.enable_grad()
    def gradient(self, name):
        """Eight derivatives w.r.t. gate values at the recipient, midpoint or IG5 nodes."""
        nodes = {"atp": (0.0,), "midpoint": (0.5,), "ig5": (0.1, 0.3, 0.5, 0.7, 0.9)}
        if name not in nodes:
            raise ValueError(f"unknown gradient method: {name}")
        # The five IG nodes are independent batch lanes. Differentiating the
        # sum returns a separate gate derivative for each node in one backward pass.
        points = self.clean_cut.new_tensor(nodes[name])
        gates = points[:, None].expand(-1, 8).clone().requires_grad_()
        cut, delta = self._condition(gates[:, self.bits[self.first]])
        hidden = self._finish(cut, delta, gates[:, self.bits[self.second]])
        derivatives = torch.autograd.grad(self._margin(hidden).sum(), gates)[0]
        return derivatives.mean(0).tolist()

    @torch.no_grad()
    def _evaluate(self, mask_ids, *, tops, batch_size=64):
        ids = [index(i) for i in mask_ids]
        if any(i < 0 or i >= 256 for i in ids) or batch_size < 1:
            raise ValueError("masks must be in [0, 255] and batch size must be positive")
        if not ids:
            return [], []
        # Each early mask needs one intervening-layer evaluation within this call.
        # Outputs and conditioned states are never cached between queries.
        early_ids = [sum(((m >> bit) & 1) << j for j, bit in enumerate(self.bits[self.first])) for m in ids]
        unique = sorted(set(early_ids))
        early = self._mask_gates(unique)[:, :4]
        cut, delta = self._condition(early)
        locations = {mask: n for n, mask in enumerate(unique)}
        lanes = torch.tensor([locations[m] for m in early_ids], device=self.device)
        late = self._mask_gates(ids)[:, self.bits[self.second]]
        values, tokens = [], []
        for lane, gates in zip(lanes.split(batch_size), late.split(batch_size), strict=True):
            hidden = self._finish(cut[lane], delta[lane], gates)
            if tops:
                logits = self.model.lm_head(hidden)
                tokens.append(logits.argmax(-1))
                values.append(logits[:, self.labels[1]] - logits[:, self.labels[0]])
            else:
                values.append(self._margin(hidden))
        effects = (torch.cat(values) - self.clean_margin).tolist()
        return effects, torch.cat(tokens).tolist() if tops else []

    def exact(self, mask_ids, batch_size=64):
        """Exact signed donor-minus-recipient margin changes, aligned to mask_ids."""
        return self._evaluate(mask_ids, tops=False, batch_size=batch_size)[0]

    @torch.no_grad()
    def _native(self, mask_ids):
        """Independent full decoder replay using Boolean substitution before o_proj."""
        gates = self._mask_gates(mask_ids).bool()
        with ExitStack() as hooks:
            for layer in self.bits:

                def replace(module, args, layer=layer):
                    state = args[0].clone()
                    as_heads = state.unflatten(-1, (self.head_count, -1))
                    # Construct the replacement without the projected-delta approximation.
                    selected = as_heads[:, :, self.heads[layer], :]
                    as_heads[:, :, self.heads[layer], :] = torch.where(
                        gates[:, None, self.bits[layer], None], self.donor[layer], selected
                    )
                    return (state,)

                hooks.enter_context(self.layers[layer].self_attn.o_proj.register_forward_pre_hook(replace))
            out = self.decoder(self.ids[0].expand(len(mask_ids), -1), use_cache=False).last_hidden_state[
                :, -1
            ]
            logits = self.model.lm_head(out)
        return logits

    def measure(self):
        """Exhaustive signed effects and full-vocab top tokens, plus fresh reference timings."""
        times = {"common": self.common_ms}
        result = dict(
            sites=self.sites,
            clean_margin=self.clean_margin,
            donor_margin=self.donor_margin,
            clean_top=self.clean_top,
            donor_top=self.donor_top,
        )
        for method in ("atp", "midpoint", "ig5"):
            start = self._clock()
            result[method] = self.gradient(method)
            times[method] = self._clock() - start
        start = self._clock()
        result["gold"], result["top"] = self._evaluate(range(256), tops=True)
        times["exact"] = self._clock() - start
        # Distinguish the full-vocabulary label pass from the optimized exact-margin baseline.
        audit_ids = (0, 1, 16, 85, 170, 255)
        native = self._native(audit_ids)
        expected = native[:, self.labels[1]] - native[:, self.labels[0]] - self.clean_margin
        actual = native.new_tensor([result["gold"][j] for j in audit_ids])
        torch.testing.assert_close(actual, expected, atol=8e-4, rtol=3e-4)
        native_top = native.max(-1).values
        chosen = native[
            torch.arange(len(audit_ids), device=self.device), [result["top"][i] for i in audit_ids]
        ]
        torch.testing.assert_close(chosen, native_top, atol=8e-4, rtol=3e-5)
        result["native_audit_max_abs"] = float((actual - expected).abs().max())
        result["times_ms"] = times
        return result
