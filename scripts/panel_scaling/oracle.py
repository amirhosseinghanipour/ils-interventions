"""Two-block interventions on sixteen native attention heads at every token position."""

from collections import defaultdict
from contextlib import ExitStack
from operator import index
from time import perf_counter

import torch
from torch.nn import functional as F


class _AttentionComplete(Exception):
    """End the second block at the input of its MLP normalization."""


class Oracle:
    """Bit i replaces sites[i] with the output of that head on the fixed donor."""

    @torch.no_grad()
    def __init__(self, model, ids, labels, sites):
        self.model = model
        self.device = model.device
        started = self._clock()
        if model.training:
            raise ValueError("model must be in evaluation mode")
        kind = model.config.model_type
        if kind == "gpt2":
            self.decoder = model.transformer
            self.blocks = self.decoder.h
            self.final_norm = self.decoder.ln_f
            self.projections = [block.attn.c_proj for block in self.blocks]
            self.mlp_norms = [block.ln_2 for block in self.blocks]
        elif kind in ("llama", "qwen2"):
            self.decoder = model.model
            self.blocks = self.decoder.layers
            self.final_norm = self.decoder.norm
            self.projections = [block.self_attn.o_proj for block in self.blocks]
            self.mlp_norms = [block.post_attention_layernorm for block in self.blocks]
        else:
            raise ValueError(f"unsupported decoder architecture {kind}")
        self.head_count = model.config.num_attention_heads
        self.sites = [tuple(map(index, site)) for site in sites]
        if len(set(self.sites)) != 16 or len(self.sites) != 16:
            raise ValueError("expected sixteen distinct head sites")
        self.bits = defaultdict(list)
        for bit, (layer, head) in enumerate(self.sites):
            if not 0 <= layer < len(self.blocks) or not 0 <= head < self.head_count:
                raise ValueError("out-of-range head site")
            self.bits[layer].append(bit)
        if len(self.bits) != 2 or any(len(bits) != 8 for bits in self.bits.values()):
            raise ValueError("expected eight heads in each of two blocks")
        self.first, self.second = sorted(self.bits)
        self.heads = {layer: [self.sites[b][1] for b in bits] for layer, bits in self.bits.items()}
        self.weights = {}
        for layer, heads in self.heads.items():
            weight = self.projections[layer].weight
            # GPT-2's Conv1D stores (in, out); nn.Linear stores (out, in).
            weight = weight.T if kind == "gpt2" else weight
            self.weights[layer] = weight.unflatten(1, (self.head_count, -1))[:, heads].contiguous()
        if len(ids) != 2 or not len(ids[0]) or len(ids[0]) != len(ids[1]):
            raise ValueError("recipient and donor need the same nonzero token length")
        self.ids = [torch.as_tensor(x, device=self.device, dtype=torch.long).reshape(1, -1) for x in ids]
        self.labels = list(map(index, labels))
        vocab = model.lm_head.weight.shape[0]
        if (
            len(set(self.labels)) != 2
            or len(self.labels) != 2
            or any(x < 0 or x >= vocab for x in self.labels)
        ):
            raise ValueError("expected two distinct answer token IDs in the model vocabulary")
        self.readout = model.lm_head.weight[self.labels]
        self.block_args = {}
        ends, saved = [], []
        for is_donor, tokens in enumerate(self.ids):
            activations = {}
            with ExitStack() as hooks:
                for layer in self.bits:

                    def capture(module, args, layer=layer):
                        activations[layer] = self._selected(layer, args[0])

                    hooks.enter_context(self.projections[layer].register_forward_pre_hook(capture))
                if not is_donor:

                    def clean_cut(module, args):
                        self.clean_cut = args[0]

                    hooks.enter_context(self.mlp_norms[self.first].register_forward_pre_hook(clean_cut))
                    for layer in range(self.first + 1, len(self.blocks)):

                        def capture_args(module, args, kwargs, layer=layer):
                            self.block_args[layer] = args[1:], dict(kwargs)

                        hooks.enter_context(
                            self.blocks[layer].register_forward_pre_hook(capture_args, with_kwargs=True)
                        )
                final = self.decoder(tokens, use_cache=False).last_hidden_state[:, -1]
                logits = model.lm_head(final)[0]
                ends.append((float(logits[self.labels[1]] - logits[self.labels[0]]), int(logits.argmax())))
            saved.append(activations)
        clean, self.donor = saved
        (self.clean_margin, self.clean_top), (self.donor_margin, self.donor_top) = ends
        diff = self.donor[self.first] - clean[self.first]
        self.first_delta = torch.einsum("btnd,ond->nto", diff, self.weights[self.first])
        self.common_ms = self._clock() - started

    def _clock(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter() * 1000

    def _selected(self, layer, state):
        return state.unflatten(-1, (self.head_count, -1))[:, :, self.heads[layer]]

    def _block(self, layer, state):
        args, kwargs = self.block_args[layer]
        output = self.blocks[layer](state, *args, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def _after(self, layer, cut):
        return cut + self.blocks[layer].mlp(self.mlp_norms[layer](cut))

    def _condition(self, early):
        edit = (early @ self.first_delta.flatten(1)).reshape(-1, *self.clean_cut.shape[1:])
        state = self._after(self.first, self.clean_cut + edit)
        for layer in range(self.first + 1, self.second):
            state = self._block(layer, state)
        saved = {}

        def heads(module, args):
            saved["delta"] = self.donor[self.second] - self._selected(self.second, args[0])

        def cut(module, args):
            saved["cut"] = args[0]
            raise _AttentionComplete

        with ExitStack() as hooks:
            hooks.enter_context(self.projections[self.second].register_forward_pre_hook(heads))
            hooks.enter_context(self.mlp_norms[self.second].register_forward_pre_hook(cut))
            try:
                self._block(self.second, state)
            except _AttentionComplete:
                pass
        return saved["cut"], saved["delta"]

    def _finish(self, cut, delta, late):
        mixed = (delta * late[:, None, :, None]).flatten(2)
        edit = F.linear(mixed, self.weights[self.second].flatten(1))
        state = self._after(self.second, cut + edit)
        for layer in range(self.second + 1, len(self.blocks)):
            state = self._block(layer, state)
        logits = F.linear(self.final_norm(state[:, -1]), self.readout)
        return logits[:, 1] - logits[:, 0] - self.clean_margin

    def _gates(self, masks, bits):
        packed = torch.tensor(masks, device=self.device, dtype=torch.long)
        shifts = torch.tensor(bits, device=self.device, dtype=torch.long)
        return ((packed[:, None] >> shifts) & 1).to(self.clean_cut.dtype)

    def _mask_ids(self, masks):
        ids = [index(value) for value in masks]
        if any(value < 0 or value >= 1 << len(self.sites) for value in ids):
            raise ValueError("mask contains bits outside the panel")
        return ids

    @torch.no_grad()
    def exact(self, masks, chunk=16):
        """Compute centered donor-minus-original margins in input order, including repeats."""
        masks = self._mask_ids(masks)
        chunk = index(chunk)
        if chunk < 1:
            raise ValueError("chunk must be positive")
        # Reuse each early state within this call only. Both conditioning and suffix
        # lanes are bounded by chunk, including when there are 256 distinct early masks.
        early_bits = self.bits[self.first]
        early_mask = sum(1 << bit for bit in early_bits)
        groups = defaultdict(list)
        for i, mask in enumerate(masks):
            groups[mask & early_mask].append(i)
        keys = list(groups)
        values = self.clean_cut.new_empty(len(masks))
        for offset in range(0, len(keys), chunk):
            batch = keys[offset : offset + chunk]
            cut, delta = self._condition(self._gates(batch, early_bits))
            rows = [(lane, i) for lane, key in enumerate(batch) for i in groups[key]]
            for start in range(0, len(rows), chunk):
                lanes, indices = zip(*rows[start : start + chunk])
                lanes = torch.tensor(lanes, device=self.device)
                dest = torch.tensor(indices, device=self.device)
                late = self._gates([masks[i] for i in indices], self.bits[self.second])
                values[dest] = self._finish(cut[lanes], delta[lanes], late)
        return values.tolist()

    @torch.enable_grad()
    def gradient(self):
        """Margin derivative with respect to every gate at the all-half intervention."""
        gates = self.clean_cut.new_full((1, len(self.sites)), 0.5, requires_grad=True)
        cut, delta = self._condition(gates[:, self.bits[self.first]])
        effect = self._finish(cut, delta, gates[:, self.bits[self.second]])
        return torch.autograd.grad(effect.sum(), gates)[0][0].tolist()

    @torch.no_grad()
    def audit(self, masks):
        """Maximum absolute difference from full-model native head replacements on these masks."""
        masks = self._mask_ids(masks)
        if not masks:
            raise ValueError("native audit needs at least one mask")
        errors = []
        for offset in range(0, len(masks), 16):
            batch = masks[offset : offset + 16]
            with ExitStack() as hooks:
                for layer, bits in self.bits.items():
                    selected = self._gates(batch, bits).bool()

                    def patch(module, args, layer=layer, selected=selected):
                        state = args[0].clone()
                        heads = state.unflatten(-1, (self.head_count, -1))
                        indices = self.heads[layer]
                        heads[:, :, indices] = torch.where(
                            selected[:, None, :, None], self.donor[layer], heads[:, :, indices]
                        )
                        return (state, *args[1:])

                    hooks.enter_context(self.projections[layer].register_forward_pre_hook(patch))
                tokens = self.ids[0].expand(len(batch), -1)
                hidden = self.decoder(tokens, use_cache=False).last_hidden_state[:, -1]
                native = self.model.lm_head(hidden)
            truth = native[:, self.labels[1]] - native[:, self.labels[0]] - self.clean_margin
            errors.append((truth - truth.new_tensor(self.exact(batch))).abs().max())
        return float(torch.stack(errors).max())
