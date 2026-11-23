"""Native replacements and additive cut representations of whole-head/MLP patches."""

from contextlib import ExitStack

import numpy as np
import torch
from torch.nn import functional as F


class Circuit:
    def __init__(self, model, ids, labels):
        self.model = model
        self.decoder = model.model
        self.layers = self.decoder.layers
        self.levels = [len(self.layers) * i // 4 - 1 for i in (1, 2, 3)]
        self.ids = [torch.tensor([x], device=model.device) for x in ids]
        self.labels = labels
        self.heads = model.config.num_attention_heads
        self.keys = [(i, j) for i in self.levels for j in range(self.heads + 1)]
        self.ends = []
        self.outputs = []
        self.kw = {}
        self.top = []
        for tokens in self.ids:
            with torch.no_grad(), ExitStack() as hooks:
                cuts, components = {}, {}
                for i in self.levels:
                    block = self.layers[i]

                    def head_capture(module, args, i=i):
                        components[i, "head"] = args[0]

                    def attn_cut(module, args, i=i):
                        cuts[i, "head"] = args[0]

                    def mlp_capture(module, args, result, i=i):
                        components[i, "mlp"] = result

                    def block_cut(module, args, result, i=i):
                        cuts[i, "mlp"] = tensor(result)

                    hooks.enter_context(block.self_attn.o_proj.register_forward_pre_hook(head_capture))
                    hooks.enter_context(block.post_attention_layernorm.register_forward_pre_hook(attn_cut))
                    hooks.enter_context(block.mlp.register_forward_hook(mlp_capture))
                    hooks.enter_context(block.register_forward_hook(block_cut))
                for i, block in enumerate(self.layers):

                    def get_kw(module, args, kwargs, i=i):
                        assert kwargs.get("past_key_values") is None
                        self.kw[i] = kwargs.copy()

                    hooks.enter_context(block.register_forward_pre_hook(get_kw, with_kwargs=True))
                out = self.decoder(tokens, use_cache=False).last_hidden_state[:, -1]
                logits = self.model.lm_head(out)
                self.top.append(int(logits.argmax()))
                self.outputs.append(float(logits[0, labels[1]] - logits[0, labels[0]]))
                self.ends.append((cuts, components))
        self.groups = []
        for i in self.levels:
            diff = self.ends[1][1][i, "head"] - self.ends[0][1][i, "head"]
            w = self.layers[i].self_attn.o_proj.weight
            head_dim = w.shape[1] // self.heads
            delta = torch.einsum(
                "nthd,ohd->hnto",
                diff.reshape(1, -1, self.heads, head_dim),
                w.reshape(w.shape[0], self.heads, head_dim),
            )[:, 0]
            self.groups.append((i, "head", delta))
            self.groups.append((i, "mlp", self.ends[1][1][i, "mlp"] - self.ends[0][1][i, "mlp"]))

    def margin(self, residual):
        logits = F.linear(self.decoder.norm(residual[:, -1]), self.model.lm_head.weight[self.labels])
        return logits[:, 1] - logits[:, 0]

    def tail(self, i, kind, state):
        if kind == "head":
            block = self.layers[i]
            state = state + block.mlp(block.post_attention_layernorm(state))
        for j in range(i + 1, len(self.layers)):
            state = tensor(self.layers[j](state, **self.kw[j]))
        return self.margin(state)

    def global_grad(self, fraction):
        embeds = [self.decoder.embed_tokens(x).detach() for x in self.ids]
        start = torch.lerp(*embeds, fraction).requires_grad_()
        cuts = {}
        with ExitStack() as hooks:
            for i in self.levels:
                block = self.layers[i]

                def attn(module, args, i=i):
                    cuts[i, "head"] = args[0]

                def post(module, args, result, i=i):
                    cuts[i, "mlp"] = tensor(result)

                hooks.enter_context(block.post_attention_layernorm.register_forward_pre_hook(attn))
                hooks.enter_context(block.register_forward_hook(post))
            out = self.decoder(inputs_embeds=start, use_cache=False).last_hidden_state[:, -1]
            score = F.linear(out, self.model.lm_head.weight[self.labels])
            order = [(i, k) for i, k, _ in self.groups]
            grads = torch.autograd.grad((score[:, 1] - score[:, 0]).sum(), [cuts[key] for key in order])
        return grads

    def estimates(self, name, strength):
        donor = self.global_grad(1.0) if name == "ils" else None
        global_grads = (
            self.global_grad(0.0 if name == "atp" else 0.5) if name in ("atp", "input_midpoint") else None
        )
        vals = []
        for n, (i, k, delta) in enumerate(self.groups):
            a, b = (e[0][i, k] for e in self.ends)
            if global_grads is None:
                mid = ((a + b) * 0.5).detach().requires_grad_()
                g = torch.autograd.grad(self.tail(i, k, mid).sum(), mid)[0]
            else:
                g = global_grads[n]
            m = (g * delta).sum((1, 2))
            if donor is not None:
                rest = delta - (b - a)
                rev = self.outputs[1] - self.outputs[0] + (donor[n] * rest).sum((1, 2))
                patch_energy = delta.square().sum((1, 2)).square()
                w = patch_energy / (patch_energy + rest.square().sum((1, 2)).square()).clamp_min(1e-30)
                m = m + strength * w * (rev - m)
            vals.append(m)
        return torch.cat(vals).detach().cpu().numpy().astype(float)

    @torch.no_grad()
    def exact(self, indices, *, tops=False):
        # A native donor activation replaces one component per independent batch lane.
        with ExitStack() as hooks:
            for i in self.levels:
                heads, mlps = [], []
                for lane, index in enumerate(indices):
                    level, j = self.keys[index]
                    if level == i:
                        if j == self.heads:
                            mlps.append(lane)
                        else:
                            heads.append((lane, j))
                if heads:

                    def replace_head(module, args, i=i, heads=heads):
                        state = args[0].clone()
                        dest = state.reshape(*state.shape[:2], self.heads, -1)
                        source = self.ends[1][1][i, "head"].reshape(1, -1, self.heads, dest.shape[-1])
                        for lane, j in heads:
                            dest[lane, :, j] = source[0, :, j]
                        return (state,)

                    hooks.enter_context(
                        self.layers[i].self_attn.o_proj.register_forward_pre_hook(replace_head)
                    )
                if mlps:

                    def replace_mlp(module, args, out, i=i, mlps=mlps):
                        state = out.clone()
                        state[mlps] = self.ends[1][1][i, "mlp"]
                        return state

                    hooks.enter_context(self.layers[i].mlp.register_forward_hook(replace_mlp))
            out = self.decoder(self.ids[0].expand(len(indices), -1), use_cache=False).last_hidden_state[
                :, -1
            ]
            logits = F.linear(out, self.model.lm_head.weight[self.labels])
            effects = (logits[:, 1] - logits[:, 0] - self.outputs[0]).cpu().numpy().astype(float)
            if tops:
                return effects, self.model.lm_head(out).argmax(-1).tolist()
            return effects

    def audit(self):
        for i, k, d in self.groups:
            a, b = (e[0][i, k] for e in self.ends)
            for state, val in zip((a, b), self.outputs):
                assert np.isclose(self.tail(i, k, state).item(), val, atol=4e-4, rtol=2e-5)
            j = self.heads if k == "mlp" else 0
            gold = self.exact([self.keys.index((i, j))])[0]
            pred = self.tail(i, k, a + d[:1]).item() - self.outputs[0]
            assert np.isclose(pred, gold, rtol=1e-4, atol=5e-4), (i, k, pred, gold)


def tensor(out):
    return out[0] if isinstance(out, tuple) else out
