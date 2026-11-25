"""Prefix-cached native attention-head donor interpolation for Llama and Qwen2.

The gate scales a head at every token immediately before its output projection.
Zero keeps that head's current recipient activation, including changes induced by
earlier interventions. One replaces it with the fixed unmodified donor activation.
All methods share the same float32 two-token readout on native decoder states.
"""

from contextlib import ExitStack
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.panel_scaling.data import encode, make_rows


def load(checkpoint, dtype="float16", device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
    ).to(device).eval().requires_grad_(False)
    if model.config.model_type not in ("llama", "qwen2"):
        raise ValueError(f"unsupported architecture {model.config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    return model, tokenizer


def rows(task, split="heldout"):
    if task not in ("gift", "route") or split not in ("development", "heldout"):
        raise ValueError("expected gift/route and development/heldout")
    return [row for row in make_rows() if row["task"] == task and row["split"] == split]


def sites(model, k):
    """A nested, query-independent panel balanced across five depth strata."""
    depth = model.config.num_hidden_layers
    nheads = model.config.num_attention_heads
    if depth < 10 or k < 1 or k > 5 * nheads:
        raise ValueError("panel exceeds available head sites")
    layers = np.linspace(depth // 2 - 1, depth - 1, 5, dtype=int)
    rng = np.random.default_rng(519743)
    orders = [rng.permutation(nheads).tolist() for _ in layers]
    return [(int(layers[j % 5]), orders[j % 5][j // 5]) for j in range(k)]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare(model, tok, row, k):
    """Measure synchronized prompt tokenization, panel setup, and clean/donor caches."""
    _sync(model.device)
    start = perf_counter()
    ids, labels = encode(tok, row)
    oracle = GateOracle(model, ids, labels, sites(model, k))
    _sync(model.device)
    oracle.common_ms = 1000 * (perf_counter() - start)
    return oracle


class GateOracle:
    @torch.no_grad()
    def __init__(self, model, ids, labels, panel):
        self.model = model
        self.device = model.device
        self.sites = panel
        self.k = len(panel)
        self.decoder = model.model
        self.blocks = self.decoder.layers
        self.first = min(layer for layer, _ in panel)
        self.head_count = model.config.num_attention_heads
        self.ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        self.readout = model.lm_head.weight[labels].float()
        self.donor = {}
        self.args = {}
        self.head_index = {}
        self.heads = {}
        self.bits = {}
        for layer in sorted({layer for layer, _ in panel}):
            pairs = [(head, bit) for bit, (level, head) in enumerate(panel) if layer == level]
            head_to_gate = [self.k] * self.head_count
            for head, bit in pairs:
                head_to_gate[head] = bit
            self.head_index[layer] = torch.tensor(head_to_gate, device=self.device)
            self.heads[layer] = [head for head, _ in pairs]
            self.bits[layer] = [bit for _, bit in pairs]

        with ExitStack() as hooks:
            for level in range(self.first, len(self.blocks)):
                def capture_args(_module, args, kwargs, level=level):
                    if level == self.first:
                        self.prefix = args[0]
                    self.args[level] = args[1:], dict(kwargs)
                hooks.enter_context(self.blocks[level].register_forward_pre_hook(capture_args, with_kwargs=True))
            hidden = self.decoder(self.ids[:1], use_cache=False).last_hidden_state
            self.clean = self._margin(hidden)
        with ExitStack() as hooks:
            for level in self.head_index:
                def capture_donor(_module, args, level=level):
                    self.donor[level] = args[0].unflatten(-1, (self.head_count, -1)).float()
                projection = self.blocks[level].self_attn.o_proj
                hooks.enter_context(projection.register_forward_pre_hook(capture_donor))
            self.decoder(self.ids[1:], use_cache=False)

    def _margin(self, hidden):
        logits = F.linear(hidden[:, -1, :].float(), self.readout)
        return logits[:, 1] - logits[:, 0]

    def _gates(self, masks):
        gates = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        if gates.ndim != 2 or gates.shape[1] != self.k or gates.shape[0] < 1:
            raise ValueError(f"expected a nonempty (batch,{self.k}) gate matrix")
        return gates

    def _effect(self, gates):
        padded = F.pad(gates, (0, 1))
        state = self.prefix.expand(len(gates), -1, -1)
        with ExitStack() as hooks:
            for level, indices in self.head_index.items():
                weights = padded.index_select(1, indices)[:, None, :, None]

                def mix(_module, args, weights=weights, level=level):
                    raw = args[0]
                    heads = raw.unflatten(-1, (self.head_count, -1)).float()
                    mixed = torch.lerp(heads, self.donor[level], weights)
                    return (mixed.flatten(2).to(raw.dtype), *args[1:])

                hooks.enter_context(self.blocks[level].self_attn.o_proj.register_forward_pre_hook(mix))
            for level in range(self.first, len(self.blocks)):
                args, kwargs = self.args[level]
                state = self.blocks[level](state, *args, **kwargs)
                if isinstance(state, tuple):
                    state = state[0]
        return self._margin(self.decoder.norm(state)) - self.clean

    @torch.no_grad()
    def exact(self, masks):
        """Return a CPU float32 array after one batched native suffix forward."""
        return self._effect(self._gates(masks)).cpu().numpy()

    @torch.enable_grad()
    def gradients(self, masks):
        """Return CPU float32 effects and per-row gate gradients in one backward."""
        gates = self._gates(masks).detach().requires_grad_()
        effect = self._effect(gates)
        grad, = torch.autograd.grad(effect.sum(), gates)
        return effect.detach().cpu().numpy(), grad.detach().cpu().numpy()

    @torch.no_grad()
    def audit(self, masks):
        """Compare the cached suffix to full-length torch.where head replacements."""
        gates = self._gates(masks)
        if not torch.all((gates == 0) | (gates == 1)):
            raise ValueError("native replacement audit requires Boolean masks")
        with ExitStack() as hooks:
            for level in self.head_index:
                bits = gates[:, self.bits[level]].bool()[:, None, :, None]
                selected = self.heads[level]

                def replace(_module, args, bits=bits, selected=selected, level=level):
                    raw = args[0].clone()
                    heads = raw.unflatten(-1, (self.head_count, -1))
                    donor = self.donor[level][:, :, selected].to(raw.dtype)
                    heads[:, :, selected] = torch.where(bits, donor, heads[:, :, selected])
                    return (raw, *args[1:])

                hooks.enter_context(self.blocks[level].self_attn.o_proj.register_forward_pre_hook(replace))
            ids = self.ids[:1].expand(len(gates), -1)
            gold = self._margin(self.decoder(ids, use_cache=False).last_hidden_state) - self.clean
        return float((gold - self._effect(gates)).abs().max())

    def kernel_profile(self, batch=4):
        """Dispatch evidence from an actual gradient operation, outside scored time."""
        from torch.profiler import profile, ProfilerActivity
        gates = np.full((batch, self.k), .5, np.float32)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            self.gradients(gates)
        return {
            "dtype": str(self.prefix.dtype),
            "attention_implementation": self.model.config._attn_implementation,
            "batch": batch,
            "operators": {
                entry.key: entry.count for entry in prof.key_averages()
                if any(substr in entry.key.lower() for substr in ("scaled_dot", "flash_attention", "efficient_attention"))
            },
        }
