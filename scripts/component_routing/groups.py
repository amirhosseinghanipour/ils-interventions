"""Predict native groups of attention-head patches from two reachable endpoints."""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import save
from engine import Circuit
from sampling import FILES, WEIGHTS, encode


def read(path):
    return json.loads(path.read_text())


def data(inputs, split):
    source = (inputs / (split + ".jsonl")).read_bytes()
    assert hashlib.sha256(source).hexdigest() == FILES[split]
    return [json.loads(s) for s in source.splitlines()]


def select(inputs, primary):
    toks = [AutoTokenizer.from_pretrained(inputs / key) for key in WEIGHTS]
    used = set()
    rng = np.random.default_rng(351968)
    result = {}

    def operands(task, row, counter):
        return {(task, str(x["operand1"]), str(x["operand2"])) for x in (row, row[counter])}

    for split, refs in read(primary).items():
        rows = data(inputs, split)
        for ref in refs:
            used.update(operands(split.split("-")[0], rows[ref["index"]], ref["counter"]))
    for task in ("addition", "subtraction"):
        rows = data(inputs, task + "-test")
        chosen = []
        for i in rng.permutation(len(rows)):
            r = rows[i]
            counter = ("ones_op1_counterfactual", "tens_op2_counterfactual")[len(chosen) % 2]
            if not r.get(counter):
                continue
            pair = operands(task, r, counter)
            if used.isdisjoint(pair) and all(encode(t, r, counter) for t in toks):
                used.update(pair)
                chosen.append(dict(index=int(i), counter=counter, template=r["template"]))
            if len(chosen) == 24:
                break
        assert len(chosen) == 24
        result[task] = chosen
    return result


@torch.no_grad()
def patches(c, layer, masks):
    def replace(module, args):
        t = args[0]
        dest = t.reshape(*t.shape[:2], c.heads, -1)
        donor = c.ends[1][1][layer, "head"].reshape(1, -1, c.heads, dest.shape[-1])
        return (torch.where(masks[:, None, :, None], donor, dest).reshape_as(t),)

    with ExitStack() as hooks:
        hooks.enter_context(c.layers[layer].self_attn.o_proj.register_forward_pre_hook(replace))
        out = c.decoder(c.ids[0].expand(len(masks), -1), use_cache=False).last_hidden_state[:, -1]
    logits = c.model.lm_head(out)
    return logits[:, c.labels[1]] - logits[:, c.labels[0]] - c.outputs[0]


def experiment(c, seed):
    layer = c.levels[1]
    _, _, heads = next(g for g in c.groups if g[:2] == (layer, "head"))
    half, q = c.heads // 2, c.heads // 4
    sizes = [0, 1, q, half, c.heads - q, c.heads - 1, c.heads]
    order = np.random.default_rng(seed).permutation(c.heads)
    masks = torch.tensor(np.argsort(order)[None, :] < np.array(sizes)[:, None], device=heads.device)
    delta = torch.einsum("bh,htd->btd", masks.float(), heads)
    h = c.ends[0][0][layer, "head"]
    k = c.ends[1][0][layer, "head"]
    end = h + delta[-1:]
    total = (c.tail(layer, "head", end) - c.outputs[0]).item()

    def grad(state):
        v = state.detach().requires_grad_()
        return torch.autograd.grad(c.tail(layer, "head", v).sum(), v)[0]

    def dot(g, d):
        return (g * d).sum((1, 2))

    g0 = grad(h)
    g1 = grad(end)
    gm = grad((h + end) * 0.5)
    forward = dot(g0, delta)
    mid = dot(gm, delta)
    reverse = total + dot(g1, delta - delta[-1:])
    alpha = dot(delta[-1:], delta) / delta[-1:].square().sum().clamp_min(1e-30)

    def weight(displacement):
        a = dot(delta, delta).square()
        rest = delta - displacement
        b = dot(rest, rest).square()
        return a / (a + b).clamp_min(1e-30)

    w = weight(delta[-1:])
    natural_mid = dot(grad((h + k) * 0.5), delta)
    natural_reverse = c.outputs[1] - c.outputs[0] + dot(grad(k), delta - (k - h))
    preds = dict(
        recipient=forward,
        natural_midpoint=natural_mid,
        natural_ils=natural_mid + 0.25 * weight(k - h) * (natural_reverse - natural_mid),
        matched_midpoint=mid,
        secant_only=alpha * total,
        secant_midpoint=mid + alpha * (total - mid[-1]),
        matched_reverse=reverse,
        nearest=torch.where(w > 0.5, reverse, forward),
        smooth=forward + w * (reverse - forward),
    )
    # Independent labels physically replace slices of the attention output projection's input.
    exact = patches(c, layer, masks)
    cut = c.tail(layer, "head", h + delta) - c.outputs[0]
    torch.testing.assert_close(cut, exact, rtol=3e-4, atol=6e-4)
    return dict(
        order=order.tolist(),
        sizes=sizes,
        layer=layer,
        gold=exact.tolist(),
        natural_distance=float((k - h).norm()),
        matched_distance=float(delta[-1].norm()),
        matched_endpoint_effect=total,
        max_native_cut_error=float((cut - exact).abs().max()),
        weight=w.tolist(),
        predictions={k: v.tolist() for k, v in preds.items()},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("action", choices=["select", *WEIGHTS])
    args = p.parse_args()
    if args.action == "select":
        save(args.dest, select(args.inputs, args.manifest))
        return
    path = args.inputs / args.action
    with (path / "model.safetensors").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHTS[args.action]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )
    tok = AutoTokenizer.from_pretrained(path)
    results = {}
    for task, refs in read(args.manifest).items():
        rows = data(args.inputs, task + "-test")
        results[task] = []
        for n, ref in enumerate(refs):
            tokens, labels = encode(tok, rows[ref["index"]], ref["counter"])
            circuit = Circuit(model, tokens, labels)
            results[task].append(dict(**ref, **experiment(circuit, 34399 + n)))
        print(args.action, task, len(refs), flush=True)
    save(args.dest, results)


if __name__ == "__main__":
    main()
