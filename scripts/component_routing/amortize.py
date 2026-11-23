"""Time a reusable within-block predictor against batched native patch queries."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .benchmark import now, save
    from .engine import Circuit
    from .groups import data, patches, read
    from .sampling import WEIGHTS, encode
else:
    from benchmark import now, save
    from engine import Circuit
    from groups import data, patches, read
    from sampling import WEIGHTS, encode


def masks_for(heads, seed, n=128):
    rng = np.random.default_rng(seed)
    masks = []
    seen = set()
    sizes = (1, heads // 4, heads // 2, heads - heads // 4, heads - 1)
    # Exhausted singleton and all-but-one sets roll forward to the next interior size.
    i = 0
    while len(masks) < n:
        size = sizes[i % len(sizes)]
        i += 1
        which = tuple(sorted(rng.choice(heads, size=size, replace=False).tolist()))
        if which not in seen:
            m = np.zeros(heads, dtype=bool)
            m[list(which)] = True
            masks.append(m)
            seen.add(which)
    return np.array(masks)


def predict(c, masks, method):
    layer = c.levels[1]
    h = c.ends[0][0][layer, "head"]
    heads = next(g[2] for g in c.groups if g[:2] == (layer, "head"))
    end = h + heads.sum(dim=0, keepdim=True)
    features = heads.flatten(1)
    m = masks.float()

    def tangent(state):
        v = state.detach().requires_grad_()
        value = c.tail(layer, "head", v)
        grad = torch.autograd.grad(value.sum(), v)[0]
        return value.detach() - c.outputs[0], features @ grad.reshape(-1)

    if method == "recipient":
        return m @ tangent(h)[1]
    if method == "midpoint":
        return m @ tangent((h + end) * 0.5)[1]
    if method == "secant_midpoint":
        grad = tangent((h + end) * 0.5)[1]
        full = c.tail(layer, "head", end) - c.outputs[0]
        ab = (end - h).reshape(-1)
        alpha = m @ (features @ ab) / ab.square().sum().clamp_min(1e-30)
        return m @ grad + alpha * (full - grad.sum())

    fwd = m @ tangent(h)[1]
    full, grad = tangent(end)
    rev = full + (m - 1) @ grad
    gram = features @ features.T
    left = ((m @ gram) * m).sum(1).clamp_min(0).square()
    right = (((1 - m) @ gram) * (1 - m)).sum(1).clamp_min(0).square()
    weight = left / (left + right).clamp_min(1e-30)
    return torch.where(left > right, rev, fwd) if method == "nearest" else fwd + weight * (rev - fwd)


def timed(c, masks, method):
    start = now()
    if method == "exact":
        out = torch.cat([patches(c, c.levels[1], batch) for batch in masks.split(8)])
    else:
        out = predict(c, masks, method)
    values = out.detach().cpu().double().numpy()
    return now() - start, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("manifest", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("model", choices=WEIGHTS)
    args = p.parse_args()
    path = args.inputs / args.model
    with (path / "model.safetensors").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHTS[args.model]
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
    runs = []
    methods = ("exact", "recipient", "midpoint", "secant_midpoint", "nearest", "smooth")
    for task, refs in read(args.manifest).items():
        rows = data(args.inputs, task + "-test")
        for n, ref in enumerate(refs[:2]):
            ids, labels = encode(tok, rows[ref["index"]], ref["counter"])
            start = now()
            c = Circuit(model, ids, labels)
            common = now() - start
            seed = 731955 + n
            all_masks = torch.tensor(masks_for(c.heads, seed), device=c.model.device)
            record = dict(task=task, **ref, seed=seed, common_ms=common, batches={})
            for k in (1, 8, 32, 128):
                mask = all_masks[:k]
                times = {m: [] for m in methods}
                predictions = {}
                for rep in range(4):
                    for m in np.random.default_rng(445761 + 10 * k + rep + n).permutation(methods):
                        ms, v = timed(c, mask, m)
                        if rep:
                            times[m].append(ms)
                        predictions[m] = v.tolist()
                record["batches"][k] = dict(times=times, values=predictions)
            runs.append(record)
            print(args.model, task, n + 1, flush=True)
    save(args.dest, runs)


if __name__ == "__main__":
    main()
