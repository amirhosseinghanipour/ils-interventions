"""Time marginal first-position predictions from identical cached clean trajectories."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from anchors import cases
from ilslens.finite_response import partial_response
from ilslens.suffix import DecoderSuffix


def measure(model, ids, weights):
    clean, donor, site = ids

    def replay(side):
        return DecoderSuffix(model, torch.tensor([side], device=model.device))

    r, d = replay(clean), replay(donor)
    depth = len(r.states)
    layer, mid, last = depth // 2 - 1, 3 * depth // 4 - 1, depth - 1
    h, k = r.states[layer], d.states[layer]
    delta = torch.zeros_like(h)
    delta[:, site] = k[:, site] - h[:, site]
    hybrid = h + delta
    complement = hybrid - k
    base = r.states[last][0, -1:]
    end = d.states[last][0, -1:] - base
    z = (d.states[mid] - r.states[mid])[0, -1:]

    def tail(state):
        return r.forward(layer, state)[0, -1:]

    def jvp(at, move):
        return torch.autograd.functional.jvp(tail, at, move)[1]

    def landmark():
        a, b = weights["donor_landmark"]
        return a * end + b * (partial_response(r, hybrid, layer, mid, len(clean) - 1) - z)

    def reverse():
        a, b = weights["donor_jvp"]
        return a * end + b * jvp(k, complement)

    def both():
        a, b, c = weights["two_anchor"]
        return a * end + b * jvp(k, complement) + c * jvp(h, delta)

    ops = {
        "exact": lambda: tail(hybrid) - base,
        "donor_landmark": landmark,
        "donor_jvp": reverse,
        "two_anchor": both,
    }
    times = {k: [] for k in ops}
    for k in ops:
        ops[k]()
    for rep in range(5):
        order = list(ops)
        for k in order[rep % len(ops) :] + order[: rep % len(ops)]:
            torch.cuda.synchronize()
            begin = time.perf_counter()
            ops[k]()
            torch.cuda.synchronize()
            times[k].append(1000 * (time.perf_counter() - begin))
    return {k: float(np.median(v)) for k, v in times.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("inputs", "manifest", "weights", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    articles = {r["title"]: r for r in json.loads((a.inputs / "train-v1.1.json").read_text())["data"]}
    rows = [r for r in json.loads(a.manifest.read_text()) if r["split"] == "test"][:8]
    w = json.loads(a.weights.read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    timings = {}
    for name in ("qwen", "smol"):
        path = a.inputs / name
        tok = AutoTokenizer.from_pretrained(path)
        m = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            .eval()
            .requires_grad_(False)
            .cuda()
        )
        timings[name] = [dict(**meta, ms=measure(m, ids, w)) for meta, ids in cases(rows, articles, tok)]
        del m
    a.out.write_text(json.dumps(timings, indent=2, allow_nan=False) + "\n")
