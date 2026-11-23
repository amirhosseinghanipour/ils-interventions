"""Evaluate training-free endpoint estimates across every nonzero residual-site patch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ilslens.suffix import DecoderSuffix
from sample import digest

INPUTS = {
    "ioi": ("ioi-test.json", "030354654ea1b0b0147eb07f427be0f1ac71beed63a784a76d1581f52e722ebc"),
    "colors": ("colors-test.jsonl", "4a205a340887c60db7561c79b2e022f7dd3f1bd312a4c77eabd450af7680d545"),
}


def encode(tok, q):
    clean, donor, old, new = q
    a, b = (tok.encode(t, add_special_tokens=False) for t in (clean, donor))
    if len(a) != len(b) or a == b:
        return None
    labels = []
    for suffix in (old, new):
        extended = tok.encode(clean + " " + suffix, add_special_tokens=False)
        check = tok.encode(donor + " " + suffix, add_special_tokens=False)
        if len(extended) != len(a) + 1 or extended[:-1] != a or check != b + [extended[-1]]:
            return None
        labels.append(extended[-1])
    return (a, b, labels) if labels[0] != labels[1] else None


def select(path, task, toks):
    assert digest(path) == INPUTS[task][1]
    data = (
        json.loads(path.read_text())
        if task == "ioi"
        else [json.loads(s) for s in path.read_text().splitlines()]
    )
    rng = np.random.default_rng(538247)
    seen = set()
    chosen = []
    for index in rng.permutation(len(data)):
        row = data[index]
        counter = row["s2_io_flip_counterfactual" if task == "ioi" else "answerPosition_counterfactual"]
        choices = row["choices"] if task == "ioi" else row["choices"]["label"]
        q = (row["prompt"], counter["prompt"], choices[row["answerKey"]], choices[counter["answerKey"]])
        if q[0] in seen or not all(encode(t, q) for t in toks):
            continue
        chosen.append((int(index), q))
        seen.add(q[0])
        if len(chosen) == 50:
            break
    return chosen


def measure(model, ids):
    a, b, labels = ids

    def replay(tokens):
        return DecoderSuffix(model, torch.tensor([tokens], device=model.device))

    r, d = replay(a), replay(b)
    layer = len(r.states) // 2 - 1
    h, k = r.states[layer], d.states[layer]
    move = k - h

    def margin(state):
        output = r.forward(layer, state)[:, -1]
        logits = model.model.norm(output) @ model.lm_head.weight[labels].T
        return logits[:, 1] - logits[:, 0]

    def gradient(x):
        x = x.detach().requires_grad_()
        score = margin(x)
        return score.detach(), torch.autograd.grad(score.sum(), x)[0].detach()

    val, g = gradient(h)
    target, gd = gradient(k)
    _, gm = gradient((h + k) * 0.5)
    end = (target - val).item()
    atp = (g * move).sum(-1)[0]
    slopes = (gd * move).sum(-1)[0]
    reverse = end - slopes.sum() + slopes
    energy = move[0].square().sum(-1)
    active = torch.nonzero(energy > 0).flatten()
    share = energy / energy.sum()
    weight = share.square() / (share.square() + (1 - share).square())
    estimates = {
        "zero": torch.zeros_like(atp),
        "atp": atp,
        "donor": reverse,
        "endpoint_blend": (1 - weight) * atp + weight * reverse,
        "nearest_endpoint": torch.where(share > 0.5, reverse, atp),
        "energy_secant": atp + share * (end - atp.sum()),
        "trapezoid": 0.5 * (atp + slopes),
        "global_midpoint": (gm * move).sum(-1)[0],
    }
    truth = []
    for sites in active.split(8):
        state = h.expand(len(sites), -1, -1).clone()
        state[torch.arange(len(sites), device=h.device), sites] = k[0, sites]
        truth.extend((margin(state) - val).tolist())
    assert len(active) > 0
    # Check that replaying the donor activation through the recipient suffix gives its native margin.
    dlogits = model.lm_head(model.model.norm(d.states[len(d.states) - 1][0, -1]))[labels]
    torch.testing.assert_close(dlogits[1] - dlogits[0], target[0], rtol=1e-4, atol=2e-4)
    return dict(
        clean_margin=val.item(),
        donor_margin=target.item(),
        positions=active.tolist(),
        share=share[active].tolist(),
        gold=truth,
        estimates={k: v[active].tolist() for k, v in estimates.items()},
    )


def summarize(rows):
    draws = np.random.default_rng(38219).integers(len(rows), size=(10000, len(rows)))
    output = {}
    gold = [np.asarray(r["gold"]) for r in rows]
    energy = np.array([np.square(y).sum() for y in gold])
    scores = {}
    for name in rows[0]["estimates"]:
        preds = [np.asarray(r["estimates"][name]) for r in rows]
        sse = np.array([np.square(y - p).sum() for y, p in zip(gold, preds)])
        scores[name] = sse
        output[name] = dict(
            normalized_error=float(sse.sum() / energy.sum()),
            mae=float(sum(abs(p - y).sum() for y, p in zip(gold, preds)) / sum(map(len, gold))),
            top_site_correct=sum(int(np.argmax(abs(y)) == np.argmax(abs(p))) for y, p in zip(gold, preds)),
        )
    for k, err in scores.items():
        diff = err - scores["endpoint_blend"]
        output[k]["gain_with_blend"] = float(diff.sum() / energy.sum())
        output[k]["paired_ci95"] = np.quantile(
            diff[draws].sum(1) / energy[draws].sum(1), [0.025, 0.975]
        ).tolist()
    return dict(
        prompts=len(rows),
        sites=sum(map(len, gold)),
        clean_prefers_reference=sum(r["clean_margin"] < 0 for r in rows),
        donor_prefers_reference=sum(r["donor_margin"] > 0 for r in rows),
        methods=output,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = ("qwen", "smol")
    toks = [AutoTokenizer.from_pretrained(args.inputs / k) for k in names]
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {}
    for name, tok in zip(names, toks):
        m = (
            AutoModelForCausalLM.from_pretrained(
                args.inputs / name, dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            .eval()
            .requires_grad_(False)
            .cuda()
        )
        for task, (filename, _) in INPUTS.items():
            rows = [
                dict(index=i, **measure(m, encode(tok, q)))
                for i, q in select(args.inputs / filename, task, toks)
            ]
            key = name + "-" + task
            (args.out / (key + ".json")).write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
            result[key] = summarize(rows)
            print(key, len(rows), flush=True)
        del m
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
