"""Measure finite source edits and their predictors on a fixed article manifest."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ilslens.finite_response import partial_response, secant_transport
from ilslens.hooks import tensor_output
from ilslens.suffix import DecoderSuffix
from sample import SQUAD_SHA, digest, get_ids

WEIGHTS = {"qwen": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
           "smol": "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"}


@torch.no_grad()
def greedy(model, tokens, stops, intervention=None):
    answer = []
    def hook(module, args, out):
        # Only used for an independent decoding check.
        hidden = tensor_output(out).clone()
        hidden[:, site] += delta
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

    handle = None
    if intervention is not None:
        layer, site, delta = intervention
        handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for _ in range(8):
            ids = torch.tensor([tokens + answer], device=model.device)
            token = int(model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].argmax())
            answer.append(token)
            if token in stops:
                break
    finally:
        if handle is not None:
            handle.remove()
    return answer


def collect(model, clean, donor, site, stops, audit):
    answer = greedy(model, clean, stops)
    start = len(clean) - 1
    def replay(tokens):
        ids = torch.tensor([tokens + answer[:-1]], device=model.device)
        return DecoderSuffix(model, ids)
    r, d = replay(clean), replay(donor)
    depth = len(model.model.layers)
    layer, landmark = depth // 2 - 1, 3 * depth // 4 - 1
    h = r.states[layer]
    delta = torch.zeros_like(h)
    delta[:, site] = d.states[layer][:, site] - h[:, site]
    # Future tokens cannot alter the source edit.
    torch.testing.assert_close(d.states[layer][:, :site], h[:, :site], rtol=0, atol=2e-4)

    def suffix(x):
        return r.forward(layer, x)[0, start:]
    base = r.states[depth - 1][0, start:]
    endpoint = d.states[depth - 1][0, start:] - base
    target = suffix(h + delta) - base
    def intermediate():
        return partial_response(r, h + delta, layer, landmark, start)
    p = intermediate()
    mid = r.states[landmark][0, start:]
    z = d.states[landmark][0, start:] - mid
    def jvp(at):
        return torch.autograd.functional.jvp(suffix, at, delta)[1]
    tangent = jvp(h)
    midpoint = jvp(h + 0.5 * delta)
    readout = lambda x: model.lm_head(model.model.norm(x)).argmax(-1).tolist()
    assert readout(base) == answer
    exact = readout(base + target)
    check = {}
    if audit:
        torch.testing.assert_close(suffix(h), base, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(suffix(d.states[layer]), base + endpoint, rtol=1e-4, atol=1e-4)
        full = greedy(model, clean, stops, (layer, site, delta[:, site]))
        diff = lambda seq: next((i for i,(a,b) in enumerate(zip(answer, seq)) if a != b), None)
        assert diff(full) == diff(exact) and (answer != full) == (answer != exact)
        numeric = (suffix(h + .005 * delta) - suffix(h - .005 * delta)) / .01
        relative = float((numeric - tangent).norm() / tangent.norm().clamp_min(1e-8))
        if not np.isfinite(relative) or relative > .1:
            raise ArithmeticError(f"Unreliable finite difference check: {relative}")
        check = {"patched_generation": full, "relative_jvp_check": relative}
        if model.device.type == "cuda":
            # Marginal response cost, given cached clean trajectories; omit shared readout head.
            ops = {"exact": lambda: suffix(h + delta) - base,
                   "secant": lambda: secant_transport(intermediate(), z, endpoint, 1.0),
                   "derivative": lambda: jvp(h)}
            timings = {name:[] for name in ops}
            for repeat in range(5):
                names = list(ops)
                for name in names[repeat % 3:] + names[:repeat % 3]:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    ops[name]()
                    torch.cuda.synchronize()
                    timings[name].append(1000 * (time.perf_counter() - begin))
            check["median_ms"] = {k:float(np.median(v)) for k,v in timings.items()}
    arrays = dict(base=base, target=target, endpoint=endpoint, partial=p, direction=z,
                  tangent=tangent, midpoint=midpoint)
    return {k:v.cpu().detach().numpy() for k,v in arrays.items()}, dict(
        source=site, tokens=len(clean), block=layer, landmark=landmark, clean_generation=answer,
        ended=answer[-1] in stops, gold_along_clean=exact, **check)


def main():
    parser = argparse.ArgumentParser()
    for name in ("inputs", "manifest", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("checkpoint", choices=WEIGHTS)
    args = parser.parse_args()
    name = args.checkpoint
    path = args.inputs / name
    assert digest(path / "model.safetensors") == WEIGHTS[name]
    source = args.inputs / "train-v1.1.json"
    assert digest(source) == SQUAD_SHA
    articles = {a["title"]:a for a in json.loads(source.read_text())["data"]}
    tok = AutoTokenizer.from_pretrained(path)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True).eval().requires_grad_(False).cuda()
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos,list) else [eos])
    result = args.out / name
    result.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(json.loads(args.manifest.read_text())):
        # The non-Qwen checkpoint never supplies labeled calibration data.
        if name == "smol" and row["split"] == "fit":
            continue
        dst = result / (row["qid"] + ".npz")
        if dst.exists():
            continue
        para = articles[row["article"]]["paragraphs"][row["paragraph"]]
        question = next(q["question"] for q in para["qas"] if q["id"] == row["qid"])
        ids = get_ids(tok, row, para["context"], question)
        values, info = collect(model, *ids, stops, i < 2 or 48 <= i < 56)
        tmp = dst.with_suffix(".part.npz")
        np.savez(tmp, **values, info=json.dumps(info), manifest=digest(args.manifest))
        tmp.replace(dst)
        if i % 10 == 0:
            print(name, i, flush=True)


if __name__ == "__main__":
    main()
