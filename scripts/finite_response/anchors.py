"""Frozen two-anchor approximation of an isolated patch, benchmarked on paired questions."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect import WEIGHTS, greedy
from decode import normalize
from ilslens.finite_response import partial_response
from ilslens.suffix import DecoderSuffix
from sample import SQUAD_SHA, digest, get_ids

# Columns of the observed response basis. Coefficients are global scalars.
COLUMNS = {
    "no_change": (),
    "fit_donor": ("end",),
    "fit_clean_jvp": ("clean_jvp",),
    "fit_midpoint": ("midpoint",),
    "partial_donor_mix": ("end", "p"),
    "donor_landmark": ("end", "gap"),
    "donor_jvp": ("end", "donor_jvp"),
    "two_anchor": ("end", "donor_jvp", "clean_jvp"),
    "donor_midpoint": ("end", "donor_midpoint"),
    "two_anchor_midpoint": ("end", "donor_midpoint", "midpoint"),
}
FIXED = {
    "clean_jvp_raw": ("clean_jvp",),
    "donor_jvp_raw": ("end", "donor_jvp"),
    "raw_donor": ("end",),
    "donor_midpoint_raw": ("end", "donor_midpoint"),
}
ROLLOUT = ("no_change", "fit_donor", "donor_landmark", "donor_jvp", "two_anchor", "exact")


def basis(model, clean, donor, site, history, start, all_columns=True):
    """Both clean runs share the same continuation (only tokens in history)."""

    def replay(tokens):
        return DecoderSuffix(model, torch.tensor([tokens + list(history)], device=model.device))

    r, d = replay(clean), replay(donor)
    depth = len(r.states)
    source, mid, last = depth // 2 - 1, 3 * depth // 4 - 1, depth - 1
    h = r.states[source]
    k = d.states[source]
    delta = torch.zeros_like(h)
    delta[:, site] = k[:, site] - h[:, site]
    hybrid = h + delta
    complement = hybrid - k

    def tail(x):
        return r.forward(source, x)[0, start:]

    def jvp(at, move):
        return torch.autograd.functional.jvp(tail, at, move)[1]

    base = r.states[last][0, start:]
    end = d.states[last][0, start:] - base
    p = partial_response(r, hybrid, source, mid, start)
    z = (d.states[mid] - r.states[mid])[0, start:]
    cols = dict(end=end, clean_jvp=jvp(h, delta), donor_jvp=jvp(k, complement), gap=p - z, p=p)
    if all_columns:
        cols.update(
            midpoint=jvp(h + 0.5 * delta, delta), donor_midpoint=jvp(k + 0.5 * complement, complement)
        )
    # The exact suffix is used solely for evaluation.
    return base, cols, tail(hybrid) - base


def prediction(cols, weights, method):
    keys = COLUMNS.get(method, FIXED.get(method))
    return sum(
        (w * cols[k] for w, k in zip(weights[method], keys, strict=True)), torch.zeros_like(cols["end"])
    )


def fit(tokens):
    gram = {}
    rhs = {}
    for cols, target in tokens:
        for name, keys in COLUMNS.items():
            if keys:
                x = np.stack([cols[k][0] for k in keys], axis=-1).astype("float64")
                gram[name] = gram.get(name, 0) + x.T @ x
                rhs[name] = rhs.get(name, 0) + x.T @ target[0].astype("float64")
    weights = {
        name: np.linalg.lstsq(gram[name], rhs[name], rcond=1e-12)[0].tolist() if keys else []
        for name, keys in COLUMNS.items()
    }
    return {**weights, **{name: [1.0] * len(keys) for name, keys in FIXED.items()}}


def cases(rows, articles, tok):
    for row in rows:
        para = articles[row["article"]]["paragraphs"][row["paragraph"]]
        by_id = {q["id"]: q for q in para["qas"]}
        for kind, qid in (("edited_answer", row["qid"]), ("other_answer", row["other_qid"])):
            q = by_id[qid]
            yield (
                dict(article=row["article"], qid=qid, kind=kind),
                get_ids(tok, row, para["context"], q["question"]),
            )


def readout(model):
    return lambda vec: model.lm_head(model.model.norm(vec)).argmax(-1).tolist()


def evaluate(model, ids, stops, weights, tok):
    clean, donor, site = ids
    top = readout(model)
    original = greedy(model, clean, stops)
    base, cols, target = basis(model, *ids, original[:-1], len(clean) - 1)
    assert top(base) == original
    gold = top(base + target)
    methods = {}
    for name in weights:
        pred = prediction(cols, weights, name)
        tokens = top(base + pred)
        methods[name] = dict(
            sse=float((pred[0] - target[0]).double().square().sum()),
            first=tokens[0],
            changed=(tokens != original),
        )

    @lru_cache(maxsize=64)
    def next_tokens(history):
        base, cols, target = basis(model, clean, donor, site, history, len(clean) + len(history) - 1, False)
        return {
            k: top(base + (target if k == "exact" else prediction(cols, weights, k)))[0] for k in ROLLOUT
        }

    answers = {}
    for name in ROLLOUT:
        answer = []
        for _ in range(8):
            answer.append(next_tokens(tuple(answer))[name])
            if answer[-1] in stops:
                break
        answers[name] = answer
        if name != "exact":
            assert methods[name]["first"] == answer[0] and methods[name]["changed"] == (answer != original)
    assert answers["no_change"] == original
    assert answers["exact"][0] == gold[0] and (answers["exact"] != original) == (original != gold)
    norm = {k: normalize(tok.decode(v, skip_special_tokens=True)) for k, v in answers.items()}
    return dict(
        energy=float(target[0].double().square().sum()),
        methods=methods,
        tokens=answers,
        exact_ended=answers["exact"][-1] in stops,
        first=gold[0],
        changed=original != gold,
        text_changed=norm["exact"] != norm["no_change"],
        text_agreement={k: t == norm["exact"] for k, t in norm.items() if k != "exact"},
    )


def main():
    parser = argparse.ArgumentParser()
    for key in ("inputs", "manifest", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    train = [r for r in rows if r["split"] == "fit"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(train) == 32 and len(test) == 64 and len(set(r["article"] for r in rows)) == 96
    assert digest(args.inputs / "train-v1.1.json") == SQUAD_SHA
    articles = {r["title"]: r for r in json.loads((args.inputs / "train-v1.1.json").read_text())["data"]}
    args.out.mkdir(parents=True, exist_ok=True)

    def save(path, obj):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)

    torch.backends.cuda.matmul.allow_tf32 = False
    for name in ("qwen", "smol"):
        path = args.inputs / name
        assert digest(path / "model.safetensors") == WEIGHTS[name]
        model = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
            )
            .eval()
            .requires_grad_(False)
            .cuda()
        )
        tok = AutoTokenizer.from_pretrained(path)
        eos = model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos])
        if name == "qwen":

            def calibration(calibration_model):
                for _, ids in cases(train, articles, tok):
                    _, cols, target = basis(calibration_model, *ids, [], len(ids[0]) - 1)
                    yield {k: v.cpu().numpy() for k, v in cols.items()}, target.cpu().numpy()

            weights = fit(calibration(model))
            save(args.out / "weights.json", weights)
        records = []
        for meta, ids in cases(test, articles, tok):
            record = args.out / (name + "-" + meta["qid"] + ".json")
            if not record.exists():
                save(record, dict(**meta, **evaluate(model, ids, stops, weights, tok)))
            records.append(json.loads(record.read_text()))
            if len(records) % 32 == 0:
                print(name, len(records), flush=True)
        save(args.out / (name + "-rows.json"), records)
        del model


if __name__ == "__main__":
    main()
