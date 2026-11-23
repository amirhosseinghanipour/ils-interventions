"""Freeze prompt families across calibration, template holdout, and task transfer."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.component_routing.engine import Circuit
from scripts.component_routing.sampling import WEIGHTS

FILES = {
    "ioi-test.json": "030354654ea1b0b0147eb07f427be0f1ac71beed63a784a76d1581f52e722ebc",
    "colors-validation.jsonl": "1164862004710da129251833a7178454d8e2bcffbafb756fdfb623bca1fdecad",
    "colors-test.jsonl": "4a205a340887c60db7561c79b2e022f7dd3f1bd312a4c77eabd450af7680d545",
}
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
LAYERS = (11, 17)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def data(inputs, filename):
    path = inputs / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FILES[filename]
    return (
        read(path) if filename.endswith(".json") else [json.loads(s) for s in path.read_text().splitlines()]
    )


def encode(tok, row):
    ioi = "s2_io_flip_counterfactual" in row
    cf = row["s2_io_flip_counterfactual" if ioi else "answerPosition_counterfactual"]
    choices = row["choices"] if ioi else row["choices"]["label"]
    answers = [str(choices[r["answerKey"]]) for r in (row, cf)]
    ids = [tok.encode(r["prompt"], add_special_tokens=False) for r in (row, cf)]
    if len(ids[0]) != len(ids[1]) or ids[0] == ids[1]:
        return None
    labels = []
    for answer in answers:
        ext = [tok.encode(r["prompt"] + " " + answer, add_special_tokens=False) for r in (row, cf)]
        label = ext[0][-1]
        if any(v != x + [label] for x, v in zip(ids, ext)):
            return None
        labels.append(label)
    return (ids, labels) if labels[0] != labels[1] else None


def manifest(inputs):
    rng = np.random.default_rng(620411)
    toks = [AutoTokenizer.from_pretrained(inputs / name) for name in WEIGHTS]
    rows = data(inputs, "ioi-test.json")
    historical = ARTIFACTS / "finite-response" / "sites"
    old = {r["index"] for name in ("qwen", "smol") for r in read(historical / (name + "-ioi.json"))}
    used = {
        rows[i][k]["prompt"] if k else rows[i]["prompt"]
        for i in old
        for k in ("", "s2_io_flip_counterfactual")
    }
    templates = rng.permutation(sorted({r["template"] for r in rows})).tolist()
    families = defaultdict(list)
    for i in rng.permutation(len(rows)):
        row = rows[i]
        texts = {row["prompt"], row["s2_io_flip_counterfactual"]["prompt"]}
        if i not in old and used.isdisjoint(texts) and all(encode(t, row) for t in toks):
            families[row["template"]].append(int(i))
            used.update(texts)

    def pick(names, start, n):
        return [
            dict(file="ioi-test.json", index=i, family=family)
            for family in names
            for i in families[family][start : start + n]
        ]

    # The first three template families are available for panel selection and development.
    result = dict(
        panel=pick(templates[:3], 0, 3),
        development=pick(templates[:3], 3, 6),
        ioi=pick(templates[3:], 0, 8),
    )
    assert [len(v) for v in result.values()] == [9, 18, 48]
    rows = data(inputs, "colors-validation.jsonl")
    previous = {r["dataset_specific_id"] for r in data(inputs, "colors-test.jsonl")}
    colors = []
    for i in rng.permutation(len(rows)):
        r = rows[i]
        ident = r["dataset_specific_id"]
        if ident not in previous and all(encode(t, r) for t in toks):
            colors.append(dict(file="colors-validation.jsonl", index=int(i), family=ident))
            previous.add(ident)
        if len(colors) == 48:
            break
    assert len(colors) >= 32
    result["colors"] = colors
    return result


def model(inputs, name):
    path = inputs / name
    with (path / "model.safetensors").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHTS[name]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval().requires_grad_(False).cuda(), AutoTokenizer.from_pretrained(path)


def panel(inputs, name, refs):
    m, t = model(inputs, name)
    rows = data(inputs, "ioi-test.json")
    scores = []
    for ref in refs:
        ids, labels = encode(t, rows[ref["index"]])
        c = Circuit(m, ids, labels)
        scores.append(c.estimates("atp", 0).reshape(3, c.heads + 1)[:, : c.heads])
    means = np.abs(scores).mean(0)
    sites = []
    for layer in LAYERS:
        v = means[c.levels.index(layer)]
        sites.extend([[layer, int(h)] for h in sorted(np.argsort(-v, kind="stable")[:4])])
    return dict(sites=sites, mean_absolute_atp=means.tolist(), calibration_size=len(refs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("action", choices=["select", *WEIGHTS])
    a = p.parse_args()
    if a.action == "select":
        save(a.out / "manifest.json", manifest(a.inputs))
    else:
        save(
            a.out / ("panel-" + a.action + ".json"),
            panel(a.inputs, a.action, read(a.out / "manifest.json")["panel"]),
        )
